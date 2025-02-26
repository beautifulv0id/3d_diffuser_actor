import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffuser_actor.utils.encoder import Encoder
from diffuser_actor.utils.layers import ParallelAttention
import pytorch3d.ops.sample_farthest_points as fps
from diffuser_actor.utils.utils import (
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix,
    measure_memory
)
from action_flow.utils.geometry import se3_from_rot_pos
from action_flow.utils.encoder import SE3GraspFPSEncoder, FeaturePCDEncoder
from action_flow.utils.se3_grasp_vector_field import SE3GraspVectorFieldSelfAttn
from action_flow.utils.decoder import SE3PCDEfficientSelfAttnDecoder

from geo3dattn.policy.se3_flowmatching.common.se3_flowmatching import RectifiedLinearFlow


class SE3FlowMatchingNURSASA(nn.Module):

    def __init__(self,
                 backbone="clip",
                 feature_res="res3",
                 embedding_dim=60,
                 fps_subsampling_factor=5,
                 workspace_bounds=None,
                 crop_workspace=False,
                 max_workspace_points=4000,
                 quaternion_format='xyzw',
                 diffusion_timesteps=100,
                 nhist=3,
                 relative=False,
                 scaling_factor=3.0,
                 rot_factor=2.0,
                 use_normals=False,
                 gripper_depth=2,
                 decoder_depth=2,
                 decoder_dropout=0.2,
                 distance_scale=1.0,
                 use_adaln=False,
                 gripper_history_as_points=False,
                 feature_type='sinusoid',
                 use_center_distance=True,
                 use_center_projection=True,
                 use_vector_projection=True,
                 add_center=True,
                 point_embedding_dim=120,
                 ):
        super().__init__()
        self._quaternion_format = quaternion_format
        self._relative = relative
        self._use_normals = use_normals
        self._rot_factor = rot_factor
        self._crop_workspace = crop_workspace
        self._max_workspace_points = max_workspace_points

        self.feature_pcd_encoder = FeaturePCDEncoder(
            backbone=backbone,
            feature_res=feature_res,
            embedding_dim=embedding_dim
            )
        encoder = SE3GraspFPSEncoder(
            dim_features=embedding_dim,
            gripper_depth=gripper_depth,
            nheads=8,
            n_steps_inf=diffusion_timesteps,
            nhist=nhist,
            gripper_history_as_points=gripper_history_as_points,
            feature_type=feature_type,
            use_adaln=use_adaln,
            use_center_distance=use_center_distance,
            use_center_projection=use_center_projection,
            use_vector_projection=use_vector_projection,
            add_center=add_center,
            fps_subsampling_factor=fps_subsampling_factor
        )
        decoder = SE3PCDEfficientSelfAttnDecoder(embedding_dim=embedding_dim,
                                         x1_depth=1,
                                         s_depth=decoder_depth,
                                         x2_depth=1,
                                         dropout=decoder_dropout,
                                         distance_scale=distance_scale,
                                         use_adaln=use_adaln,
                                         use_center_distance=use_center_distance,
                                         use_center_projection=use_center_projection,
                                         use_vector_projection=use_vector_projection,
                                         add_center=add_center,
                                         point_embedding_dim=point_embedding_dim
                                          )

        self.model = SE3GraspVectorFieldSelfAttn(
            encoder=encoder, 
            decoder=decoder, 
            latent_dim=embedding_dim)

        ## Flow Model ##
        self.scaling_factor = torch.tensor(scaling_factor, requires_grad=False)
        self.flow = RectifiedLinearFlow(n_action_steps=1, num_steps=diffusion_timesteps)
        std = torch.Tensor([1.0, 1.0, 1.0, torch.pi, torch.pi, torch.pi])[None, ...]
        mean = torch.Tensor([0, 0, 0, 0, 0, 0])[None, ...]
        self.flow.set_mean_std(mean, std)


        self.n_steps = diffusion_timesteps
        self.workspace_bounds = nn.Parameter(workspace_bounds, requires_grad=False)

        self.fit_embedding_parameters(point_embedding_dim)

    @torch.no_grad()
    def fit_embedding_parameters(self, point_embedding_dim):
        fit_data = torch.rand(point_embedding_dim, 3) * 2 - 1
        apply_to_module(self, NURSATransformer, lambda x: x.fit_embedding(fit_data))
        apply_to_module(self, NURSATransformerEncoder, lambda x: x.fit_embedding(fit_data))

    # ========= utils  ============
    def vec_to_pose(self, vec):
        p, r = self.flow._vector_to_pose(vec)
        H = torch.eye(4)[None, None, ...].repeat(vec.shape[0], vec.shape[1], 1, 1).to(vec.device)
        H[:, :, :3, -1] = p
        H[:, :, :3, :3] = r
        return H

    def pose_to_vec(self, H):
        p, r = H[:, :, :3, -1], H[:, :, :3, :3]
        vec = self.flow._pose_to_vector(p, r)
        return vec

    def normalize_pos(self, pos):
        pos_min = self.workspace_bounds[0].float().to(pos.device)
        pos_max = self.workspace_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.workspace_bounds[0].float().to(pos.device)
        pos_max = self.workspace_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min


    def conditional_sample(self, obs):
        B = obs["pcd"].shape[0]
        self.model.set_context(*self.model.encode_obs(obs))
        # Iterative denoising
        with torch.no_grad():
            at = self.flow.generate_random_initial_pose(B)
            for s in range(0, self.flow.num_steps):
                step = s * torch.ones_like(at[:, 0, 0])
                at_H = self.vec_to_pose(at)
                d_act, gripper_open = self.model.forward_act({
                    'act': at_H,
                    'time':step})
                at = self.flow.step(at, d_act, s)

        trajectory = self.vec_to_pose(at)

        return trajectory, gripper_open

    def compute_trajectory(self, obs):
        # Sample
        trajectory, gripper_open = self.conditional_sample(obs)

        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory, gripper_open)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])

        return trajectory

    def convert_rot(self, signal):
        signal = signal.clone()
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        # The following code expects wxyz quaternion format!
        if self._quaternion_format == 'xyzw':
            signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
        rot = quaternion_to_matrix(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        H = se3_from_rot_pos(rot, signal[..., :3])
        return H, res

    def unconvert_rot(self, H, res=None):
        quat = matrix_to_quaternion(H[..., :3, :3])
        pos = H[..., :3, 3]
        signal = torch.cat([pos, quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        # The above code handled wxyz quaternion format!
        if self._quaternion_format == 'xyzw':
            signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal


    def convert2rel(self, pcd, curr_gripper):
        """Convert coordinate system relaative to current gripper."""
        center = curr_gripper[:, -1, :3]  # (batch_size, 3)
        bs = center.shape[0]
        pcd = pcd - center.view(bs, 1, 3, 1, 1)
        curr_gripper = curr_gripper.clone()
        curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
        return pcd, curr_gripper

    @torch.no_grad()
    def crop_and_resample_batch(self, pcd, rgb):
        """
        Crop point clouds to a 3D workspace and resample to a fixed size.
        
        Args:
            pcd (torch.Tensor): Input batch of point clouds. Shape: (B, N, 3)
            rgb (torch.Tensor): Input batch of RGB images. Shape: (B, C, H, W)
        Returns:
            torch.Tensor: Processed point clouds with shape (B, max_points, 3)
        """
        device = pcd.device
        B, N, _ = pcd.shape

        min_bound = self.workspace_bounds[0].float()
        max_bound = self.workspace_bounds[1].float()
        
        # Create mask for pcd within workspace
        mask = (pcd >= min_bound) & (pcd <= max_bound)
        mask = mask.all(dim=-1)  # (B, N)
        
        pcds = []
        rgbs = []
        
        for i in range(B):
            # Extract valid points for this cloud
            valid_pcds = pcd[i][mask[i]]  # (K, 3)
            valid_rgbs = rgb[i][mask[i]]  
            K = valid_pcds.size(0)
            if K == 0:
                raise ValueError(f"All points filtered in cloud {i}. Consider checking bounds or input data.")
            if K >= self._max_workspace_points:
                indices = torch.randperm(K, device=device)[:self._max_workspace_points]
            else:
                indices = torch.randint(0, K, (self._max_workspace_points,), device=device)
            
            # Select points and maintain gradient flow
            sampled_pcds = valid_pcds[indices]
            sampled_rgbs = valid_rgbs[indices]
            pcds.append(sampled_pcds)
            rgbs.append(sampled_rgbs)
        
        pcds = torch.stack(pcds, dim=0)
        rgbs = torch.stack(rgbs, dim=0)
        return pcds, rgbs

    def forward(
        self,
        gt_trajectory,
        rgb_obs,
        pcd_obs,
        normal_obs,
        instruction,
        curr_gripper,
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            timestep: (B, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        # feature_obs, pcd_obs, normal_obs = self.feature_pcd_encoder(rgb_obs, pcd_obs, normal_obs)
        feature_obs, pcd_obs, normal_obs = measure_memory(self.feature_pcd_encoder.forward, rgb_obs, pcd_obs, normal_obs)
        if self._crop_workspace:
            pcd_obs, feature_obs = self.crop_and_resample_batch(pcd_obs, feature_obs)
        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = self.normalize_pos(pcd_obs[..., :3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        if gt_trajectory is not None:
            gt_trajectory = gt_trajectory.clone()
            gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])

        # Convert rotation parametrization
        curr_gripper, _ = self.convert_rot(curr_gripper)
        if gt_trajectory is not None:
            gt_trajectory, _ = self.convert_rot(gt_trajectory)

        obs = {
            'pcd': pcd_obs,
            'normals': normal_obs,
            'current_gripper': curr_gripper,
            'pcd_features': feature_obs,
            'instruction': instruction
        }        

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(obs)

        # Prepare inputs
        batch_size = pcd_obs.shape[0]
        device, dtype = pcd_obs.device, pcd_obs.dtype
        act_vector = self.flow._pose_to_vector(gt_trajectory[...,:3, -1], gt_trajectory[...,:3, :3])

        # 2. Compute Flow Matching Variables
        a1 = act_vector
        a0 = self.flow.generate_random_initial_pose(batch_size)
        time = torch.randint(0, self.flow.num_steps, (batch_size,)).to(device=device, dtype=dtype)

        at = self.flow.flow_at_t(a0, a1, time)
        target = self.flow.vector_field_at_t(a0, a1, at, time)

        ## 3. Set Context
        self.model.set_context(*measure_memory(self.model.encode_obs, obs))


        # Predict the noise residual
        at_pose = self.vec_to_pose(at)
        input_data = {'act': at_pose, 'time': time}
        d_act, openess = measure_memory(self.model.forward_act, input_data)

        # Compute loss
        loss = (
                30 * F.l1_loss(d_act[...,:3], target[...,:3], reduction='mean')
                + self._rot_factor * 10 * F.l1_loss(d_act[..., 3:6], target[..., 3:6], reduction='mean')
        )
        if torch.numel(gt_openess) > 0:
            loss += F.binary_cross_entropy(openess, gt_openess)
        return loss
