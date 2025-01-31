import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffuser_actor.utils.encoder import Encoder
from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.utils import (
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)
from action_flow.utils.geometry import se3_from_rot_pos
from action_flow.utils.encoder import SE3GraspFPSEncoder, FeaturePCDEncoder
from action_flow.utils.se3_grasp_vector_field import SE3GraspVectorFieldSelfAttn
from action_flow.utils.decoder import SE3PCDSelfAttnDecoder

from geo3dattn.policy.se3_flowmatching.common.se3_flowmatching import RectifiedLinearFlow

class SE3FlowMatchingSelfAttn(nn.Module):

    def __init__(self,
                 backbone="clip",
                 feature_res="res3",
                 embedding_dim=60,
                 fps_subsampling_factor=5,
                 gripper_loc_bounds=None,
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
                ):
        super().__init__()
        self._quaternion_format = quaternion_format
        self._relative = relative
        self._use_normals = use_normals
        self._rot_factor = rot_factor

        self.feature_pcd_encoder = FeaturePCDEncoder(
            backbone=backbone,
            feature_res=feature_res,
            embedding_dim=embedding_dim
            )
        encoder = SE3GraspFPSEncoder(
            dim_features=embedding_dim,
            gripper_depth=gripper_depth,
            nheads=8,
            fps_subsampling_factor=fps_subsampling_factor,
            nhist=nhist,
        )
        decoder = SE3PCDSelfAttnDecoder(embedding_dim=embedding_dim,
                                         x1_depth=1,
                                         s_depth=decoder_depth,
                                         x2_depth=1,
                                         dropout=decoder_dropout)
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
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

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
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
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
        feature_obs, pcd_obs, normal_obs = self.feature_pcd_encoder(rgb_obs, pcd_obs, normal_obs)
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
        self.model.set_context(*self.model.encode_obs(obs))

        # Predict the noise residual
        at_pose = self.vec_to_pose(at)
        input_data = {'act': at_pose, 'time': time}
        d_act, openess = self.model.forward_act(input_data)

        # Compute loss
        loss = (
                30 * F.l1_loss(d_act[...,:3], target[...,:3], reduction='mean')
                + self._rot_factor * 10 * F.l1_loss(d_act[..., 3:6], target[..., 3:6], reduction='mean')
        )
        if torch.numel(gt_openess) > 0:
            loss += F.binary_cross_entropy(openess, gt_openess)
        return loss
