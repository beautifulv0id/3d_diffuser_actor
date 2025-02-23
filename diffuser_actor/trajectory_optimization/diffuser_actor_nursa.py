import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from geo3dattn.model.nursa_transformer.ursa_transformer import NURSATransformer, NURSATransformerEncoder
from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformer
from diffuser_actor.utils.encoder_ursa import EncoderURSA
from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.position_encodings import (
    SinusoidalPosEmb
)
from diffuser_actor.utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix,
    merge_geometric_args,
    measure_memory
)
from utils.common_utils import apply_to_module

class DiffuserActorNURSA(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 workspace_bounds=None,
                 crop_workspace=False,
                 max_workspace_points=4000,
                 rotation_parametrization='6D',
                 quaternion_format='xyzw',
                 diffusion_timesteps=100,
                 nhist=3,
                 relative=False,
                 lang_enhanced=False,
                 history_as_point=True,
                 point_embedding_dim=1000):
        super().__init__()
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self._relative = relative
        self.use_instruction = use_instruction
        self.history_as_point = history_as_point
        self._crop_workspace = crop_workspace
        self._max_workspace_points = max_workspace_points

        self.encoder = EncoderURSA(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            nhist=nhist,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            history_as_point=history_as_point,
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
            lang_enhanced=lang_enhanced,
            point_embedding_dim=point_embedding_dim
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        self.n_steps = diffusion_timesteps
        self.workspace_bounds = nn.Parameter(workspace_bounds, requires_grad=False)

        self.fit_embedding_parameters(point_embedding_dim)

    @torch.no_grad()
    def fit_embedding_parameters(self, point_embedding_dim):
        fit_data = torch.rand(point_embedding_dim, 3) * 2 - 1
        apply_to_module(self, NURSATransformer, lambda x: x.fit_embedding(fit_data))
        apply_to_module(self, NURSATransformerEncoder, lambda x: x.fit_embedding(fit_data))

    def encode_inputs(self, context_feats, context, instruction,
                      curr_gripper):
        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats = self.encoder.encode_instruction(instruction)

        # Cross-attention vision to language
        if self.use_instruction:
            # Attention from vision to language
            context_feats = self.encoder.vision_language_attention(
                context_feats, instr_feats
            )

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats,
            context
        )

        return (
            context_feats, context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            curr_gripper,  # current gripper pose
            fps_feats, fps_pos  # sampled visual features
        )
    
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

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            curr_gripper,
            fps_feats,
            fps_pos
        ) = fixed_inputs

        pos = trajectory[..., :3]
        rot = compute_rotation_matrix_from_ortho6d(trajectory[..., 3:9].view(-1, 6)).reshape(-1, pos.shape[1], 3, 3)
        trajectory_geom_args = {
            "centers": pos,
            "vectors": rot
        }
        context_geom_args = {
            "centers": context,
            "vectors": torch.zeros((3, 3))[None, None, :, :].repeat(
                context.shape[0], context.shape[1], 1, 1
            ).to(context.device)
        }
        pos = curr_gripper[..., :3]
        if self.history_as_point:
            rot = torch.zeros((3, 3))[None, None, :, :].repeat(
                curr_gripper.shape[0], curr_gripper.shape[1], 1, 1
            ).to(curr_gripper.device
            )
        else:
            rot = compute_rotation_matrix_from_ortho6d(curr_gripper[..., 3:9].view(-1, 6)).reshape(-1, pos.shape[1], 3, 3)
        gripper_geom_args = {
            "centers": pos,
            "vectors": rot
        }
        fps_geom_args = {
            "centers": fps_pos,
            "vectors": torch.zeros((3, 3))[None, None, :, :].repeat(
                fps_pos.shape[0], fps_pos.shape[1], 1, 1
            ).to(fps_pos.device)
        }
        return self.prediction_head(
            trajectory_geom_args,
            timestep,
            context_feats=context_feats,
            context_geom_args=context_geom_args,
            fps_feats=fps_feats,
            fps_geom_args=fps_geom_args,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            gripper_geom_args=gripper_geom_args
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.position_noise_scheduler.timesteps[0])
        noise_pos = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
        trajectory = torch.where(
            condition_mask, noisy_condition_data, noise
        )

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs
            )
            out = out[-1]  # keep only last layer's output
            pos = self.position_noise_scheduler.step(
                out[..., :3], t, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_noise_scheduler.step(
                out[..., 3:9], t, trajectory[..., 3:9]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        trajectory = torch.cat((trajectory, out[..., 9:]), -1)

        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        context_feats,
        context,
        instruction,
        curr_gripper
    ):
        # Normalize all pos
        context = context.clone()
        curr_gripper = curr_gripper.clone()
        context = self.normalize_pos(context)
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            context_feats, context, instruction, curr_gripper
        )

        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=context_feats.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        return trajectory

    def normalize_pos(self, pos):
        pos_min = self.workspace_bounds[0].float().to(pos.device)
        pos_max = self.workspace_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.workspace_bounds[0].float().to(pos.device)
        pos_max = self.workspace_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == '6D':
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization == '6D':
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            # The above code handled wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal
    
    def trajectory_to_se3(self, trajectory):
        pos = trajectory[..., :3]
        rot = trajectory[..., 3:7]
        if self._quaternion_format == 'xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        rot = quaternion_to_matrix(rot)
        return pos, rot

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
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            trajectory_mask: (B, trajectory_length)
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
        rgb_feats_pyramid, pcd_pyramid = measure_memory(self.encoder.encode_images,
            rgb_obs, pcd_obs
        )
        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0],
            "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0]

        if self._crop_workspace:
            context, context_feats = self.crop_and_resample_batch(
                context, 
                context_feats
            )

        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                context_feats,
                context,
                instruction,
                curr_gripper
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        context = context.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        context = self.normalize_pos(context)
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = measure_memory(self.encode_inputs,
            context_feats, context, instruction, curr_gripper
        )

        print("fps size", fixed_inputs[5].shape)

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3],
            timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9],
            timesteps
        )
        noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
        assert not cond_mask.any()

        # Predict the noise residual
        pred = measure_memory(self.policy_forward_pass,
            noisy_trajectory, timesteps, fixed_inputs
        )

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            loss = (
                30 * F.l1_loss(trans, noise[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot, noise[..., 3:9], reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 9:]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss


class DiffusionHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instruction=False,
                 rotation_parametrization='quat',
                 nhist=3,
                 lang_enhanced=False,
                 point_embedding_dim=1000):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        if '6D' in rotation_parametrization:
            rotation_dim = 6  # continuous 6D
        else:
            rotation_dim = 4  # quaternion

        # Encoders
        self.trajectory_feats = nn.Parameter(torch.randn(1, embedding_dim))
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim*nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = URSATransformer(d_model=embedding_dim, nhead=num_attn_heads, num_layers=2, use_adaln=True)
        self.self_attn = NURSATransformerEncoder(d_model=embedding_dim, nhead=num_attn_heads, num_layers=4, use_adaln=True, point_embedding_dim=point_embedding_dim)
        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn = NURSATransformerEncoder(d_model=embedding_dim, nhead=num_attn_heads, num_layers=2, use_adaln=True, point_embedding_dim=point_embedding_dim)
        else:  # interleave cross-attention to language
            raise NotImplementedError
        
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = NURSATransformerEncoder(d_model=embedding_dim, nhead=num_attn_heads, num_layers=2, use_adaln=True, point_embedding_dim=point_embedding_dim)
        else:  # interleave cross-attention to language
            raise NotImplementedError
        
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, trajectory_geom_args, timestep,
                context_feats, context_geom_args, 
                fps_feats, fps_geom_args,
                instr_feats, adaln_gripper_feats,
                gripper_geom_args):
        """
        Arguments:
            trajectory: 
            timestep: (B, 1)
            context_feats: (B, N, F)
            context_geom_args: 
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
        """
        # Trajectory features
        traj_feats = self.trajectory_feats[None].repeat(len(trajectory_geom_args["centers"]), 1, 1)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            trajectory_geom_args, traj_feats,
            context_geom_args, context_feats,
            fps_feats, fps_geom_args,
            timestep, adaln_gripper_feats,
            gripper_geom_args
        )
        return [torch.cat((pos_pred, rot_pred, openess_pred), -1)]

    def prediction_head(self,
                        trajectory_geometric_args, trajectory_features,
                        context_pcd_geometric_args, context_pcd_features,
                        context_fps_features, context_fps_geometric_args,
                        timesteps, curr_gripper_features,
                        gripper_geometric_args):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            trajectory_geometric_args: 
            gripper_features: A tensor of shape (N, B, F)
            context_pcd_geometric_args: 
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        context_features = torch.cat((context_pcd_features, curr_gripper_features), 1)
        geometric_args = {
            'query': trajectory_geometric_args,
            'key': merge_geometric_args(context_pcd_geometric_args, gripper_geometric_args),
        }

        # Cross attention from gripper to full context
        trajectory_features = measure_memory(self.cross_attn.forward,tgt = trajectory_features, 
                                              memory = context_features, 
                                              geometric_args = geometric_args, diff_ts = time_embs)
        # Self-attention
        features = torch.cat((trajectory_features, context_fps_features, curr_gripper_features), 1)
        geometric_args = {
            'query': merge_geometric_args(trajectory_geometric_args, context_fps_geometric_args, gripper_geometric_args)
        }
        features = measure_memory(self.self_attn.forward,tgt = features, geometric_args = geometric_args, diff_ts = time_embs)

        n_act = trajectory_features.shape[1]


        # Rotation head
        rotation = self.predict_rot(
            features, geometric_args, time_embs, n_act
        )
        # Position head
        position, position_features = self.predict_pos(
            features, geometric_args, time_embs, n_act
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return position, rotation, openess

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep).unsqueeze(1)
        curr_gripper_feats = curr_gripper_features.flatten(1, 2)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_feats).unsqueeze(1)
        return time_feats + curr_gripper_feats

    def predict_pos(self, features, geometric_args, time_embs, n_act):
        position_features = measure_memory(self.position_self_attn.forward,tgt = features,
                                                     geometric_args = geometric_args,
                                                     diff_ts = time_embs)
        position_features = position_features[:,:n_act]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, geometric_args, time_embs, n_act):
        rotation_features = measure_memory(self.rotation_self_attn.forward,tgt = features,
                                                     geometric_args = geometric_args,
                                                     diff_ts = time_embs)
        rotation_features = rotation_features[:,:n_act]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
