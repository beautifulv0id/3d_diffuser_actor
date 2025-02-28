import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch3d.ops.sample_farthest_points as fps
from pytorch3d.ops.knn import knn_points
import einops
from geo3dattn.model.common.module_attr_mixin import ModuleAttrMixin
from geo3dattn.encoder.superpoint_encoder.super_point_encoder import SuperPointEncoder
from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformer
from geo3dattn.encoder.common.position_encoder import PositionalEncoding
from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.resnet import load_resnet50, load_resnet18
from diffuser_actor.utils.clip import load_clip
from geo3dattn.model.ipa_transformer.ipa_transformer import IPATransformer


def load_model(output_dim, nhead, num_layers, modeltype='ursa'):
    if modeltype == 'ursa':
        from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformer
        return URSATransformer(output_dim, nhead=nhead, num_layers=num_layers)
    elif modeltype=='rope3d':
        from geo3dattn.model.rope3d_transformer.rope_3d_transformer import RoPE3DTransformer
        return RoPE3DTransformer(output_dim, nhead=nhead, num_layers=num_layers)
    elif modeltype=='direct':
        from geo3dattn.model.direct_transformer.direct_transformer import DirectTransformer
        return DirectTransformer(output_dim, nhead=nhead, num_layers=num_layers)
    elif modeltype=='point_distance':
        from geo3dattn.model.direct_transformer.direct_transformer import DirectTransformer
        return DirectTransformer(output_dim, nhead=nhead, num_layers=num_layers)


class SuperPointEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=256, fps_subsampling_factor=14,
                 nheads = 4, num_layers = 2, knn = 10, model_type='ursa'):
        super(SuperPointEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fps_subsampling_factor = fps_subsampling_factor
        self.knn = knn

        self.linear = nn.Linear(input_dim, output_dim)

        self.model = load_model(output_dim, nheads, num_layers, modeltype=model_type)

    def forward(self, tgt, geometric_args):

        x = tgt
        pcd_x = geometric_args['query']['centers']
        pcd_v = geometric_args['query']['vectors']

        n_points_out = x.shape[1] // self.fps_subsampling_factor
        ## Get n points via FPS ##
        out_pcd_x, out_indices = fps(pcd_x, K=n_points_out)


        ## Find KNN to Output Points ##
        k = self.knn
        out = knn_points(out_pcd_x, pcd_x, K=k)
        ## Given index (B,Q,K) with B batch and pcd (B,N,3), select (B,Q,K,3) ##
        knn_pcd = torch.gather(pcd_x.unsqueeze(1).expand(-1, n_points_out, -1, -1), 2, out[1].unsqueeze(-1).expand(-1, -1, -1, 3))

        ## Prepare data
        ## For the computation we move each SuperPoint to the batch and transform the KNN to the key tokens
        query_f = torch.gather(x, 1, out_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        _query_f = query_f.reshape(-1, query_f.shape[-1])[:,None,:]
        query_c = out_pcd_x
        _query_c = query_c.reshape(-1, query_c.shape[-1])[:,None,:]
        query_v = torch.gather(pcd_v, 1, out_indices[...,None, None].expand(-1, -1, pcd_v.shape[-2], pcd_v.shape[-1]))
        _query_v = query_v.reshape(-1, query_v.shape[-2], query_v.shape[-1])[:,None,...]
        _query_geo = {'centers': _query_c, 'vectors': _query_v}

        key_f = torch.gather(x.unsqueeze(1).expand(-1, n_points_out, -1, -1), 2, out[1].unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        _key_f = key_f.reshape(-1, key_f.shape[2], key_f.shape[3])
        key_c = knn_pcd
        _key_c = key_c.reshape(-1, key_c.shape[2], key_c.shape[3])
        key_v = torch.gather(pcd_v.unsqueeze(1).expand(-1, n_points_out, -1, -1, -1), 2, out[1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, pcd_v.shape[-2], pcd_v.shape[-1]))
        _key_v = key_v.reshape(-1, key_v.shape[2], key_v.shape[3], key_v.shape[4])
        _key_geo = {'centers': _key_c, 'vectors': _key_v}

        ## Compute the attention
        _geo = {'query': _query_geo, 'key': _key_geo}
        _key_f = self.linear(_key_f)
        _query_f = self.linear(_query_f)
        out_query_f = self.model(_query_f, _key_f, geometric_args=_geo)

        ## Reshape the output
        out_query_f = out_query_f.reshape(-1, n_points_out, out_query_f.shape[-1])
        return out_query_f, {'query':{'centers': query_c, 'vectors': query_v}}


class SE3GraspPointCloudEncoder(ModuleAttrMixin):
    def __init__(self, dim_features=128,
                    gripper_depth=3,
                    nheads=4,
                    n_steps_inf=50,
                    nhist=3,
                    num_vis_ins_attn_layers=2,
                    gripper_history_as_points=False,
                    feature_type='sinusoid',
                    use_adaln=True,
                    use_center_distance=True,
                    use_center_projection=True,
                    use_vector_projection=True,
                    add_center=True
                    ):

        super(SE3GraspPointCloudEncoder, self).__init__()

        ## Learnable observation features (Data in Acronym is purely geometrical, no semantics involved)
        self.gripper_features = nn.Parameter(torch.randn(nhist,dim_features))

        ## Learnable action features
        self.action_features = nn.Parameter(torch.randn(1, dim_features))

        ## Gripper History Encoder ##
        self.gripper_decoder = URSATransformer(dim_features, 
                                                nhead=nheads, 
                                                num_layers=gripper_depth, 
                                                feature_type=feature_type,
                                                use_adaln=use_adaln,
                                                use_center_distance=use_center_distance,
                                                use_center_projection=use_center_projection,
                                                use_vector_projection=use_vector_projection,
                                                add_center=add_center)

        ## Instruction Encoder ##
        self.instruction_encoder = nn.Linear(512, dim_features)

        # Attention from vision to language
        self.vl_attention = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=dim_features, n_heads=nheads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )

        # Attention from action to language
        self.al_attention = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=dim_features, n_heads=nheads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )

        ## Time Encoder ##
        self.time_encoder = nn.Sequential(
            PositionalEncoding(n_positions=dim_features, max_len=n_steps_inf),
               nn.Linear(dim_features * 2, 4 * dim_features),
                nn.LayerNorm(4 * dim_features),
                nn.GELU(),
                nn.Linear(4 * dim_features, dim_features)
            )

        self.obs_merger = nn.Sequential(
                nn.Linear(2 * dim_features, dim_features),
                nn.LayerNorm(dim_features),
                nn.GELU()
            )

        self.act_merger = nn.Sequential(
                nn.Linear(2 * dim_features, dim_features),
                nn.LayerNorm(dim_features),
                nn.GELU()
            )
        
        self.gripper_history_as_points = gripper_history_as_points
        
    def forward(self, x):
        obs_points, obs_features = self.encode_obs(x['obs'])
        act_points, act_features = self.encode_act(x['act'])
        time_emb = self.encode_time(x['time'])
        obs_f, act_f = self.combine_time(obs_features, act_features, time_emb)
        return obs_points, obs_f, act_points, act_f

    def encode_time(self, time):
        time_emb = self.time_encoder(time)
        return time_emb[:,None,:]
    
    def encode_instruction(self, instruction):
        return self.instruction_encoder(instruction)

    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention(
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats

    def action_language_attention(self, feats, instr_feats):
        feats, _ = self.al_attention(
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats

    def encode_gripper(self, gripper, obs_features, obs_geom_args):
        gripper_features = self.gripper_features[None,...].repeat(gripper.size(0), 1, 1)
        if self.gripper_history_as_points:
            vectors = torch.zeros((3,3))[None,None,:,:].repeat(gripper.shape[0], gripper.shape[1], 1, 1).to(gripper.device)
        else:
            vectors = gripper[:,:,:3,:3]
        geom_args = {
                    'query': 
                        {'centers': gripper[:,:,:3,-1], 'vectors': vectors},
                    'key': obs_geom_args
                     }
        gripper_features = self.gripper_decoder(tgt=gripper_features, memory=obs_features, geometric_args=geom_args)
        return gripper_features

    def encode_obs(self, obs):
        pcd = obs['pcd']
        pcd_features = obs['pcd_features']
        current_gripper = obs['current_gripper']
        instruction = obs.get('instruction', None)
        normals = obs.get('normals', None)

        batch = pcd.shape[0]
        device = pcd.device

        # encode pcd
        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
        if normals is not None:
            vectors[...,:3,0] = normals
        geom_args = {'centers': pcd, 'vectors': vectors}

        obs_features = pcd_features
        
        # add gripper features
        gripper_features = self.encode_gripper(current_gripper, obs_features, geom_args)        
        obs_features = torch.cat((obs_features, gripper_features), dim=1)
        
        if self.gripper_history_as_points:
            gripper_vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, current_gripper.shape[1], 1, 1).to(device)
        else:
            gripper_vectors = current_gripper[:,:,:3,:3]
        geom_args['vectors'] = torch.cat((geom_args["vectors"], gripper_vectors), dim=1)
        geom_args['centers'] = torch.cat((geom_args["centers"], current_gripper[:,:,:3,-1]), dim=1)

        if instruction is not None:
            instr_features = self.encode_instruction(instruction)
            obs_features = self.vision_language_attention(obs_features, instr_features)
        else:
            instr_features = None

        return geom_args, obs_features, instr_features

    def encode_act(self, act):
        act_points = {'centers': act[..., :3, -1], 'vectors': act[..., :3, :3]}
        act_features = self.action_features[None,...].repeat(act.shape[0], 1, 1)

        return act_points, act_features

    def combine_time(self, obs_f, act_f, time_emb):
        obs_time_f = self.obs_combine_time(obs_f, time_emb)
        act_time_f = self.act_combine_time(act_f, time_emb)

        return obs_time_f, act_time_f

    def obs_combine_time(self, obs_f, time_emb):
        obs_time_f = torch.cat((obs_f, time_emb.repeat(1, obs_f.shape[1],1)), dim=-1)
        return self.obs_merger(obs_time_f)

    def act_combine_time(self, act_f, time_emb):
        act_time_f = torch.cat((act_f, time_emb.repeat(1, act_f.shape[1],1)), dim=-1)
        return self.act_merger(act_time_f)

class SE3GraspPointCloudSuperEncoder(SE3GraspPointCloudEncoder):
    def __init__(self, dim_features=128, gripper_depth=3, nheads=4, n_steps_inf=50, fps_subsampling_factor=5, nhist=3, dim_pcd_features=64, spe_depth=3, num_vis_ins_attn_layers=2):
        super(SE3GraspPointCloudSuperEncoder, self).__init__(dim_features, gripper_depth, nheads, n_steps_inf, nhist, num_vis_ins_attn_layers)

        ## Pointcloud Encoder ##
        self.obs_encoder = SuperPointEncoder(input_dim=dim_pcd_features, output_dim=dim_features, fps_subsampling_factor=fps_subsampling_factor,
                                             nheads=nheads, num_layers=spe_depth)

    def encode_obs(self, obs):
        pcd = obs['pcd']
        pcd_features = obs['pcd_features']
        current_gripper = obs['current_gripper']
        normals = obs.get('normals', None)
        instruction = obs.get('instruction', None)

        batch = pcd.shape[0]
        device = pcd.device

        # encode pcd
        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
        if normals is not None:
            vectors[...,:3,0] = normals

        obs_points = {'centers': pcd, 'vectors': vectors}

        obs_features = pcd_features
        if self.obs_encoder is not None:
            obs_features, obs_geo = self.obs_encoder(tgt=obs_features, geometric_args={'query':obs_points})
            obs_points = obs_geo['query']
        
        # add gripper features
        gripper_features = self.encode_gripper(current_gripper, obs_features, obs_points)        
        obs_features = torch.cat((obs_features, gripper_features), dim=1)
        obs_points['vectors'] = torch.cat((obs_points["vectors"], current_gripper[:,:,:3,:3]), dim=1)
        obs_points['centers'] = torch.cat((obs_points["centers"], current_gripper[:,:,:3,-1]), dim=1)

        if instruction is not None:
            instr_features = self.encode_instruction(instruction)
            obs_features = self.vision_language_attention(obs_features, instr_features)
        else:
            instr_features = None

        return obs_points, obs_features, instr_features
    
class SE3GraspFPSSuperEncoder(SE3GraspPointCloudSuperEncoder):
    def __init__(self, dim_features=128, depth=3, nheads=4, n_steps_inf=50, fps_subsampling_factor=5, nhist=3, dim_pcd_features=64):
        super(SE3GraspFPSSuperEncoder, self).__init__(dim_features, depth, nheads, n_steps_inf, fps_subsampling_factor, nhist, dim_pcd_features)
        input_dim = dim_pcd_features
        output_dim = dim_features
        self.linear = nn.Linear(input_dim, output_dim)

    def encode_obs(self, obs):
        obs_pcd_x, obs_pcd_f, inst_f = super().encode_obs(obs)
        pcd, obs_f = obs['pcd'], obs['pcd_features']
        normals = obs.get('normals', None)
        batch = pcd.shape[0]
        device = pcd.device
        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
        if normals is not None:
            vectors[...,:3,0] = normals
        obs_x = {'centers': pcd, 'vectors': vectors}
        obs_f = self.linear(obs_f)
        return obs_x, obs_f, obs_pcd_x, obs_pcd_f, inst_f

class SE3GraspFPSEncoder(SE3GraspPointCloudEncoder):
    def __init__(self, 
                    dim_features=128, 
                    gripper_depth=3, 
                    nheads=4, 
                    n_steps_inf=50, 
                    fps_subsampling_factor=5, 
                    nhist=3, 
                    gripper_history_as_points=False,
                    feature_type='sinusoid',
                    use_adaln=True,
                    use_center_distance=True,
                    use_center_projection=True,
                    use_vector_projection=True,
                    add_center=True
                    ):
        super(SE3GraspFPSEncoder, self).__init__(dim_features, gripper_depth, nheads, n_steps_inf, nhist, 
                                                gripper_history_as_points=gripper_history_as_points,
                                                feature_type=feature_type,
                                                use_adaln=use_adaln,
                                                use_center_distance=use_center_distance,
                                                use_center_projection=use_center_projection,
                                                use_vector_projection=use_vector_projection,
                                                add_center=add_center)
        self.fps_subsampling_factor = fps_subsampling_factor
        output_dim = dim_features
        self.linear = nn.Linear(dim_features, output_dim)

    def compute_fps(self, pcd, pcd_features, normals=None):
        n_points_out = pcd_features.shape[1] // self.fps_subsampling_factor
        ## Get n points via FPS ##
        out_pcd, out_indices = fps(pcd, K=n_points_out)
        obs_vectors_fps = torch.zeros((3,3))[None,None,:,:].repeat(out_pcd.shape[0], out_pcd.shape[1], 1, 1).to(pcd.device)
        if normals is not None:
            normals_fps = torch.gather(normals, 1, out_indices.unsqueeze(-1).expand(-1, -1, normals.shape[-1]))
            obs_vectors_fps[...,:3,0] = normals_fps
        obs_points_fps = {'centers': out_pcd, 'vectors': obs_vectors_fps}
        out_pcd_features = torch.gather(pcd_features, 1, out_indices.unsqueeze(-1).expand(-1, -1, pcd_features.shape[-1]))
        return obs_points_fps, out_pcd_features

    def encode_obs(self, obs):
        obs_points, obs_features, instr_features = super().encode_obs(obs)
        obs_points_fps, pcd_features_fps = self.compute_fps(obs['pcd'], obs_features, obs.get('normals', None))
        return obs_points, obs_features, obs_points_fps, pcd_features_fps, instr_features

class FeaturePCDEncoder(ModuleAttrMixin):
    
    def __init__(self,
                 backbone="clip",
                 feature_res="res3",
                 embedding_dim=60):
        super().__init__()

        assert feature_res in ["res1", "res2", "res3"], "Feature resolution must be either res1, res2 or res3"
        
        # 3D relative positional embeddings
        # Frozen backbone
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False


        self.feature_res = feature_res
        if self.feature_res == "res1":
            self.out_dim = 64
        elif self.feature_res == "res2":
            self.out_dim = 256
        elif self.feature_res == "res3":
            self.out_dim = 512

        self.to_out = nn.Linear(self.out_dim, embedding_dim)
        
    def forward(self, rgb, pcd, normals=None):
        """
        Compute visual features

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats: (B, ncam, F, H, W)
            - pcd: (B, ncam * H * W, 3), resampled point cloud
        """            
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)
        rgb_features = rgb_features[self.feature_res]

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        if normals is not None:
            normals = einops.rearrange(normals, "bt ncam c h w -> (bt ncam) c h w")

        # Interpolate xy-depth to get the locations for this level
        feat_h, feat_w = rgb_features.shape[-2:]
        pcd = F.interpolate(
            pcd,
            (feat_h, feat_w),
            mode='bilinear'
        )

        if normals is not None:
            normals = F.interpolate(
                normals,
                (feat_h, feat_w),
                mode='bilinear'
            )

        # Merge different cameras for clouds, separate for rgb features
        pcd = einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        rgb_features = einops.rearrange(
            rgb_features,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        if normals is not None:
            normals = einops.rearrange(
                normals,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )

        # Project to embedding dim
        rgb_features = self.to_out(rgb_features)

        return rgb_features, pcd, normals
    
class SE3IPAGraspPointCloudEncoder(ModuleAttrMixin):
    def __init__(self, dim_features=128, gripper_depth=3, nheads=4, n_steps_inf=50, nhist=3, num_vis_ins_attn_layers=2, use_adaln=False, gripper_history_as_points=False):
        super(SE3IPAGraspPointCloudEncoder, self).__init__()

        ## Learnable observation features (Data in Acronym is purely geometrical, no semantics involved)
        self.gripper_features = nn.Parameter(torch.randn(nhist,dim_features))

        ## Learnable action features
        self.action_features = nn.Parameter(torch.randn(1, dim_features))

        self._gripper_history_as_points = gripper_history_as_points

        ## Gripper History Encoder ##
        self.gripper_decoder = IPATransformer(d_model=dim_features,
                                                nhead=nheads,
                                                num_layers=gripper_depth,
                                                use_adaln=use_adaln)

        ## Instruction Encoder ##
        self.instruction_encoder = nn.Linear(512, dim_features)

        # Attention from vision to language
        self.vl_attention = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=dim_features, n_heads=nheads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )

        # Attention from action to language
        self.al_attention = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=dim_features, n_heads=nheads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )

        ## Time Encoder ##
        self.time_encoder = nn.Sequential(
            PositionalEncoding(n_positions=dim_features, max_len=n_steps_inf),
               nn.Linear(dim_features * 2, 4 * dim_features),
                nn.LayerNorm(4 * dim_features),
                nn.GELU(),
                nn.Linear(4 * dim_features, dim_features)
            )

        self.obs_merger = nn.Sequential(
                nn.Linear(2 * dim_features, dim_features),
                nn.LayerNorm(dim_features),
                nn.GELU()
            )

        self.act_merger = nn.Sequential(
                nn.Linear(2 * dim_features, dim_features),
                nn.LayerNorm(dim_features),
                nn.GELU()
            )
        
    def forward(self, x):
        obs_points, obs_features = self.encode_obs(x['obs'])
        act_points, act_features = self.encode_act(x['act'])
        time_emb = self.encode_time(x['time'])
        obs_f, act_f = self.combine_time(obs_features, act_features, time_emb)
        return obs_points, obs_f, act_points, act_f

    def encode_time(self, time):
        time_emb = self.time_encoder(time)
        return time_emb[:,None,:]
    
    def encode_instruction(self, instruction):
        return self.instruction_encoder(instruction)

    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention(
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats

    def action_language_attention(self, feats, instr_feats):
        feats, _ = self.al_attention(
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats

    def encode_gripper(self, gripper, obs_features, obs_points):
        gripper_features = self.gripper_features[None,...].repeat(gripper.size(0), 1, 1)
        if self._gripper_history_as_points:
            vectors = torch.zeros((3,3))[None,None,:,:].repeat(gripper.shape[0], gripper.shape[1], 1, 1).to(gripper.device)
        else:
            vectors = gripper[:,:,:3,:3]
        geom_args = {
                    'query': 
                        {'centers': gripper[:,:,:3,-1], 'vectors': vectors},
                    'key': obs_points
                     }
        gripper_features = self.gripper_decoder(tgt=gripper_features, memory=obs_features, geometric_args=geom_args)
        return gripper_features

    def encode_obs(self, obs):
        pcd = obs['pcd']
        pcd_features = obs['pcd_features']
        current_gripper = obs['current_gripper']
        instruction = obs.get('instruction', None)
        normals = obs.get('normals', None)

        batch = pcd.shape[0]
        device = pcd.device

        # encode pcd
        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
        if normals is not None:
            vectors[...,:3,0] = normals
        obs_points = {'centers': pcd, 'vectors': vectors}

        obs_features = pcd_features
        
        # add gripper features
        gripper_features = self.encode_gripper(current_gripper, obs_features, obs_points)        
        obs_features = torch.cat((obs_features, gripper_features), dim=1)
        obs_points['vectors'] = torch.cat((obs_points["vectors"], current_gripper[:,:,:3,:3]), dim=1)
        obs_points['centers'] = torch.cat((obs_points["centers"], current_gripper[:,:,:3,-1]), dim=1)

        if instruction is not None:
            instr_features = self.encode_instruction(instruction)
            obs_features = self.vision_language_attention(obs_features, instr_features)
        else:
            instr_features = None

        return obs_points, obs_features, instr_features

    def encode_act(self, act):
        act_points = {'centers': act[..., :3, -1], 'vectors': act[..., :3, :3]}
        act_features = self.action_features[None,...].repeat(act.shape[0], 1, 1)

        return act_points, act_features

    def combine_time(self, obs_f, act_f, time_emb):
        obs_time_f = self.obs_combine_time(obs_f, time_emb)
        act_time_f = self.act_combine_time(act_f, time_emb)

        return obs_time_f, act_time_f

    def obs_combine_time(self, obs_f, time_emb):
        obs_time_f = torch.cat((obs_f, time_emb.repeat(1, obs_f.shape[1],1)), dim=-1)
        return self.obs_merger(obs_time_f)

    def act_combine_time(self, act_f, time_emb):
        act_time_f = torch.cat((act_f, time_emb.repeat(1, act_f.shape[1],1)), dim=-1)
        return self.act_merger(act_time_f)


class SE3IPAGraspFPSEncoder(SE3IPAGraspPointCloudEncoder):
    def __init__(self, dim_features=128, gripper_depth=3, nheads=4, n_steps_inf=50, fps_subsampling_factor=5, nhist=3, num_vis_ins_attn_layers=2, use_adaln=False, gripper_history_as_points=False):
        super(SE3IPAGraspFPSEncoder, self).__init__(dim_features, gripper_depth, nheads, n_steps_inf, nhist, num_vis_ins_attn_layers, use_adaln, gripper_history_as_points)
        self.fps_subsampling_factor = fps_subsampling_factor
        output_dim = dim_features
        self.linear = nn.Linear(dim_features, output_dim)

    def compute_fps(self, pcd, pcd_features, normals=None):
        n_points_out = pcd_features.shape[1] // self.fps_subsampling_factor
        ## Get n points via FPS ##
        out_pcd, out_indices = fps(pcd, K=n_points_out)
        obs_vectors_fps = torch.zeros((3,3))[None,None,:,:].repeat(out_pcd.shape[0], out_pcd.shape[1], 1, 1).to(pcd.device)
        if normals is not None:
            normals_fps = torch.gather(normals, 1, out_indices.unsqueeze(-1).expand(-1, -1, normals.shape[-1]))
            obs_vectors_fps[...,:3,0] = normals_fps
        obs_points_fps = {'centers': out_pcd, 'vectors': obs_vectors_fps}
        out_pcd_features = torch.gather(pcd_features, 1, out_indices.unsqueeze(-1).expand(-1, -1, pcd_features.shape[-1]))
        return obs_points_fps, out_pcd_features

    def encode_obs(self, obs):
        pcd = obs['pcd']
        pcd_features = obs['pcd_features']
        current_gripper = obs['current_gripper']
        normals = obs.get('normals', None)
        instruction = obs.get('instruction', None)

        batch = pcd.shape[0]
        device = pcd.device

        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
        if normals is not None:
            vectors[...,:3,0] = normals
        obs_points = {'centers': pcd, 'vectors': vectors}
        obs_features = pcd_features

        # pcd - instruction attention
        if instruction is not None:
            instr_features = self.encode_instruction(instruction)
            obs_features = self.vision_language_attention(obs_features, instr_features)
        else:
            instr_features = None

        # add gripper features
        gripper_features = self.encode_gripper(current_gripper, obs_features, obs_points)        
        obs_features = torch.cat((obs_features, gripper_features), dim=1)
        obs_points['vectors'] = torch.cat((obs_points["vectors"], current_gripper[:,:,:3,:3]), dim=1)
        obs_points['centers'] = torch.cat((obs_points["centers"], current_gripper[:,:,:3,-1]), dim=1)

        obs_points_fps, pcd_features_fps = self.compute_fps(pcd, pcd_features, normals)

        return obs_points, obs_features, obs_points_fps, pcd_features_fps, instr_features
