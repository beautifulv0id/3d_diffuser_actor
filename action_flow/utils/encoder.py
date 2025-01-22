import torch
import torch.nn as nn
from geo3dattn.model.common.module_attr_mixin import ModuleAttrMixin
from geo3dattn.encoder.superpoint_encoder.super_point_encoder import SuperPointEncoder
from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformerEncoder, URSATransformer
from geo3dattn.encoder.common.position_encoder import PositionalEncoding
from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.resnet import load_resnet50, load_resnet18
from diffuser_actor.utils.clip import load_clip
import einops

from torch.nn import functional as F


class SE3GraspPointCloudSuperEncoder(ModuleAttrMixin):
    def __init__(self, dim_features=128, depth=3, nheads=4, n_steps_inf=50, subsampling_factor=5, nhist=3, dim_pcd_features=64, num_vis_ins_attn_layers=2):
        super(SE3GraspPointCloudSuperEncoder, self).__init__()

        ## Learnable observation features (Data in Acronym is purely geometrical, no semantics involved)
        self.gripper_features = nn.Parameter(torch.randn(nhist,dim_features))

        ## Learnable action features
        self.action_features = nn.Parameter(torch.randn(1, dim_features))

        ## Pointcloud Encoder ##
        self.obs_encoder = SuperPointEncoder(input_dim=dim_pcd_features, output_dim=dim_features, subsampling_factor=subsampling_factor,
                                             nheads=nheads, num_layers=depth)
        
        ## Gripper History Encoder ##
        self.gripper_decoder = URSATransformer(dim_features, nhead=nheads, num_layers=depth)

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
        geom_args = {
                    'query': 
                        {'centers': gripper[:,:,:3,-1], 'vectors': gripper[:,:,:3,:3]},
                    'key': obs_points
                     }
        gripper_features = self.gripper_decoder(tgt=gripper_features, memory=obs_features, geometric_args=geom_args)
        return gripper_features

    def encode_obs(self, obs):
        pcd = obs['pcd']
        pcd_features = obs['pcd_features']
        current_gripper = obs['current_gripper']
        instruction = obs.get('instruction', None)

        batch = pcd.shape[0]
        device = pcd.device

        # encode pcd
        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
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
            # instr_features = self.encode_instruction(instruction)
            instr_features = instruction
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
    
class SE3GraspFPSEncoder(SE3GraspPointCloudSuperEncoder):
    def __init__(self, dim_features=128, depth=3, nheads=4, n_steps_inf=50, n_points_out=100, nhist=3, dim_pcd_features=64):
        super(SE3GraspFPSEncoder, self).__init__(dim_features, depth, nheads, n_steps_inf, n_points_out, nhist, dim_pcd_features)
        input_dim = dim_pcd_features
        output_dim = dim_features
        self.linear = nn.Linear(input_dim, output_dim)

    def encode_obs(self, obs):
        obs_pcd_x, obs_pcd_f, inst_f = super().encode_obs(obs)
        pcd, obs_f = obs['pcd'], obs['pcd_features']
        batch = pcd.shape[0]
        device = pcd.device
        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
        obs_x = {'centers': pcd, 'vectors': vectors}
        obs_f = self.linear(obs_f)
        return obs_x, obs_f, obs_pcd_x, obs_pcd_f, inst_f


class FeaturePCDEncoder(ModuleAttrMixin):
    
    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 feature_res="res2"):
        super().__init__()

        assert feature_res in ["res1", "res2", "res3"]
        assert image_size in [(128, 128), (256, 256)]

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

        if image_size == (128, 128):
            self.feature_res = "res2"
            self.out_dim = 256
        elif image_size == (256, 256):
            self.feature_res = "res3"
            self.out_dim = 512
        self.obs_features = nn.Parameter(torch.randn(1, self.out_dim))
        
    def forward(self, rgb, pcd):
        if rgb is not None:
            return self.get_feature_pcd(rgb, pcd)
        else:
            return self.get_lowdim_feature_pcd(pcd)
    
    def get_lowdim_feature_pcd(self, pcd):
        batch, npts = pcd.shape[0:2]
        feats = self.obs_features.unsqueeze(0).expand(batch, npts, -1)
        return feats, pcd


    def get_feature_pcd(self, rgb, pcd):
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

        # Interpolate xy-depth to get the locations for this level
        feat_h, feat_w = rgb_features.shape[-2:]
        pcd = F.interpolate(
            pcd,
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

        return rgb_features, pcd
    
if __name__=='__main__':
    # Example usage
    batch = 30
    n_tokens = 3000
    emb_dim = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = {
        'obs': {
            'pcd': torch.randn((batch, n_tokens, 3)).to(device),
            'pcd_features': torch.randn((batch, n_tokens, emb_dim)).to(device),
            'current_gripper': torch.randn((batch, 1, 4, 4)).to(device)
        },
        'act': torch.randn((batch, 1, 4, 4)).to(device),
        'time': torch.randn((batch)).to(device)
    }

    encoder = SE3GraspPointCloudSuperEncoder(dim_features=emb_dim, depth=3, nheads=4, n_steps_inf=50, n_points_out=20, nhist=3).to(device)
    obs_points, obs_f, act_points, act_f = encoder(x)
    print("Obs Points:", obs_points['centers'].shape, obs_points['vectors'].shape)
    print("Obs Features:", obs_f.shape)
    print("Act Points:", act_points['centers'].shape, act_points['vectors'].shape)
    print("Act Features:", act_f.shape)