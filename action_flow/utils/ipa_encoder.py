import torch
import torch.nn as nn
from geo3dattn.model.common.module_attr_mixin import ModuleAttrMixin
from action_flow.utils.ipa import InvariantPointTransformer as IPA
from action_flow.utils.ipa import InvariantPointAttention 
from geo3dattn.encoder.common.position_encoder import PositionalEncoding
from diffuser_actor.utils.layers import ParallelAttention


class SE3IPAGraspPointCloudEncoder(ModuleAttrMixin):
    def __init__(self, dim_features=128, gripper_depth=3, nheads=4, n_steps_inf=50, nhist=3, num_vis_ins_attn_layers=2):
        super(SE3IPAGraspPointCloudEncoder, self).__init__()

        ## Learnable observation features (Data in Acronym is purely geometrical, no semantics involved)
        self.gripper_features = nn.Parameter(torch.randn(nhist,dim_features))

        ## Learnable action features
        self.action_features = nn.Parameter(torch.randn(1, dim_features))

        dim_head = dim_features // nheads

        ## Gripper History Encoder ##
        self.gripper_decoder = IPA(dim=dim_features, 
                                   depth=gripper_depth, 
                                   heads=nheads,
                                   dim_head=dim_head,
                                   kv_dim=dim_features,
                                   attention_module=InvariantPointAttention,
                                   use_adaln=True)

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
