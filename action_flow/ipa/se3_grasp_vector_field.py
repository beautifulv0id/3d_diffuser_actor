import torch.nn as nn
from action_flow.utils.se3_grasp_vector_field import SE3GraspVectorField

class SE3GraspVectorFieldLangEnhancedIPASelfAttn(SE3GraspVectorField):
    def __init__(self,
                 encoder,
                 decoder,
                 latent_dim=64,
                 output_dim=6
                 ):
        super(SE3GraspVectorFieldLangEnhancedIPASelfAttn, self).__init__(encoder, decoder, latent_dim, output_dim)
    def encode_obs(self, x):
        obs_x, obs_f, obs_pcd_x, obs_pcd_f, inst_f = self.encoder.encode_obs(x)
        return obs_x, obs_f, obs_pcd_x, obs_pcd_f, inst_f

    def set_context(self, obs_x, obs_f, obs_fps_x, obs_fps_f, inst_f=None):
        self.obs_x = obs_x
        self.obs_f = obs_f
        self.obs_fps_x = obs_fps_x
        self.obs_fps_f = obs_fps_f
        self.inst_f = inst_f

    def forward_act(self, x):
        act_x, act_f = self.encoder.encode_act(x['act'])
        time_emb = self.encoder.encode_time(x['time'])
        act_x['time'] = time_emb

        obs_f = self.encoder.obs_combine_time(self.obs_f, time_emb)
        obs_fps_f = self.encoder.obs_combine_time(self.obs_fps_f, time_emb)
        act_f = self.encoder.act_combine_time(act_f, time_emb)
        if self.inst_f is not None:
            act_f = self.encoder.action_language_attention(act_f, self.inst_f)

        out = self.decoder(tgt = act_f, cross_memory = obs_f, self_memory = obs_fps_f, 
                           query_geometric_args = act_x, cross_geometric_args = self.obs_x, 
                           self_geometric_args = self.obs_fps_x, lang_memory = self.inst_f, diff_ts=time_emb)
        return self.out_fn(out), self.grasp_out_fn(out).sigmoid()
