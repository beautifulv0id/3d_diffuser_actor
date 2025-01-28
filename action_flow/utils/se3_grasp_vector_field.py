import torch.nn as nn
from geo3dattn.model.common.module_attr_mixin import ModuleAttrMixin


class SE3GraspVectorField(ModuleAttrMixin):
    def __init__(self,
                 encoder,
                 decoder,
                 latent_dim=64,
                 output_dim=6
                 ):
        super(SE3GraspVectorField, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.out_fn = nn.Linear(latent_dim, output_dim)
        self.grasp_out_fn = nn.Linear(latent_dim, 1)

    def forward(self, x):
        obs_x, obs_f, act_x, act_f = self.encoder(x)
        geo = {'query':act_x, 'key':obs_x}
        out = self.decoder(tgt = act_f, memory = obs_f, geometric_args = geo)
        return self.out_fn(out)

    def encode_obs(self, x):
        obs_x, obs_f, inst_f = self.encoder.encode_obs(x)
        return obs_x, obs_f, inst_f

    def encode_act(self, x):
        act_x, act_f = self.encoder.encode_act(x)
        return act_x, act_f

    def set_context(self, obs_x, obs_f, inst_f=None):
        self.obs_x = obs_x
        self.obs_f = obs_f
        self.inst_f = inst_f

    def forward_act(self, x):
        act_x, act_f = self.encoder.encode_act(x['act'])
        time_emb = self.encoder.encode_time(x['time'])
        act_x['time'] = time_emb

        obs_f, act_f = self.encoder.combine_time(self.obs_f, act_f, time_emb)
        if self.inst_f is not None:
            act_f = self.encoder.action_language_attention(act_f, self.inst_f)

        geo = {'query':act_x, 'key':self.obs_x}
        out = self.decoder(tgt = act_f, memory = obs_f, geometric_args = geo)
        return self.out_fn(out), self.grasp_out_fn(out).sigmoid()
    

class SE3GraspVectorFieldLangEnhanced(SE3GraspVectorField):
    def __init__(self,
                 encoder,
                 decoder,
                 latent_dim=64,
                 output_dim=6
                 ):
        super(SE3GraspVectorFieldLangEnhanced, self).__init__(encoder, decoder, latent_dim, output_dim)

    def forward_act(self, x):
        act_x, act_f = self.encoder.encode_act(x['act'])
        time_emb = self.encoder.encode_time(x['time'])
        act_x['time'] = time_emb

        obs_f, act_f = self.encoder.combine_time(self.obs_f, act_f, time_emb)
        if self.inst_f is not None:
            act_f = self.encoder.action_language_attention(act_f, self.inst_f)

        geo = {'query':act_x, 'key':self.obs_x}
        out = self.decoder(tgt = act_f, memory = obs_f, lang_memory = self.inst_f, geometric_args = geo, diff_ts = time_emb)
        return self.out_fn(out), self.grasp_out_fn(out).sigmoid()

class SE3GraspVectorFieldSelfAttn(SE3GraspVectorField):
    def __init__(self,
                 encoder,
                 decoder,
                 latent_dim=64,
                 output_dim=6
                 ):
        super(SE3GraspVectorFieldSelfAttn, self).__init__(encoder, decoder, latent_dim, output_dim)

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
                           self_geometric_args = self.obs_fps_x)
        return self.out_fn(out), self.grasp_out_fn(out).sigmoid()