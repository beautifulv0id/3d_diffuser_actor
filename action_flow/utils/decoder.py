import torch
import torch.nn as nn
from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformerEncoder, URSATransformer
from geo3dattn.model.nursa_transformer.ursa_transformer import NURSATransformer, NURSATransformerEncoder

from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.utils import (
    measure_memory, merge_geometric_args
)
from geo3dattn.model.ipa_transformer.ipa_transformer import InvariantPointTransformer
class LangEnhancedIPADecoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, use_adaln=False, dropout=0.0, attention_module=InvariantPointTransformer, point_dim=4):
        super().__init__()   
        self.ipa_layer = nn.ModuleList() 
        self.lang_layer = nn.ModuleList()
        dim_head = d_model // nhead

        for _ in range(num_layers):
            
            self.ipa_layer.append(InvariantPointTransformer(dim=d_model,
                                                            depth=1,
                                                            heads=nhead,
                                                            kv_dim=d_model,
                                                            use_adaln=use_adaln, 
                                                            dropout=dropout,
                                                            attention_module=attention_module,
                                                            point_dim=point_dim
                                                            ))
             
            self.lang_layer.append(ParallelAttention(
            num_layers=1,
            d_model=d_model, n_heads=nhead,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        ))
            
    def forward(self, tgt, cross_memory, lang_memory, geometric_args, diff_ts=None):
        tgt_len = tgt.size(1)
        for ipa_layer, lang_layer in zip(self.ipa_layer, self.lang_layer):
            tgt = ipa_layer(tgt, cross_memory, geometric_args=geometric_args, diff_ts=diff_ts)
            feats = torch.cat([tgt, cross_memory], dim=1)
            feats, _ = lang_layer(
                seq1=feats, seq1_key_padding_mask=None,
                seq2=lang_memory, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
            )
            tgt, cross_memory = feats[:,:tgt_len], feats[:,tgt_len:]
        return tgt


class LangEnhancedIPASADecoder(nn.Module):
    def __init__(self, embedding_dim, x1_depth=2, s_depth=2, x2_depth=2, nhead=8, dropout=0.2, use_adaln=False, attention_module=InvariantPointTransformer, point_dim=4):
        super().__init__()
        dim_head = embedding_dim // nhead
        self.cross_attn1 = InvariantPointTransformer(
            dim=embedding_dim, depth=x1_depth, heads=nhead, dim_head=dim_head, kv_dim=embedding_dim, use_adaln=use_adaln, dropout=dropout, attention_module=attention_module, point_dim=point_dim
        )
        
        self.self_attn = InvariantPointTransformer(
            dim=embedding_dim, depth=s_depth, heads=nhead, dim_head=dim_head, kv_dim=None, use_adaln=use_adaln, dropout=dropout, attention_module=attention_module, point_dim=point_dim
        )
        
        self.cross_attn2 = InvariantPointTransformer(
            dim=embedding_dim, depth=x2_depth, heads=nhead, dim_head=dim_head, kv_dim=embedding_dim, use_adaln=use_adaln, dropout=dropout, attention_module=attention_module, point_dim=point_dim
        )

        self.lang1 = ParallelAttention(
            num_layers=1,
            d_model=embedding_dim, n_heads=nhead,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.lang2 = ParallelAttention(
            num_layers=1,
            d_model=embedding_dim, n_heads=nhead,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )

    def forward(self, tgt, cross_memory, self_memory, query_geometric_args, cross_geometric_args, self_geometric_args, lang_memory, diff_ts=None):
        nact = tgt.size(1)
        geometric_args = {'query': query_geometric_args, 'key': cross_geometric_args}
        out = self.cross_attn1(tgt, cross_memory, geometric_args=geometric_args, diff_ts=diff_ts)

        # Language attention
        out = torch.cat([out, self_memory], dim=1)
        out, _ = self.lang1(
            seq1=out, seq1_key_padding_mask=None,
            seq2=lang_memory, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )

        centers = torch.cat([query_geometric_args['centers'], self_geometric_args['centers']], dim=1)
        vectors = torch.cat([query_geometric_args['vectors'], self_geometric_args['vectors']], dim=1)
        geometric_args = {'query': {'centers': centers, 'vectors': vectors}}
        
        self_out = self.self_attn(tgt=out, geometric_args=geometric_args, diff_ts=diff_ts)

        self_out, _ = self.lang2(
            seq1=self_out, seq1_key_padding_mask=None,
            seq2=lang_memory, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )

        tgt = self_out[:, :nact]
        cross_memory = self_out[:, nact:]
        geometric_args = {'query': {'centers': query_geometric_args['centers'][:, :nact], 'vectors': query_geometric_args['vectors'][:, :nact]},
                          'key': {'centers': geometric_args['query']['centers'][:, nact:], 'vectors': geometric_args['query']['vectors'][:, nact:]}}
        out = self.cross_attn2(tgt, cross_memory, geometric_args=geometric_args, diff_ts=diff_ts)

        return out
    

class DiffuserActorIPADecoder(nn.Module):
    def __init__(self, embedding_dim, nhead=8, dropout=0.2, use_adaln=False, attention_module=InvariantPointTransformer, point_dim=4):
        super().__init__()
        dim_head = embedding_dim // nhead
        self.cross_attn1 = InvariantPointTransformer(
            dim=embedding_dim, depth=2, heads=nhead, dim_head=dim_head, kv_dim=embedding_dim, use_adaln=use_adaln, dropout=dropout, attention_module=attention_module, point_dim=point_dim
        )
        
        self.self_attn = InvariantPointTransformer(
            dim=embedding_dim, depth=4, heads=nhead, dim_head=dim_head, kv_dim=None, use_adaln=use_adaln, dropout=dropout, attention_module=attention_module, point_dim=point_dim
        )
        
        self.rotation_self_attn = InvariantPointTransformer(
            dim=embedding_dim, depth=4, heads=nhead, dim_head=dim_head, kv_dim=None, use_adaln=use_adaln, dropout=dropout, attention_module=attention_module, point_dim=point_dim
        )
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_predictor = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 3)
        )

        self.position_self_attn = InvariantPointTransformer(
            dim=embedding_dim, depth=4, heads=nhead, dim_head=dim_head, kv_dim=None, use_adaln=use_adaln, dropout=dropout, attention_module=attention_module, point_dim=point_dim
        )
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_predictor = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 3)
            )

        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )


        self.lang1 = ParallelAttention(
            num_layers=1,
            d_model=embedding_dim, n_heads=nhead,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.lang2 = ParallelAttention(
            num_layers=1,
            d_model=embedding_dim, n_heads=nhead,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )

    def forward(self, tgt, cross_memory, self_memory, query_geometric_args, cross_geometric_args, self_geometric_args, lang_memory, diff_ts=None):
        nact = tgt.size(1)
        geometric_args = {'query': query_geometric_args, 'key': cross_geometric_args}
        out = self.cross_attn1(tgt, cross_memory, geometric_args=geometric_args, diff_ts=diff_ts)

        # Language attention
        out = torch.cat([out, self_memory], dim=1)
        out, _ = self.lang1(
            seq1=out, seq1_key_padding_mask=None,
            seq2=lang_memory, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )

        # Self attention
        geometric_args = {'query': merge_geometric_args(query_geometric_args, self_geometric_args)}
        self_out = self.self_attn(tgt=out, geometric_args=geometric_args, diff_ts=diff_ts)

        # Language attention
        self_out, _ = self.lang2(
            seq1=self_out, seq1_key_padding_mask=None,
            seq2=lang_memory, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )

        # Predict rotation and position
        rotation = self.predict_rot(self_out, geometric_args, diff_ts, nact)
        position, features = self.predict_pos(self_out, geometric_args, diff_ts, nact)

        openness = self.openess_predictor(features).sigmoid()
        out = torch.cat([position, rotation], dim=-1)
        return out, openness

    def predict_pos(self, tgt, geometric_args, diff_ts, n_act):
        tgt = self.position_self_attn(tgt=tgt, geometric_args=geometric_args, diff_ts=diff_ts)
        tgt = tgt[:, :n_act]
        pos = self.position_proj(tgt)
        pos = self.position_predictor(pos)
        return pos, tgt
    
    def predict_rot(self, tgt, geometric_args, diff_ts, n_act):
        tgt = self.rotation_self_attn(tgt=tgt, geometric_args=geometric_args, diff_ts=diff_ts)
        tgt = tgt[:, :n_act]
        rot = self.rotation_proj(tgt)
        rot = self.rotation_predictor(rot)
        return rot

    

class SE3PCDSelfAttnDecoder(nn.Module):
    def __init__(self, 
                embedding_dim, 
                x1_depth=2, 
                s_depth=2, 
                x2_depth=2, 
                nhead=8, 
                dropout=0.2, 
                feature_type='sinusoid',
                distance_scale=1.0, 
                use_adaln=False, 
                use_center_distance=True,
                use_center_projection=True,
                use_vector_projection=True,
                add_center=True):
        super().__init__()

        self.cross_attn1 = URSATransformer(d_model=embedding_dim, nhead=nhead, num_layers=x1_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center, distance_scale=distance_scale, use_adaln=use_adaln)
        self.self_attn = URSATransformerEncoder(d_model=embedding_dim, nhead=nhead, num_layers=s_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center)
        self.cross_attn2 = URSATransformer(d_model=embedding_dim, nhead=nhead, num_layers=x2_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center, distance_scale=distance_scale, use_adaln=use_adaln)

    def forward(self, tgt, cross_memory, self_memory, query_geometric_args, cross_geometric_args, self_geometric_args):
        nact = tgt.size(1)
        geometric_args = {'query': query_geometric_args, 'key': cross_geometric_args}
        out = measure_memory(self.cross_attn1.forward, tgt, cross_memory, geometric_args=geometric_args)

        out = torch.cat([out, self_memory], dim=1)
        centers = torch.cat([query_geometric_args['centers'], self_geometric_args['centers']], dim=1)
        vectors = torch.cat([query_geometric_args['vectors'], self_geometric_args['vectors']], dim=1)
        geometric_args = {'query': {'centers': centers, 'vectors': vectors}}
        
        self_out = measure_memory(self.self_attn.forward, tgt=out, geometric_args=geometric_args)

        tgt = self_out[:, :nact]
        cross_memory = self_out[:, nact:]
        geometric_args = {'query': {'centers': query_geometric_args['centers'][:, :nact], 'vectors': query_geometric_args['vectors'][:, :nact]},
                          'key': {'centers': geometric_args['query']['centers'][:, nact:], 'vectors': geometric_args['query']['vectors'][:, nact:]}}
        out = measure_memory(self.cross_attn2.forward, tgt, cross_memory, geometric_args=geometric_args)

        return out
    

class SE3PCDEfficientSelfAttnDecoder(nn.Module):
    def __init__(self, 
                embedding_dim, 
                x1_depth=2, 
                s_depth=2, 
                x2_depth=2, 
                nhead=8, 
                dropout=0.2, 
                feature_type='sinusoid',
                distance_scale=1.0, 
                use_adaln=False, 
                use_center_distance=True,
                use_center_projection=True,
                use_vector_projection=True,
                add_center=True,
                point_embedding_dim=64):
        super().__init__()

        self.cross_attn1 = URSATransformer(d_model=embedding_dim, nhead=nhead, num_layers=x1_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center, distance_scale=distance_scale, use_adaln=use_adaln)
        self.self_attn = NURSATransformerEncoder(d_model=embedding_dim, nhead=nhead, num_layers=s_depth, dropout=dropout, point_embedding_dim=point_embedding_dim)
        self.cross_attn2 = URSATransformer(d_model=embedding_dim, nhead=nhead, num_layers=x2_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center, distance_scale=distance_scale, use_adaln=use_adaln)

    def forward(self, tgt, cross_memory, self_memory, query_geometric_args, cross_geometric_args, self_geometric_args):
        nact = tgt.size(1)
        geometric_args = {'query': query_geometric_args, 'key': cross_geometric_args}
        out = measure_memory(self.cross_attn1.forward, tgt, cross_memory, geometric_args=geometric_args)

        out = torch.cat([out, self_memory], dim=1)
        centers = torch.cat([query_geometric_args['centers'], self_geometric_args['centers']], dim=1)
        vectors = torch.cat([query_geometric_args['vectors'], self_geometric_args['vectors']], dim=1)
        geometric_args = {'query': {'centers': centers, 'vectors': vectors}}
        
        self_out = measure_memory(self.self_attn.forward, tgt=out, geometric_args=geometric_args)

        tgt = self_out[:, :nact]
        cross_memory = self_out[:, nact:]
        geometric_args = {'query': {'centers': query_geometric_args['centers'][:, :nact], 'vectors': query_geometric_args['vectors'][:, :nact]},
                          'key': {'centers': geometric_args['query']['centers'][:, nact:], 'vectors': geometric_args['query']['vectors'][:, nact:]}}
        out = measure_memory(self.cross_attn2.forward, tgt, cross_memory, geometric_args=geometric_args)

        return out

class URSATransformerLangEnhanced(nn.Module):
    def __init__(self, num_layers, nhead, d_model, **kwargs):
        super().__init__()
        self.ursa_layer = nn.ModuleList([
            URSATransformer(num_layers=1, nhead=nhead, d_model=d_model, **kwargs) for _ in range(num_layers)
        ])
        self.lang_layer = nn.ModuleList([
            ParallelAttention(
            num_layers=1,
            d_model=d_model, n_heads=nhead,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        ) for _ in range(num_layers)
        ])
        
    def forward(self, tgt, cross_memory, lang_memory, geometric_args, diff_ts=None):
        tgt_len = tgt.size(1)
        for ursa_layer, lang_layer in zip(self.ursa_layer, self.lang_layer):
            tgt = ursa_layer(tgt, cross_memory, geometric_args=geometric_args, diff_ts=diff_ts)
            feats = torch.cat([tgt, cross_memory], dim=1)
            feats, _ = lang_layer(
                seq1=feats, seq1_key_padding_mask=None,
                seq2=lang_memory, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
            )
            tgt, cross_memory = feats[:,:tgt_len], feats[:,tgt_len:]
        return tgt
        
class URSATransformerEncoderLangEnhanced(nn.Module):
    def __init__(self, num_layers, nhead, d_model, **kwargs):
        super().__init__()
        self.ursa_layer = nn.ModuleList([
            URSATransformerEncoder(num_layers=1, nhead=nhead, d_model=d_model, **kwargs) for _ in range(num_layers)
        ])
        self.lang_layer = nn.ModuleList([
            ParallelAttention(
            num_layers=1,
            d_model=d_model, n_heads=nhead,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        ) for _ in range(num_layers)
        ])
        
    def forward(self, tgt, geometric_args, lang_memory, diff_ts=None):
        for ursa_layer, lang_layer in zip(self.ursa_layer, self.lang_layer):
            tgt = ursa_layer(tgt, geometric_args=geometric_args, diff_ts=diff_ts)
            tgt, _ = lang_layer(
                seq1=tgt, seq1_key_padding_mask=None,
                seq2=lang_memory, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
            )
        return tgt

class URSAXSAXLangEnhancedDecoder(nn.Module):
    def __init__(self, 
                d_model, 
                x1_depth=2, 
                s_depth=2, 
                x2_depth=2, 
                nhead=8, 
                dropout=0.2, 
                feature_type='sinusoid',
                distance_scale=1.0, 
                use_adaln=False, 
                use_center_distance=True,
                use_center_projection=True,
                use_vector_projection=True,
                add_center=True):
        super().__init__()

        self.cross_attn1 = URSATransformerLangEnhanced(d_model=d_model, nhead=nhead, num_layers=x1_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center, distance_scale=distance_scale, use_adaln=use_adaln)
        self.self_attn = URSATransformerEncoderLangEnhanced(d_model=d_model, nhead=nhead, num_layers=s_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center)
        self.cross_attn2 = URSATransformerLangEnhanced(d_model=d_model, nhead=nhead, num_layers=x2_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center, distance_scale=distance_scale, use_adaln=use_adaln)

    def forward(self, tgt, cross_memory, self_memory, query_geometric_args, cross_geometric_args, self_geometric_args, lang_memory, diff_ts=None):
        nact = tgt.size(1)
        geometric_args = {'query': query_geometric_args, 'key': cross_geometric_args}
        out = measure_memory(self.cross_attn1.forward, tgt, cross_memory, geometric_args=geometric_args, lang_memory=lang_memory, diff_ts=diff_ts)

        out = torch.cat([out, self_memory], dim=1)
        centers = torch.cat([query_geometric_args['centers'], self_geometric_args['centers']], dim=1)
        vectors = torch.cat([query_geometric_args['vectors'], self_geometric_args['vectors']], dim=1)
        geometric_args = {'query': {'centers': centers, 'vectors': vectors}}
        
        self_out = measure_memory(self.self_attn.forward, tgt=out, geometric_args=geometric_args, lang_memory=lang_memory, diff_ts=diff_ts)

        tgt = self_out[:, :nact]
        cross_memory = self_out[:, nact:]
        geometric_args = {'query': {'centers': query_geometric_args['centers'][:, :nact], 'vectors': query_geometric_args['vectors'][:, :nact]},
                          'key': {'centers': geometric_args['query']['centers'][:, nact:], 'vectors': geometric_args['query']['vectors'][:, nact:]}}
        out = measure_memory(self.cross_attn2.forward, tgt, cross_memory, geometric_args=geometric_args, lang_memory=lang_memory, diff_ts=diff_ts)

        return out
