import torch
import torch.nn as nn
from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformerEncoder, URSATransformer
from geo3dattn.model.nursa_transformer.ursa_transformer import NURSATransformer, NURSATransformerEncoder

from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.utils import (
    measure_memory
)

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
                add_center=True):
        super().__init__()

        self.cross_attn1 = URSATransformer(d_model=embedding_dim, nhead=nhead, num_layers=x1_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center, distance_scale=distance_scale, use_adaln=use_adaln)
        self.self_attn = NURSATransformerEncoder(d_model=embedding_dim, nhead=nhead, num_layers=s_depth, dropout=dropout, feature_type=feature_type, use_center_distance=use_center_distance, use_center_projection=use_center_projection, use_vector_projection=use_vector_projection, add_center=add_center)
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


class LangEnhancedURSADecoder(nn.Module):

    def __init__(self, 
                        d_model, 
                        nhead, 
                        num_layers, 
                        dropout=0.0, 
                        distance_scale=1.0, 
                        use_adaln=False, 
                        feature_type='sinusoid', 
                        use_center_distance=True,
                        use_center_projection=True,
                        use_vector_projection=True,
                        add_center=True):
        super().__init__()   
        self.ursa_layer = nn.ModuleList() 
        self.lang_layer = nn.ModuleList()
        for _ in range(num_layers):
            self.ursa_layer.append(URSATransformer(d_model=d_model, 
                                                    nhead=nhead, 
                                                    num_layers=1, 
                                                    dropout=dropout, 
                                                    distance_scale=distance_scale, 
                                                    use_adaln=use_adaln, 
                                                    feature_type=feature_type,
                                                    use_center_distance=use_center_distance,
                                                    use_center_projection=use_center_projection,
                                                    use_vector_projection=use_vector_projection,
                                                    add_center=add_center))
            self.lang_layer.append(ParallelAttention(
            num_layers=1,
            d_model=d_model, n_heads=nhead,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        ))
            
    def forward(self, tgt, memory, lang_memory, geometric_args, diff_ts=None):
        tgt_len = tgt.size(1)
        for ursa_layer, lang_layer in zip(self.ursa_layer, self.lang_layer):
            tgt = ursa_layer(tgt, memory, geometric_args=geometric_args, diff_ts=diff_ts)
            feats = torch.cat([tgt, memory], dim=1)
            feats, _ = lang_layer(
                seq1=feats, seq1_key_padding_mask=None,
                seq2=lang_memory, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
            )
            tgt, memory = feats[:,:tgt_len], feats[:,tgt_len:]
        return tgt
