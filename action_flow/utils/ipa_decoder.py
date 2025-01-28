import torch
import torch.nn as nn
from diffuser_actor.utils.layers import ParallelAttention
from action_flow.utils.ipa import InvariantPointTransformer as IPA
from action_flow.utils.ipa import InvariantPointAttention

class LangEnhancedURSADecoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, use_adaln=False):
        super().__init__()   
        self.ursa_layer = nn.ModuleList() 
        self.lang_layer = nn.ModuleList()
        dim_head = d_model // nhead

        for _ in range(num_layers):
            self.ursa_layer.append(IPA(dim=d_model, 
                                   depth=1, 
                                   heads=dim_head,
                                   dim_head=dim_head,
                                   kv_dim=d_model,
                                   attention_module=InvariantPointAttention,
                                   use_adaln=use_adaln))
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

