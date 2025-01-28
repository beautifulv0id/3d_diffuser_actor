import torch
import torch.nn as nn
from diffuser_actor.utils.layers import ParallelAttention
from action_flow.ipa.transformer import InvariantPointTransformer as IPA
from action_flow.ipa.transformer import InvariantPointAttention

class LangEnhancedIPADecoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, use_adaln=False):
        super().__init__()   
        self.ipa_layer = nn.ModuleList() 
        self.lang_layer = nn.ModuleList()
        dim_head = d_model // nhead

        for _ in range(num_layers):
            self.ipa_layer.append(IPA(dim=d_model, 
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
        for ipa_layer, lang_layer in zip(self.ipa_layer, self.lang_layer):
            tgt = ipa_layer(tgt, memory, geometric_args=geometric_args, diff_ts=diff_ts)
            feats = torch.cat([tgt, memory], dim=1)
            feats, _ = lang_layer(
                seq1=feats, seq1_key_padding_mask=None,
                seq2=lang_memory, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
            )
            tgt, memory = feats[:,:tgt_len], feats[:,tgt_len:]
        return tgt


class LangEnhancedIPASADecoder(nn.Module):
    def __init__(self, embedding_dim, x1_depth=2, s_depth=2, x2_depth=2, nhead=8, dropout=0.2, use_adaln=False):
        super().__init__()
        dim_head = embedding_dim // nhead
        self.cross_attn1 = IPA(
            dim=embedding_dim,
            depth=x1_depth,
            heads=nhead,
            dim_head=dim_head,
            kv_dim=embedding_dim,
            attention_module=InvariantPointAttention,
            dropout=dropout,
            use_adaln=use_adaln
        )
        self.self_attn = IPA(
            dim=embedding_dim,
            depth=x1_depth,
            heads=nhead,
            dim_head=dim_head,
            kv_dim=None,
            attention_module=InvariantPointAttention,
            dropout=dropout,
            use_adaln=use_adaln
        )
        self.cross_attn2 = IPA(
            dim=embedding_dim,
            depth=x1_depth,
            heads=nhead,
            dim_head=dim_head,
            kv_dim=embedding_dim,
            attention_module=InvariantPointAttention,
            dropout=dropout,
            use_adaln=use_adaln
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
    
