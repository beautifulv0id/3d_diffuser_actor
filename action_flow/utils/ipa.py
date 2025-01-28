import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.cuda.amp import autocast
from contextlib import contextmanager
from diffuser_actor.utils.layers import AdaLN

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value

def FeedForward(dim, mult = 1., num_layers = 2, act = nn.ReLU):
    layers = []
    dim_hidden = dim * mult

    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden

        layers.append(nn.Linear(dim_in, dim_out))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)

class InvariantPointAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kv_dim=None,
                 point_dim=4, eps=1e-8, use_adaln=False, **kwargs):

        super().__init__()
        if kv_dim is None:
            kv_dim = dim

        self.eps = eps
        self.heads = heads

        # num attention contributions
        num_attn_logits = 2

        # qkv projection for scalar attention (normal)
        self.scalar_attn_logits_scale = (num_attn_logits * dim_head) ** -0.5

        self.to_scalar_q = nn.Linear(dim, dim_head * heads, bias = False)
        self.to_scalar_k = nn.Linear(kv_dim, dim_head * heads, bias = False)
        self.to_scalar_v = nn.Linear(kv_dim, dim_head * heads, bias = False)

        # qkv projection for point attention (coordinate and orientation aware)
        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_dim * heads * 3, bias = False)
        self.to_point_k = nn.Linear(kv_dim, point_dim * heads * 3, bias = False)
        self.to_point_v = nn.Linear(kv_dim, point_dim * heads * 3, bias = False)


        # pairwise representation projection to attention bias
        pairwise_repr_dim = 0

        # combine out - scalar dim + point dim * (3 for coordinates in R3 and then 1 for norm)
        self.to_out = nn.Linear(heads * (dim_head + pairwise_repr_dim + point_dim * (3 + 1)), dim)

        if use_adaln:
            self.adaln = AdaLN(dim)

    def forward(self, tgt, memory=None, geometric_args=None,  mask=None, diff_ts=None):
        if diff_ts is not None:
            tgt = self.adaln(tgt, diff_ts)
        else:
            tgt = tgt
        tgt, b, h, eps = tgt, tgt.shape[0], self.heads, self.eps
        if memory is None:
            memory=tgt
        query_centers = geometric_args['query']['centers']
        query_vectors = geometric_args['query']['vectors']
        key_centers = geometric_args['key']['centers']
        key_vectors = geometric_args['key']['vectors']

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways
        q_scalar, k_scalar, v_scalar = self.to_scalar_q(tgt), self.to_scalar_k(memory), self.to_scalar_v(memory)

        q_point, k_point, v_point = self.to_point_q(tgt), self.to_point_k(memory), self.to_point_v(memory)


        # split out heads
        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h = h, c = 3), (q_point, k_point, v_point))
        
        ## Extract Poses ##
        rotations_q = repeat(query_vectors, 'b n r1 r2 -> (b h) n r1 r2', h = h)
        translations_q = repeat(query_centers, 'b n c -> (b h) n () c', h = h)
        rotations_kv = repeat(key_vectors, 'b n r1 r2 -> (b h) n r1 r2', h=h)
        translations_kv = repeat(key_centers, 'b n c -> (b h) n () c', h=h)

        # rotate qkv points into global frame
        q_point = einsum('b n c r, b n d r -> b n d c', rotations_q, q_point) + translations_q
        k_point = einsum('b n c r, b n d r -> b n d c', rotations_kv, k_point) + translations_kv
        v_point = einsum('b n c r, b n d r -> b n d c', rotations_kv, v_point) + translations_kv

        # derive attn logits for scalar and pairwise
        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale

        # derive attn logits for point attention
        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2).sum(dim = (-1, -2))

        # self.point_qk_diff = point_qk_diff

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () ()', b = b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale)

        # combine attn logits
        attn_logits = attn_logits_scalar + attn_logits_points

        # mask
        if exists(mask):
            #mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask[None,...], 'b i j -> (b h) i j', h = attn_logits.shape[0])
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        # attention
        attn = attn_logits.softmax(dim = - 1)

        with disable_tf32(), autocast(enabled = False):
            # disable TF32 for precision
            # aggregate values
            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)
            # aggregate point values
            results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)
            # rotate aggregated point values back into local frame
            results_points = einsum('b n c r, b n d r -> b n d c', rotations_q.transpose(-1, -2), results_points - translations_q[:,:,0,None,:])
            results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + eps)

        # merge back heads
        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h = h)
        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h = h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h = h)

        results = (results_scalar, results_points, results_points_norm)

        # concat results and project out
        results = torch.cat(results, dim = -1)
        return self.to_out(results)

class IPABlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        dropout=0.,
        kv_dim=None,
        ff_mult = 1,
        ff_num_layers = 3,          # in the paper, they used 3 layer transition (feedforward) block
        post_norm = True,           # in the paper, they used post-layernorm - offering pre-norm as well
        post_attn_dropout = 0.,
        post_ff_dropout = 0.,
        attention_module = InvariantPointAttention,
        point_dim = 4,
        use_adaln = False,
        **kwargs
    ):
        super().__init__()
        self.post_norm = post_norm

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = attention_module(dim, heads=heads, dim_head=dim_head,
                dropout=dropout, kv_dim=kv_dim, point_dim=point_dim, use_adaln=use_adaln)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult = ff_mult, num_layers = ff_num_layers)
        self.post_ff_dropout = nn.Dropout(post_ff_dropout)

    def forward(self, tgt, memory=None, geometric_args=None,  mask=None, diff_ts=None, **kwargs):
        post_norm = self.post_norm

        attn_input = tgt if post_norm else self.attn_norm(tgt)
        tgt = self.attn(attn_input, memory=memory, geometric_args=geometric_args, mask=mask, diff_ts=diff_ts, **kwargs) + tgt
        tgt = self.post_attn_dropout(tgt)
        tgt = self.attn_norm(tgt) if post_norm else tgt

        ff_input = tgt if post_norm else self.ff_norm(tgt)
        tgt = self.ff(ff_input) + tgt
        tgt = self.post_ff_dropout(tgt)
        tgt = self.ff_norm(tgt) if post_norm else tgt
        return tgt

# add an IPA Transformer - iteratively updating rotations and translations

# this portion is not accurate to AF2, as AF2 applies a FAPE auxiliary loss on each layer, as well as a stop gradient on the rotations
# just an attempt to see if this could evolve to something more generally usable

class InvariantPointTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads, dim_head,
        kv_dim=None,
        dropout=0.,
        attention_module=InvariantPointAttention,
        point_dim=4,
        use_adaln=False,
        **kwargs
    ):
        super().__init__()


        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                IPABlock(dim=dim, heads = heads,dim_head = dim_head,
                         dropout = dropout, kv_dim=kv_dim,
                         attention_module=attention_module,
                         point_dim=point_dim, use_adaln=use_adaln))

        # whether to detach rotations or not, for stability during training
        self.to_points = nn.Linear(dim, dim)

    def forward(self, tgt, memory=None, geometric_args=None, mask=None, diff_ts=None):

        # go through the layers and apply invariant point attention and feedforward
        for block in self.layers:
            tgt = block(tgt, memory=memory, geometric_args=geometric_args, mask=mask, diff_ts=diff_ts)

        return self.to_points(tgt)

