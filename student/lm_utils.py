import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device=device, dtype=dtype))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return einsum(input, self.weight, '... in_features, out_features in_features -> ... out_features')

# class Linear(nn.Module):
    
#     def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
#         super(Linear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
#         self.weight = self.linear.weight
#     def forward(self, x):
#         return self.linear(x)
    

class Embedding(nn.Module):
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim).to(device=device, dtype=dtype))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids, :]
    
class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        normalized_input = x / rms
        out = normalized_input * self.weight
        out = out.to(in_type)

        return out

class SiLU(nn.Module):
    
    def __init__(self):
        super(SiLU, self).__init__()
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return in_features / (1 + torch.exp(-in_features))


class Swiglu(nn.Module):
    
    def __init__(self, d_ff: int, d_model: int, device=None, dtype=None):
        super(Swiglu, self).__init__()
        self.w1 = nn.Parameter(torch.Tensor(d_ff, d_model).to(device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.Tensor(d_model, d_ff).to(device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.Tensor(d_ff, d_model).to(device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einsum(x, self.w1, '... d_model, d_ff d_model -> ... d_ff')
        x_silu = x1 / (1 + torch.exp(-x1))
        x3 = einsum(x, self.w3, '... d_model, d_ff d_model -> ... d_ff')
        x_silu_3 = x_silu * x3
        out = einsum(self.w2, x_silu_3, 'd_model d_ff, ... d_ff -> ... d_model')

        return out
    
class RoPE(nn.Module):
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, dtype=None, device=None):
        super(RoPE, self).__init__()
        self.dim = d_k
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k)) # (d_k/2,)
        inv_freq = rearrange(inv_freq, 'd -> 1 d')  # (1, d_k/2)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = rearrange(token_positions, '... -> ... 1')
        inv_freq_i = einsum(token_positions, self.inv_freq, '... l, l d -> ... d')  # (..., d_k/2)
        R_i = torch.stack([
            torch.stack([torch.cos(inv_freq_i), -torch.sin(inv_freq_i)], dim=-1),
            torch.stack([torch.sin(inv_freq_i), torch.cos(inv_freq_i)], dim=-1)
        ], dim=-1)  # (..., d_k/2, 2, 2)

        elements_per_R = self.dim // 2
        rorate_d = 2
        x_rearranged = rearrange(x, '... (elements_per_R rotate_d) -> ... elements_per_R rotate_d', elements_per_R=elements_per_R, rotate_d=rorate_d)
        rotated_x = einsum(R_i, x_rearranged, '... d i j, ... d i -> ... d j')
        out_x = rearrange(rotated_x, '... elements_per_R rotate_d -> ... (elements_per_R rotate_d)')
        return out_x
    
class Softmax(nn.Module):
    
    def __init__(self, dim: int = -1):
        super(Softmax, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_x = torch.exp(x - torch.max(x, dim=self.dim, keepdim=True).values)
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp_x
    
class Attention(nn.Module):
    
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = Softmax(dim=-1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        E = einsum(Q, K, '... q d, ... k d -> ... q k')/torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))  # /sqrt(d_k)
        E = E.masked_fill(mask == False, float('-inf'))
        A = self.softmax(E)
        attn_out = einsum(A, V, '... q k, ... k d -> ... q d')
        return attn_out

class MultiheadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = nn.Parameter(torch.Tensor(d_model, d_model).to(device=device, dtype=dtype))
        self.W_k = nn.Parameter(torch.Tensor(d_model, d_model).to(device=device, dtype=dtype))
        self.W_v = nn.Parameter(torch.Tensor(d_model, d_model).to(device=device, dtype=dtype))
        self.W_o = nn.Parameter(torch.Tensor(d_model, d_model).to(device=device, dtype=dtype))

        self.attention = Attention()

    def forward(self, x: torch.Tensor, rope: RoPE = None, token_positions: torch.Tensor = None) -> torch.Tensor:

        mask = torch.tril(torch.ones((x.shape[-2], x.shape[-2]), dtype=torch.bool))
        
        Q = einsum(x, self.W_q, '... seq_len d_model, d_model_out d_model -> ... seq_len d_model_out')
        K = einsum(x, self.W_k, '... seq_len d_model, d_model_out d_model -> ... seq_len d_model_out')
        V = einsum(x, self.W_v, '... seq_len d_model, d_model_out d_model -> ... seq_len d_model_out')

        Q = rearrange(Q, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        K = rearrange(K, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        V = rearrange(V, '... seq_len (num_heads d_v) -> ... num_heads seq_len d_v', num_heads=self.num_heads)

        if rope is not None:
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        attn_out = self.attention(Q, K, V, mask)
        attn_out = rearrange(attn_out, '... num_heads seq_len d_v -> ... seq_len (num_heads d_v)')

        out = einsum(attn_out, self.W_o, '... seq_len d_model, d_model_out d_model -> ... seq_len d_model_out')

        return out
    
class TransformerBlock(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, eps=1e-5, device=None, dtype=None):
        super(TransformerBlock, self).__init__()
        self.mha = MultiheadAttention(d_model, num_heads, device=device, dtype=dtype)
        self.rms1 = RMSNorm(d_model, eps)
        self.swiglu = Swiglu(d_ff, d_model, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model, eps)
    
    def forward(self, x: torch.Tensor, rope: RoPE = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.mha(self.rms1(x), rope=rope, token_positions=token_positions)
        x = x + attn_out
        ff_out = self.swiglu(self.rms2(x))
        x = x + ff_out
        return x
    
class TransformerLM(nn.Module):
    
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, eps=1e-5, rope_theta: float = 10000.0, device=None, dtype=None):
        super(TransformerLM, self).__init__()
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.position_ids = torch.arange(self.context_length, device=device).unsqueeze(0)
        self.rope = RoPE(theta=self.rope_theta, d_k=d_model // num_heads, max_seq_len=self.context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, eps) for _ in range(num_layers)
        ])
        self.rms_final = RMSNorm(d_model, eps)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        token_positions = self.position_ids[..., :input_ids.shape[-1]]

        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)
        x = self.rms_final(x)
        logits = self.output_linear(x)
        return logits