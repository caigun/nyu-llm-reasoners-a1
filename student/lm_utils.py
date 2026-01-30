import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return einsum(input, self.weight, '... in_features, out_features in_features -> ... out_features')
    

class Embedding(nn.Module):
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids, :]
    
class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, eps=1e-8, device=None, dtype=None):
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
    
class Swiglu(nn.Module):
    
    def __init__(self, d_ff: int, d_model: int):
        super(Swiglu, self).__init__()
        self.w1 = nn.Parameter(torch.Tensor(d_ff, d_model))
        self.w2 = nn.Parameter(torch.Tensor(d_model, d_ff))
        self.w3 = nn.Parameter(torch.Tensor(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einsum(x, self.w1, '... d_model, d_ff d_model -> ... d_ff')
        x_silu = x1 / (1 + torch.exp(-x1))
        x3 = einsum(x, self.w3, '... d_model, d_ff d_model -> ... d_ff')
        x_silu_3 = x_silu * x3
        out = einsum(self.w2, x_silu_3, 'd_model d_ff, ... d_ff -> ... d_model')

        return out
    
class RoPE(nn.Module):
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
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