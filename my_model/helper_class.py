import torch.nn as nn
import torch

try:
    from mamba_ssm import Mamba as _MambaImpl
    _MAMBA_BACKEND = "mamba_ssm"
except ImportError:
    from mambapy.mamba import Mamba as _MambaPy, MambaConfig
    _MAMBA_BACKEND = "mambapy"

print(f"[MambaBlock] backend: {_MAMBA_BACKEND}")


#This grabs the relative position of each word instead of its absolute position , called RoPE
class RotaryPositionalEmbeddings(nn.Module):

  def __init__(self, d: int, base: int = 10_000):

    super().__init__()
    self.base = base
    self.d = d
    self.cos_cached = None
    self.sin_cached = None

  def _build_cache(self, x: torch.Tensor):

    if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
      return

    seq_len = x.shape[0]

    theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device) # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)

    seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device) #Position Index -> [0,1,2...seq-1]

    idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1) # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]


    self.cos_cached = idx_theta2.cos()[:, None, None, :] #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
    self.sin_cached = idx_theta2.sin()[:, None, None, :] #cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

  def _neg_half(self, x: torch.Tensor):

    d_2 = self.d // 2 #

    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]


  def forward(self, x: torch.Tensor):

    self._build_cache(x)

    neg_half_x = self._neg_half(x)

    x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]) # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

    return x_rope
  

class CausalSelfAttentionBlock(nn.Module):
    """
    A pre-norm causal multi-head self-attention block with RoPE applied to Q and K.
    Replaces nn.TransformerEncoderLayer so we can inject RoPE inside attention.
 
    Layout per block:
        x -> LayerNorm -> MHA (RoPE on Q,K) -> Dropout -> residual add
          -> LayerNorm -> MLP (Linear -> GELU -> Linear) -> Dropout -> residual add
    """
 
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
 
        self.n_head   = n_head
        self.head_dim = d_model // n_head
 
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
 
        # RoPE operates on each head's query/key vectors (head_dim)
        self.rope = RotaryPositionalEmbeddings(self.head_dim)
 
        # Multi Layer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(), #TODO: can we do some other one here?
            nn.Linear(4 * d_model, d_model),
        )
 
        # Norms + dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)  — batch_first convention
        Returns:
            x: (batch, seq, d_model)
        """
        B, T, C = x.shape
 
        # ---- Self-attention with RoPE ----
        residual = x
        x = self.norm1(x)
 
        # Project to Q, K, V and split into heads
        # (B, T, C) -> (B, T, n_head, head_dim) -> (T, B, n_head, head_dim) for RoPE
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).permute(1, 0, 2, 3)
 
        q = split_heads(self.q_proj(x))  # (T, B, n_head, head_dim)
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
 
        # Apply RoPE to Q and K only
        q = self.rope(q)
        k = self.rope(k)
 
        # Reshape for scaled dot-product attention: (B, n_head, T, head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
 
        # Causal scaled dot-product attention (uses Flash Attention when available)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )  # (B, n_head, T, head_dim)
 
        # Merge heads back: (B, T, C)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        x = residual + self.drop(self.out_proj(attn_out))
 
        # ---- MLP ----
        x = x + self.drop(self.mlp(self.norm2(x)))
 
        return x
 

#1 block layer of Mamba SSM
class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.silu = nn.SiLU()
        if _MAMBA_BACKEND == "mamba_ssm":
            # mamba-ssm: just the SSM layer, no internal norm/residual
            self.mamba = _MambaImpl(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            # mambapy: wraps a 1-layer model
            cfg = MambaConfig(d_model=d_model, n_layers=1, d_state=d_state, d_conv=d_conv, expand_factor=expand)
            self.mamba = _MambaPy(cfg)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mamba(self.norm(x)))