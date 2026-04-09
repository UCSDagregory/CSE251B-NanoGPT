import torch
import torch.nn as nn
import os
import math
from torch.nn import functional as F

MODEL_CONFIG = "model_config"
OPT_CONFIG = "optimizer_config"
MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
CHECKPOINT_EXT = ".pt"


# ---------------------------------------------------------------------------
# Muon Optimizer
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-schulz.
    Apply to 2-D weight matrices in attn/MLP only.
    Use AdamW for embeddings, norms, and biases.
    Reference: https://github.com/KellerJordan/modded-nanogpt
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, nesterov, ns_steps = (
                group['lr'], group['momentum'], group['nesterov'], group['ns_steps']
            )
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                if update.ndim >= 2:
                    update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
                    p.add_(update, alpha=-lr * max(update.size(0), update.size(1)) ** 0.5)
                else:
                    p.add_(update, alpha=-lr)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _precompute_rope(head_dim: int, block_size: int, base: float = 10000.0):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(block_size).float(), theta)  # (T, head_dim//2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim)"""
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([x_even * cos - x_odd * sin,
                        x_even * sin + x_odd * cos], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.c_attn   = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj   = nn.Linear(n_embd, n_embd,     bias=False)
        cos, sin = _precompute_rope(self.head_dim, block_size)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, self.rope_cos, self.rope_sin)
        k = _apply_rope(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # pre-norm
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class nanoGPT(nn.Module):
    def __init__(self, model_folder_name: str, author: str = "N/A",
                 vocab_size: int = 50257, n_embd: int = 768, n_head: int = 12,
                 n_layer: int = 8, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer    = n_layer
        self.n_head     = n_head
        self.n_embd     = n_embd
        self.author     = author

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks    = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f      = nn.LayerNorm(n_embd)
        self.lm_head   = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self.apply(self._init_weights)
        self.num_parameters = sum(v.numel() for v in self.parameters())
        self.checkpoint_path = os.path.join(os.getcwd(), model_folder_name, "checkpoints")
        self.opt_weight_decay  = 0
        self.opt_learning_rate = 0
        self.opt_betas         = 0
        self.opt_device_type   = 0
        self.opt_muon_lr       = 0.02

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.num_parameters
        L, H, Q, T = self.n_layer, self.n_head, self.n_embd // self.n_head, self.block_size
        flops_per_iter = (6 * N + 12 * L * H * Q * T) * T * fwdbwd_per_iter
        return flops_per_iter / (dt * 312e12)

    def saveCheckpoint(self, optimizer, val_loss: float):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        opt_sd = [o.state_dict() for o in optimizer] if isinstance(optimizer, (list, tuple)) \
                 else optimizer.state_dict()
        checkpoint = {
            MODEL_CONFIG: {
                "author":     self.author,
                "vocab_size": self.vocab_size,
                "n_embd":     self.n_embd,
                "n_head":     self.n_head,
                "n_layer":    self.n_layer,
                "block_size": self.block_size,
            },
            OPT_CONFIG: {
                "weight_decay":  self.opt_weight_decay,
                "learning_rate": self.opt_learning_rate,
                "betas":         self.opt_betas,
                "device_type":   self.opt_device_type,
                "muon_lr":       self.opt_muon_lr,
            },
            MODEL_STATE_DICT:     self.state_dict(),
            OPTIMIZER_STATE_DICT: opt_sd,
        }
        fname = f"{val_loss:08.4f}val_loss_{nanoGPT.__name__}_{self.author.replace(' ', '')}{CHECKPOINT_EXT}"
        torch.save(checkpoint, os.path.join(self.checkpoint_path, fname))

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, muon_lr=0.02):
        """Returns (muon_optimizer, adamw_optimizer)."""
        self.opt_weight_decay  = weight_decay
        self.opt_learning_rate = learning_rate
        self.opt_betas         = betas
        self.opt_device_type   = device_type
        self.opt_muon_lr       = muon_lr

        muon_params, adamw_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and 'emb' not in name:
                muon_params.append(param)   # 2-D attn/MLP weights → Muon
            else:
                adamw_params.append(param)  # embeddings, norms → AdamW

        muon  = Muon(muon_params, lr=muon_lr, momentum=0.95)
        adamw = torch.optim.AdamW(adamw_params, lr=learning_rate,
                                  betas=betas, weight_decay=weight_decay)
        print(f"Muon params: {sum(p.numel() for p in muon_params):,} | "
              f"AdamW params: {sum(p.numel() for p in adamw_params):,}")
        return muon, adamw


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def loadFromCheckpoint(model_folder_name: str, checkpoint_file_path: str):
    load_path = os.path.join(os.getcwd(), model_folder_name, checkpoint_file_path + CHECKPOINT_EXT)
    ckpt = torch.load(load_path, weights_only=True)
    cfg = ckpt[MODEL_CONFIG]
    model = nanoGPT(model_folder_name, author=cfg["author"], vocab_size=cfg["vocab_size"],
                    n_embd=cfg["n_embd"], n_head=cfg["n_head"],
                    n_layer=cfg["n_layer"], block_size=cfg["block_size"])
    opt_cfg  = ckpt[OPT_CONFIG]
    opt_args = [opt_cfg["weight_decay"], opt_cfg["learning_rate"],
                opt_cfg["betas"],        opt_cfg["device_type"],
                opt_cfg.get("muon_lr", 0.02)]
    return model, ckpt[MODEL_STATE_DICT], opt_args, ckpt[OPTIMIZER_STATE_DICT]


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Required interface for evaluate.py: input_ids (B,T) -> logits (B,T,50257)"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg  = ckpt[MODEL_CONFIG]
    model = nanoGPT(model_folder_name="", author=cfg.get("author", "N/A"),
                    vocab_size=cfg["vocab_size"], n_embd=cfg["n_embd"],
                    n_head=cfg["n_head"], n_layer=cfg["n_layer"], block_size=cfg["block_size"])
    model.load_state_dict(ckpt[MODEL_STATE_DICT])
    model.to(device)
    model.eval()
    return model
