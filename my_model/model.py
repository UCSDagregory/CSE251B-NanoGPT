import torch
import torch.nn as nn
import os
import inspect
from torch.nn import functional as F
from typing import Any
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


MODEL_CONFIG = "model_config"
OPT_CONFIG = "optimizer_config"
MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
ITER_NUM = "iter_num"
CHECKPOINT_DEFAULT = "checkpoints"
CHECKPOINT_EXT = ".pt"

# --- RoPE (Rotary Position Embeddings) ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- RMSNorm ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# --- Mixture of Experts (MoE) Components ---
class SwiGLUExpert(nn.Module):
    """A LLaMA-style SwiGLU feed-forward network acting as a single expert."""
    def __init__(self, n_embd):
        super().__init__()
        # Standard hidden is 4*n_embd. SwiGLU typically uses 8/3 to match parameter counts.
        hidden_dim = int(8 * n_embd / 3) 
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SparseMoE(nn.Module):
    """Routes tokens to the Top-K experts and computes Auxiliary Load Balancing Loss."""
    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(n_embd) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape

        # 1. Get routing probabilities
        router_logits = self.router(x)
        routing_probs = F.softmax(router_logits, dim=-1)

        # --- Aux Loss Calculation ---
        mean_probs = routing_probs.mean(dim=(0, 1))
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # Fraction of tokens routed to each expert (flattened to 1D for bincount)
        flat_indices_1d = top_k_indices.view(-1)
        counts = torch.bincount(flat_indices_1d, minlength=self.num_experts).to(x.dtype)
        route_frac = counts / (B * T * self.top_k)

        # Load balancing loss
        aux_loss = self.num_experts * torch.sum(mean_probs * route_frac)
        # ----------------------------

        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True) # Normalize

        # 3. Route tokens
        flat_x = x.view(-1, C) # [B*T, C]
        flat_indices = top_k_indices.view(-1, self.top_k) # [B*T, top_k]
        flat_weights = top_k_weights.view(-1, self.top_k) # [B*T, top_k]
        flat_output = torch.zeros_like(flat_x) # [B*T, C]

        for i, expert in enumerate(self.experts):
            # Mask of shape [B*T, top_k]
            expert_mask = (flat_indices == i)

            # Mask of shape [B*T] indicating if token goes to expert 'i'
            token_mask = expert_mask.any(dim=-1)

            if token_mask.any():
                expert_inputs = flat_x[token_mask]
                expert_outputs = expert(expert_inputs)

                # Extract routing weights for this expert and sum over top_k dimension
                weights = (flat_weights[token_mask] * expert_mask[token_mask].float()).sum(dim=-1, keepdim=True)

                # Apply weights and accumulate
                flat_output[token_mask] += expert_outputs * weights

        return flat_output.view(B, T, C), aux_loss

class CausalSelfAttention(nn.Module):
    """Standard Multi-Head Attention using efficient Flash Attention + RoPE."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for RoPE
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Transpose for Flash Attention
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # F.scaled_dot_product_attention natively handles the causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MoEBlock(nn.Module):
    """Replaces nn.TransformerEncoderLayer. Combines Attention, Shared Expert, and Sparse MoE."""
    def __init__(self, n_embd, n_head, num_experts, top_k):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = RMSNorm(n_embd)
        
        # Shared Expert (always active) + Sparse Router
        self.shared_expert = SwiGLUExpert(n_embd)
        self.moe = SparseMoE(n_embd, num_experts, top_k)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        
        # Forward pass through normalized state
        norm_x = self.ln_2(x)
        shared_out = self.shared_expert(norm_x)
        sparse_out, aux_loss = self.moe(norm_x)
        
        x = x + shared_out + sparse_out
        return x, aux_loss

# --- Main Model ---
class nanoGPT(nn.Module):
    def __init__(self, model_folder_name:str, chkpt_folder_name:str=None,
                 author:str="N/A",
                 vocab_size=50257, n_embd=128, n_head=4, n_layer=2, block_size=1024,
                 num_experts=8, top_k=2, aux_loss_coef=0.01): 

        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(n_embd // n_head, block_size * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # custom MoE blocks
        self.blocks = nn.ModuleList([
            MoEBlock(n_embd, n_head, num_experts, top_k) for _ in range(n_layer)
        ])

        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight 

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

        # MoE params
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        self.num_parameters = sum(val.numel() for val in self.parameters())
        self.author = author
        self.model_path = os.path.join(os.getcwd(), model_folder_name)
        if (chkpt_folder_name is None):
            self.checkpoint_folder_name = CHECKPOINT_DEFAULT
        else:
            self.checkpoint_folder_name = chkpt_folder_name

        self.opt_weight_decay = 0
        self.opt_learning_rate = 0
        self.opt_betas = 0
        self.opt_device_type = 0
        self.opt_type = "muon"

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.token_emb(input_ids)
        
        freqs_cis = self.freqs_cis[:T]

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x, freqs_cis)
            total_aux_loss += aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # Main CE Loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # Add scaled auxiliary loss
            loss = loss + (self.aux_loss_coef * total_aux_loss)
            return logits, loss

        return logits

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.num_parameters
        L, H, Q, T = self.n_layer, self.n_head, self.n_embd//self.n_head, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) 
        flops_promised = 312e12 
        mfu = flops_achieved / flops_promised
        return mfu

    def getCheckpointPath(self):
        full_path = os.path.join(self.model_path, self.checkpoint_folder_name)
        return full_path

    def saveCheckpoint(self, optimizer:nn.Module, val_loss:int, iter_num, non_eval_checkpoint=False):
        checkpoint = {
            MODEL_CONFIG: {
                "author": self.author,
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
                "block_size": self.block_size,
                "num_experts": self.num_experts, 
                "top_k": self.top_k,
                "aux_loss_coef": self.aux_loss_coef
            },
            OPT_CONFIG:{
                "weight_decay":self.opt_weight_decay,
                "learning_rate":self.opt_learning_rate,
                "betas":self.opt_betas,
                "optimizer":self.opt_type,
                "device_type":self.opt_device_type,
            },
            MODEL_STATE_DICT: self.state_dict(),
            OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            ITER_NUM: iter_num,
        }
        save_file_name = f"ITER{iter_num}_{val_loss:08.4f}val_loss" + "_" + nanoGPT.__name__ + "_" + self.author.replace(" ", "") + CHECKPOINT_EXT
        if (non_eval_checkpoint):
            save_file_name = f"ITER{iter_num}_CHKPT{val_loss:08.4f}val_loss" + "_" + nanoGPT.__name__ + "_" + self.author.replace(" ", "") + CHECKPOINT_EXT

        full_checkpoint_path = os.path.join(self.getCheckpointPath(), save_file_name)
        torch.save(checkpoint, full_checkpoint_path)

    def configure_optimizers(self, weight_decay, learning_rate, betas, optimizer_type, device_type):
        self.opt_weight_decay = weight_decay
        self.opt_learning_rate = learning_rate
        self.opt_betas = betas
        self.opt_type = optimizer_type
        self.opt_device_type = device_type

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if optimizer_type == "muon":
            print("Using muon optimizer")

            hidden_weights = []
            hidden_gains_biases = []
            nonhidden_decay = []
            nonhidden_no_decay = []

            # Added SwiGLU projection names to Muon path
            nonhidden_keywords = (
                "embed", "embedding", "wte", "wpe",
                "head", "lm_head", "output", "classifier",
                "router", "freqs_cis"
            )

            seen = set()

            for name, p in param_dict.items():
                if id(p) in seen:
                    continue
                seen.add(id(p))

                lname = name.lower()

                if any(k in lname for k in nonhidden_keywords):
                    if p.ndim >= 2:
                        nonhidden_decay.append(p)      
                    else:
                        nonhidden_no_decay.append(p)   
                elif p.ndim >= 2:
                    hidden_weights.append(p)
                else:
                    hidden_gains_biases.append(p)

            num_hidden_weights = sum(p.numel() for p in hidden_weights)
            num_hidden_gains_biases = sum(p.numel() for p in hidden_gains_biases)
            num_nonhidden_params = sum(p.numel() for p in nonhidden_decay)

            print(f"num muon parameter tensors: {len(hidden_weights)}, with {num_hidden_weights:,} parameters")
            print(f"num hidden gain/bias tensors: {len(hidden_gains_biases)}, with {num_hidden_gains_biases:,} parameters")
            print(f"num nonhidden parameter tensors: {len(nonhidden_decay)}, with {num_nonhidden_params:,} parameters")

            hidden_lr,nonhidden_lr = learning_rate[0], learning_rate[1]
            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=hidden_lr,
                    weight_decay=weight_decay,
                ),
                dict(
                    params=hidden_gains_biases + nonhidden_no_decay,
                    use_muon=False,
                    lr=nonhidden_lr,
                    betas=betas,
                    weight_decay=0.0,
                ),
                dict(
                    params=nonhidden_decay,
                    use_muon=False,
                    lr=nonhidden_lr,
                    betas=betas,
                    weight_decay=weight_decay,  
                ),
            ]

            use_distributed_muon = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )

            if use_distributed_muon:
                print("Using distributed MuonWithAuxAdam")
                optimizer = MuonWithAuxAdam(param_groups)
            else:
                print("Using SingleDeviceMuonWithAuxAdam")
                optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

            return optimizer

        if optimizer_type == "adam":
            decay_params = []
            nodecay_params = []
            seen = set()

            for _, p in param_dict.items():
                if id(p) in seen:
                    continue
                seen.add(id(p))

                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    nodecay_params.append(p)

            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]

            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == "cuda"
            extra_args = dict(fused=True) if use_fused else dict()

            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=learning_rate,
                betas=betas,
                **extra_args,
            )
            print(f"using fused AdamW: {use_fused}")

            return optimizer

        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

def getArgs(checkpoint, model_folder_name="N/A", chkpt_folder_name="N/A"):
    model_args = [model_folder_name, chkpt_folder_name]
    model_saved_config = checkpoint[MODEL_CONFIG]
    for key in model_saved_config:
        model_args.append(model_saved_config[key])
    opt_args = []
    opt_saved_config = checkpoint[OPT_CONFIG]
    for key in opt_saved_config:
        opt_args.append(opt_saved_config[key])
    return model_args, opt_args

def loadFromCheckpoint(model_folder_name:str, checkpoint_file_path:str) -> tuple[nn.Module, Any, Any, Any, int]:
    split_path = checkpoint_file_path.split('/')
    if (len(split_path) != 2):
        raise ValueError("Checkpoint path should only be folder_name/checkpoint_to_load.ext")
    chkpt_folder_name, ckpt_file_name  = split_path
    load_path = os.path.join(os.getcwd(), model_folder_name, chkpt_folder_name, ckpt_file_name)
    checkpoint = torch.load(load_path, weights_only=True)
    model_args, opt_args = getArgs(checkpoint, model_folder_name, chkpt_folder_name)

    gpt_model = nanoGPT(*model_args)
    gpt_model.checkpoint_folder_name = chkpt_folder_name

    model_sd = checkpoint[MODEL_STATE_DICT]
    opt_sd = checkpoint[OPTIMIZER_STATE_DICT]
    iter_num = checkpoint[ITER_NUM]
    checkpoint = None
    return gpt_model, model_sd, opt_args, opt_sd, iter_num

def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_args, opt_args = getArgs(checkpoint)
    model = nanoGPT(*model_args)
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    checkpoint = None
    print(f"#Params: {model.num_parameters}")
    model.to(device)
    model.eval()
    return model
