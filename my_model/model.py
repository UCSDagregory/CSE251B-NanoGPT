import torch
import torch.nn as nn
import os
import math
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

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, slightly faster than LayerNorm)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.training and self.attn_dropout > 0:
                att = F.dropout(att, p=self.attn_dropout)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class nanoGPT(nn.Module):
    def __init__(
        self,
        model_folder_name: str,
        chkpt_folder_name: str = None,
        author: str = "N/A",
        vocab_size: int = 50257,
        n_embd: int = 720,
        n_head: int = 12,
        n_layer: int = 10,
        block_size: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        self.num_parameters = sum(val.numel() for val in self.parameters())
        self.author = author
        self.model_path = os.path.join(os.getcwd(), model_folder_name)
        self.checkpoint_folder_name = chkpt_folder_name if chkpt_folder_name else CHECKPOINT_DEFAULT

        self.opt_weight_decay = 0
        self.opt_learning_rate = 0
        self.opt_betas = 0
        self.opt_device_type = 0
        self.opt_type = "adam"

        print(f"Model initialized: {self.num_parameters / 1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.num_parameters
        L, H, Q, T = self.n_layer, self.n_head, self.n_embd // self.n_head, self.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    def getCheckpointPath(self):
        return os.path.join(self.model_path, self.checkpoint_folder_name)

    def saveCheckpoint(self, optimizer: nn.Module, val_loss: int, iter_num=0, non_eval_checkpoint=False):
        checkpoint = {
            MODEL_CONFIG: {
                "author": self.author,
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
                "block_size": self.block_size,
                "dropout": 0.0,
            },
            OPT_CONFIG: {
                "weight_decay": self.opt_weight_decay,
                "learning_rate": self.opt_learning_rate,
                "betas": self.opt_betas,
                "optimizer": self.opt_type,
                "device_type": self.opt_device_type,
            },
            MODEL_STATE_DICT: self.state_dict(),
            OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            ITER_NUM: iter_num,
        }
        save_file_name = (
            f"ITER{iter_num}_{val_loss:08.4f}val_loss_{nanoGPT.__name__}_{self.author.replace(' ', '')}{CHECKPOINT_EXT}"
        )
        if non_eval_checkpoint:
            save_file_name = (
                f"ITER{iter_num}_CHKPT{val_loss:08.4f}val_loss_{nanoGPT.__name__}_{self.author.replace(' ', '')}{CHECKPOINT_EXT}"
            )
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
            nonhidden_params = []

            nonhidden_keywords = (
                "embed", "embedding", "wte", "wpe",
                "head", "lm_head", "output", "classifier",
            )

            seen = set()

            for name, p in param_dict.items():
                if id(p) in seen:
                    continue
                seen.add(id(p))

                lname = name.lower()

                if any(k in lname for k in nonhidden_keywords):
                    nonhidden_params.append(p)
                elif p.ndim >= 2:
                    hidden_weights.append(p)
                else:
                    hidden_gains_biases.append(p)

            num_hidden_weights = sum(p.numel() for p in hidden_weights)
            num_hidden_gains_biases = sum(p.numel() for p in hidden_gains_biases)
            num_nonhidden_params = sum(p.numel() for p in nonhidden_params)

            print(f"num muon parameter tensors: {len(hidden_weights)}, with {num_hidden_weights:,} parameters")
            print(f"num hidden gain/bias tensors: {len(hidden_gains_biases)}, with {num_hidden_gains_biases:,} parameters")
            print(f"num nonhidden parameter tensors: {len(nonhidden_params)}, with {num_nonhidden_params:,} parameters")

            hidden_lr, nonhidden_lr = learning_rate[0], learning_rate[1]
            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=hidden_lr,
                    weight_decay=weight_decay,
                ),
                dict(
                    params=hidden_gains_biases + nonhidden_params,
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

        # default AdamW path
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
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")

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


# ---------------------------------------------------------------------------
# Checkpoint helpers (required by train.py / train_helper.py)
# ---------------------------------------------------------------------------

def getArgs(checkpoint, model_folder_name="N/A", chkpt_folder_name="N/A"):
    model_args = [model_folder_name, chkpt_folder_name]
    for key in checkpoint[MODEL_CONFIG]:
        model_args.append(checkpoint[MODEL_CONFIG][key])
    opt_args = []
    for key in checkpoint[OPT_CONFIG]:
        opt_args.append(checkpoint[OPT_CONFIG][key])
    return model_args, opt_args


def loadFromCheckpoint(model_folder_name: str, checkpoint_file_path: str) -> tuple:
    split_path = checkpoint_file_path.split("/")
    if len(split_path) != 2:
        raise ValueError("Checkpoint path should be folder_name/checkpoint_to_load.ext")
    chkpt_folder_name, ckpt_file_name = split_path
    load_path = os.path.join(os.getcwd(), model_folder_name, chkpt_folder_name, ckpt_file_name)
    checkpoint = torch.load(load_path, weights_only=True)
    model_args, opt_args = getArgs(checkpoint, model_folder_name, chkpt_folder_name)
    gpt_model = nanoGPT(*model_args)
    gpt_model.checkpoint_folder_name = chkpt_folder_name
    model_sd = checkpoint[MODEL_STATE_DICT]
    opt_sd = checkpoint[OPTIMIZER_STATE_DICT]
    iter_num = checkpoint.get(ITER_NUM, 0)
    checkpoint = None
    return gpt_model, model_sd, opt_args, opt_sd, iter_num


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load trained model from checkpoint. Called by evaluate.py.
    Returns model where: model(input_ids) -> logits
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_args, _ = getArgs(checkpoint)
    model = nanoGPT(*model_args)
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    checkpoint = None
    print(f"#Params: {model.num_parameters}")
    model.to(device)
    model.eval()
    return model