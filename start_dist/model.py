import os
import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


MODEL_CONFIG = "model_config"
OPT_CONFIG = "optimizer_config"
MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
ITER_NUM = "iter_num"
CHECKPOINT_DEFAULT = "checkpoints"
CHECKPOINT_EXT = ".pt"
MAX_ZEROS_IN_CKPTFN = 8


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {dim}")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)

        cos = self.cos_cached[:seq_len].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[:seq_len].to(device=q.device, dtype=q.dtype)

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        q_even = q[..., ::2]
        q_odd = q[..., 1::2]
        k_even = k[..., ::2]
        k_odd = k[..., 1::2]

        q_rot = torch.empty_like(q)
        k_rot = torch.empty_like(k)

        q_rot[..., ::2] = q_even * cos - q_odd * sin
        q_rot[..., 1::2] = q_even * sin + q_odd * cos

        k_rot[..., ::2] = k_even * cos - k_odd * sin
        k_rot[..., 1::2] = k_even * sin + k_odd * cos

        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        bias: bool = False,
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.attn_dropout = float(attn_dropout)

        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {self.head_dim}")

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=block_size,
            base=rope_base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.block_size}")

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(channels, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(
        self,
        n_embd: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int((8 * n_embd) / 3)
            hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)

        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GPTBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        mlp_hidden_dim: int | None = None,
        bias: bool = False,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            bias=bias,
            rope_base=rope_base,
            attn_dropout=attn_dropout,
        )

        self.mlp_norm = RMSNorm(n_embd)
        self.mlp = SwiGLU(
            n_embd=n_embd,
            hidden_dim=mlp_hidden_dim,
            bias=bias,
        )

        # Residual-branch dropout.
        #
        # This is the safest first dropout test for a late-stage checkpoint:
        # it regularizes the residual updates without dropping token embeddings
        # and without directly perturbing attention probabilities unless
        # attn_dropout is set separately.
        self.resid_dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.attn_norm(x)))
        x = x + self.resid_dropout(self.mlp(self.mlp_norm(x)))
        return x


class nanoGPT(nn.Module):
    def __init__(
        self,
        model_folder_name: str,
        chkpt_folder_name: str = None,
        author: str = "N/A",
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=1024,
        resume=False,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)

        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # No learned absolute position embeddings. RoPE is applied to q/k inside attention.
        self.pos_emb = None

        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    bias=False,
                    rope_base=10000.0,
                    dropout=self.dropout,
                    attn_dropout=self.attn_dropout,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        if not resume:
            self.apply(self._init_weights)

            # GPT-2/LLaMA-style residual projection scaling.
            for block in self.blocks:
                nn.init.normal_(
                    block.attn.out_proj.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * n_layer),
                )

                nn.init.normal_(
                    block.mlp.down_proj.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * n_layer),
                )

        # Tie token embedding and output head after initialization.
        self.lm_head.weight = self.token_emb.weight

        self.num_parameters = sum(p.numel() for p in self.parameters())

        self.author = author
        self.model_path = os.path.join(os.getcwd(), model_folder_name)

        if chkpt_folder_name is None:
            self.checkpoint_folder_name = CHECKPOINT_DEFAULT
        else:
            self.checkpoint_folder_name = chkpt_folder_name

        self.opt_weight_decay = 0
        self.opt_learning_rate = 0
        self.opt_betas = 0
        self.opt_type = "muon"
        self.backoff_thresh = None
        self.backoff_rates = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids, targets=None):
        _, seq_len = input_ids.shape

        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.block_size}")

        x = self.token_emb(input_ids)

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
        L = self.n_layer
        H = self.n_head
        Q = self.n_embd // self.n_head
        T = self.block_size

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)

        flops_promised = 312e12
        return flops_achieved / flops_promised

    def getCheckpointPath(self):
        return os.path.join(self.model_path, self.checkpoint_folder_name)

    def saveCheckpoint(self, optimizer: nn.Module, val_loss: int, iter_num: int, batch_helper, non_eval_checkpoint=False):
        checkpoint = {
            MODEL_CONFIG: {
                "author": self.author,
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
                "block_size": self.block_size,
                "dropout": self.dropout,
                "attn_dropout": self.attn_dropout,
            },
            OPT_CONFIG: {
                "weight_decay": self.opt_weight_decay,
                "learning_rate": self.opt_learning_rate,
                "betas": self.opt_betas,
                "optimizer": self.opt_type,
                "backoff_thresh": self.backoff_thresh,
                "backoff_rates": self.backoff_rates,
            },
            MODEL_STATE_DICT: self.state_dict(),
            OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            ITER_NUM: iter_num,
        }

        copy_iter_num = iter_num
        trailing_zeros = 1
        while copy_iter_num >= 10:
            trailing_zeros += 1
            copy_iter_num = int(float(copy_iter_num) / 10.0)

        zeros_to_add = MAX_ZEROS_IN_CKPTFN - trailing_zeros
        prefix = ""
        while zeros_to_add > 0:
            prefix += "0"
            zeros_to_add -= 1

        save_file_name = (
            f"ITER{prefix}{iter_num}_{val_loss:08.4f}val_loss"
            + "_"
            + nanoGPT.__name__
            + "_"
            + self.author.replace(" ", "")
            + CHECKPOINT_EXT
        )

        if non_eval_checkpoint:
            save_file_name = (
                f"ITER{prefix}{iter_num}_CHKPT{val_loss:08.4f}val_loss"
                + "_"
                + nanoGPT.__name__
                + "_"
                + self.author.replace(" ", "")
                + CHECKPOINT_EXT
            )

        full_checkpoint_path = os.path.join(self.getCheckpointPath(), save_file_name)
        os.makedirs(os.path.dirname(full_checkpoint_path), exist_ok=True)
        torch.save(checkpoint, full_checkpoint_path)
        if batch_helper is not None:
            batch_helper.save_checkpoint_sidecar(
                full_checkpoint_path,
                wait_for_builder=False,
            )

    def configure_optimizers(self, args):
        print(args)
        weight_decay, learning_rate, betas, optimizer_type, bkoff_thresh, bkoff_rates, device_type = args

        self.opt_weight_decay = weight_decay
        self.opt_learning_rate = learning_rate
        self.opt_betas = betas
        self.opt_type = optimizer_type
        self.backoff_thresh = bkoff_thresh
        self.backoff_rates = bkoff_rates

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        param_groups = None

        if optimizer_type == "muon":
            print("Using muon optimizer")

            hidden_weights = []
            hidden_gains_biases = []
            nonhidden_decay = []
            nonhidden_no_decay = []

            nonhidden_keywords = (
                "token_emb",
                "tok_emb",
                "wte",
                "wpe",
                "pos_emb",
                "embed",
                "embedding",
                "head",
                "lm_head",
                "output",
                "classifier",
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
            num_nonhidden_decay = sum(p.numel() for p in nonhidden_decay)
            num_nonhidden_no_decay = sum(p.numel() for p in nonhidden_no_decay)

            print(f"num muon parameter tensors: {len(hidden_weights)}, with {num_hidden_weights:,} parameters")
            print(f"num hidden gain/bias tensors: {len(hidden_gains_biases)}, with {num_hidden_gains_biases:,} parameters")
            print(f"num nonhidden decay tensors: {len(nonhidden_decay)}, with {num_nonhidden_decay:,} parameters")
            print(f"num nonhidden no-decay tensors: {len(nonhidden_no_decay)}, with {num_nonhidden_no_decay:,} parameters")

            hidden_lr, nonhidden_lr = learning_rate[0], learning_rate[1]

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
                    weight_decay=0.0,
                ),
            ]

        if optimizer_type == "adam":
            decay_params = []
            nodecay_params = []
            seen = set()

            for name, p in param_dict.items():
                if id(p) in seen:
                    continue
                seen.add(id(p))

                lname = name.lower()
                is_norm = "norm" in lname
                is_bias_or_gain = p.ndim < 2
                is_embedding_or_head = any(
                    k in lname
                    for k in (
                        "token_emb",
                        "tok_emb",
                        "wte",
                        "wpe",
                        "pos_emb",
                        "embed",
                        "embedding",
                        "head",
                        "lm_head",
                        "output",
                        "classifier",
                    )
                )

                if is_bias_or_gain or is_norm or is_embedding_or_head:
                    nodecay_params.append(p)
                else:
                    decay_params.append(p)

            param_groups = [
                {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate, "betas": betas},
                {"params": nodecay_params, "weight_decay": 0.0, "lr": learning_rate, "betas": betas},
            ]

            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        if param_groups is None:
            raise ValueError(f"Invalid optimizer type: {optimizer_type}")

        return [param_groups, optimizer_type, device_type]


def getArgs(checkpoint, model_folder_name="N/A", chkpt_folder_name="N/A", resume=False):
    """
    Checkpoint-compatible argument reconstruction.

    Old checkpoints may not have dropout/attn_dropout.
    New checkpoints save both. This function handles both cases safely.
    """
    cfg = checkpoint[MODEL_CONFIG]

    model_args = [
        model_folder_name,
        chkpt_folder_name,
        cfg.get("author", "N/A"),
        cfg["vocab_size"],
        cfg["n_embd"],
        cfg["n_head"],
        cfg["n_layer"],
        cfg["block_size"],
        resume,
        cfg.get("dropout", 0.0),
        cfg.get("attn_dropout", 0.0),
    ]

    opt_args = []
    opt_saved_config = checkpoint[OPT_CONFIG]
    for key in opt_saved_config:
        opt_args.append(opt_saved_config[key])

    return model_args, opt_args


def loadFromCheckpoint(model_folder_name: str, checkpoint_file_path: str, BatchHelper=None) -> tuple[nn.Module, Any, Any, Any, int, list|None]:
    split_path = checkpoint_file_path.split("/")

    if len(split_path) != 2:
        raise ValueError("Checkpoint path should only be folder_name/checkpoint_to_load.ext")

    chkpt_folder_name, ckpt_file_name = split_path
    load_path = os.path.join(os.getcwd(), model_folder_name, chkpt_folder_name, ckpt_file_name)

    checkpoint = torch.load(load_path, weights_only=True)
    model_args, opt_args = getArgs(checkpoint, model_folder_name, chkpt_folder_name, resume=True)

    gpt_model = nanoGPT(*model_args)
    gpt_model.checkpoint_folder_name = chkpt_folder_name

    model_sd = checkpoint[MODEL_STATE_DICT]
    opt_sd = checkpoint[OPTIMIZER_STATE_DICT]
    iter_num = checkpoint[ITER_NUM]

    checkpoint = None

    dataloader_items = BatchHelper.load_checkpoint_sidecar_items(load_path)
    
    other_args = [dataloader_items]
    return gpt_model, model_sd, opt_args, opt_sd, iter_num, other_args


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_args, _ = getArgs(checkpoint, resume=True)

    model = nanoGPT(*model_args)
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])

    checkpoint = None

    print(f"#Params: {model.num_parameters}")

    model.to(device)
    model.eval()

    return model