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
CHECKPOINT_DEFAULT = "checkpoints"
CHECKPOINT_EXT = ".pt"

# --- Mixture of Experts (MoE) Components ---
class Expert(nn.Module):
    """A standard feed-forward network (MLP) acting as a single expert."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    """Routes tokens to the Top-K experts."""
    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        # 1. Get routing probabilities
        router_logits = self.router(x) 
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 2. Select Top-K experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True) # Normalize
        
        # 3. Route tokens
        flat_x = x.view(-1, C)
        flat_indices = top_k_indices.view(-1, self.top_k)
        flat_weights = top_k_weights.view(-1, self.top_k)
        flat_output = torch.zeros_like(flat_x)
        
        for i, expert in enumerate(self.experts):
            expert_mask = (flat_indices == i)
            if expert_mask.any():
                token_indices = expert_mask.any(dim=-1)
                expert_inputs = flat_x[token_indices]
                expert_outputs = expert(expert_inputs)
                
                # Apply weights
                weights = flat_weights[token_indices][expert_mask[token_indices]].unsqueeze(-1)
                flat_output[token_indices] += expert_outputs * weights
                
        return flat_output.view(B, T, C)

class CausalSelfAttention(nn.Module):
    """Standard Multi-Head Attention using efficient Flash Attention."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # F.scaled_dot_product_attention natively handles the causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MoEBlock(nn.Module):
    """Replaces nn.TransformerEncoderLayer. Combines Attention and MoE."""
    def __init__(self, n_embd, n_head, num_experts, top_k):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.moe = SparseMoE(n_embd, num_experts, top_k)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x
    
# --- Main Model ---
class nanoGPT(nn.Module):
    def __init__(self, model_folder_name:str, chkpt_folder_name:str=None, 
                 author:str="N/A", 
                 vocab_size=50257, n_embd=128, n_head=4, n_layer=2, block_size=1024,
                 num_experts=8, top_k=2): # Added MoE params
        # Model : Tokens -> Transformer Block(TB) -> TB -> ... -> TB -> Vocab Projection -> Logits
        # TB    : Input -> Multi headed attention(MHA) -> Residual_Add -> Normalization -> MLP(2 linear layers) -> Residual_Add -> Normalization -> hidden rep.
        # AF    : Non-linear function applied to a given input that outputs the same shape, IN(R,C): AF(IN) -> OUT(R,C)
        # MLP   : Linear layer(LL_0) -> AF -> LL_1
        # LL_0  : (d_feedforward X d_model) matrix of weights
        # LL_1  : (d_model X d_feedforward) matrix of weights
        
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # custom MoE blocks
        self.blocks = nn.ModuleList([
            MoEBlock(n_embd, n_head, num_experts, top_k) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight # set the output and input token embeddings to be the same as it can save parameters without losing much accuracy

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

        # MoE params
        self.num_experts = num_experts
        self.top_k = top_k

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
        self.opt_type = "adam"

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, 50257)
        """
        B, T = input_ids.shape
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb

        # mask removed, handled in CausalSelfAttention
        # mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
            
        return logits

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.num_parameters
        L, H, Q, T = self.n_layer, self.n_head, self.n_embd//self.n_head, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        # mfu = flops_achieved
        return mfu

    def getCheckpointPath(self):
        full_path = os.path.join(self.model_path, self.checkpoint_folder_name)
        return full_path

    # file_name -> no extension, the needed one is added in the func. 
    # def saveCheckpoint(self, file_name:str, epoch:int, optimizer:nn.Module, loss:int):
    def saveCheckpoint(self, optimizer:nn.Module, val_loss:int, non_eval_checkpoint=False):
        checkpoint = {
            MODEL_CONFIG: {
                "author": self.author,
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
                "block_size": self.block_size,
                "num_experts": self.num_experts, # added
                "top_k": self.top_k              # added
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
        }
        save_file_name = f"{val_loss:08.4f}val_loss" + "_" + nanoGPT.__name__ + "_" + self.author.replace(" ", "") + CHECKPOINT_EXT
        if (non_eval_checkpoint):
            save_file_name = f"CHKPT{val_loss:08.4f}val_loss" + "_" + nanoGPT.__name__ + "_" + self.author.replace(" ", "") + CHECKPOINT_EXT

        full_checkpoint_path = os.path.join(self.getCheckpointPath(), save_file_name)
        torch.save(checkpoint, full_checkpoint_path)

    def configure_optimizers(self, weight_decay, learning_rate, betas, optimizer_type, device_type):
        self.opt_weight_decay = weight_decay
        self.opt_learning_rate = learning_rate
        self.opt_betas = betas
        self.opt_type = optimizer_type
        self.opt_device_type = device_type

        # gather trainable params once
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if optimizer_type == "muon":
            print("Using muon optimizer")

            hidden_weights = []
            hidden_gains_biases = []
            nonhidden_params = []

            # params that should stay on Adam path, not Muon
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

                # embeddings / output heads stay on Adam path
                if any(k in lname for k in nonhidden_keywords):
                    nonhidden_params.append(p)
                # hidden matrix weights -> Muon
                elif p.ndim >= 2:
                    hidden_weights.append(p)
                # hidden biases / norm gains -> Adam path
                else:
                    hidden_gains_biases.append(p)

            num_hidden_weights = sum(p.numel() for p in hidden_weights)
            num_hidden_gains_biases = sum(p.numel() for p in hidden_gains_biases)
            num_nonhidden_params = sum(p.numel() for p in nonhidden_params)

            print(f"num muon parameter tensors: {len(hidden_weights)}, with {num_hidden_weights:,} parameters")
            print(f"num hidden gain/bias tensors: {len(hidden_gains_biases)}, with {num_hidden_gains_biases:,} parameters")
            print(f"num nonhidden parameter tensors: {len(nonhidden_params)}, with {num_nonhidden_params:,} parameters")

            hidden_lr,nonhidden_lr = learning_rate[0], learning_rate[1]
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

        if optimizer_type == "adam":
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

def loadFromCheckpoint(model_folder_name:str, checkpoint_file_path:str) -> tuple[nn.Module, Any, Any, Any]:
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
    checkpoint = None
    return gpt_model, model_sd, opt_args, opt_sd

# --- Required: load_model function ---
def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load your trained model from a checkpoint.

    This function is called by evaluate.py. It must return a model where:
        model(input_ids) -> logits
        - input_ids: LongTensor of shape (batch, seq_len)
        - logits: FloatTensor of shape (batch, seq_len, 50257)

    Args:
        checkpoint_path: Path to your checkpoint.pt file
        device: Device to load onto ("cuda" or "cpu")

    Returns:
        model: nn.Module in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_args, opt_args = getArgs(checkpoint)
    model = nanoGPT(*model_args)
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    print(f"#Params: {model.num_parameters}")
    model.to(device)
    model.eval()
    return model
