import torch
import torch.nn as nn
import os
import inspect
from torch.nn import functional as F
from typing import Any
import math
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


MODEL_CONFIG = "model_config"
OPT_CONFIG = "optimizer_config"
MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
ITER_NUM = "iter_num"
CHECKPOINT_DEFAULT = "checkpoints"
CHECKPOINT_EXT = ".pt"
MAX_ZEROS_IN_CKPTFN = 8

# class nanoGPT(nn.Module):
#     def __init__(self, model_folder_name:str, chkpt_folder_name:str=None, 
#                  author:str="N/A", # Metadata args
#                  vocab_size=50257, n_embd=128, n_head=4, n_layer=2, block_size=1024):

#         super().__init__()
#         self.block_size = block_size
#         self.token_emb = nn.Embedding(vocab_size, n_embd)
#         self.pos_emb = nn.Embedding(block_size, n_embd)
#         self.blocks = nn.ModuleList([
#             # nn.TransformerEncoderLayer(
#             #     d_model=n_embd, nhead=n_head,
#             #     dim_feedforward=4 * n_embd, dropout=0.1,
#             #     activation="gelu", batch_first=True,
#             # )
#             nn.TransformerEncoderLayer(
#                 norm_first=True,
#                 d_model=n_embd, nhead=n_head,
#                 dim_feedforward=4 * n_embd, dropout=0.0,
#                 activation="gelu", batch_first=True,
#             )
#             for _ in range(n_layer)
#         ])
#         self.ln_f = nn.LayerNorm(n_embd)
#         self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
#         self.lm_head.weight = self.token_emb.weight # set the output and input token embeddings to be the same as it can save parameters without losing much accuracy

#         self.vocab_size = vocab_size
#         self.n_layer = n_layer
#         self.n_head = n_head
#         self.n_embd = n_embd
#         self.block_size = block_size

#         self.num_parameters = sum(val.numel() for val in self.parameters())
#         self.author = author
#         self.model_path = os.path.join(os.getcwd(), model_folder_name)
#         if (chkpt_folder_name is None):
#             self.checkpoint_folder_name = CHECKPOINT_DEFAULT
#         else:
#             self.checkpoint_folder_name = chkpt_folder_name

#         self.opt_weight_decay = 0
#         self.opt_learning_rate = 0
#         self.opt_betas = 0
#         self.opt_device_type = 0
#         self.opt_type = "muon"

#     def forward(self, input_ids, targets=None):
#         """
#         Args:
#             input_ids: LongTensor of shape (batch_size, seq_len)
#         Returns:
#             logits: FloatTensor of shape (batch_size, seq_len, 50257)
#         """
#         B, T = input_ids.shape
#         tok_emb = self.token_emb(input_ids)
#         pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
#         x = tok_emb + pos_emb

#         # Causal mask
#         mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()

#         for block in self.blocks:
#             x = block(x, src_mask=mask, is_causal=True)

#         x = self.ln_f(x)
#         logits = self.lm_head(x)

#         if targets is not None:
#             # if we are given some desired targets also calculate the loss
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
#         # else:
#         #     # inference-time mini-optimization: only forward the lm_head on the very last position
#         #     logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
#         #     loss = None

#         # Evaluation return path, only expects logits
#         if (targets is None):
#             return logits
#         # Training return path, wants logits and loss with respect to a target for simplicitly
#         return logits, loss

#     def estimate_mfu(self, fwdbwd_per_iter, dt):
#         """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
#         # first estimate the number of flops we do per iteration.
#         # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
#         N = self.num_parameters
#         L, H, Q, T = self.n_layer, self.n_head, self.n_embd//self.n_head, self.block_size
#         flops_per_token = 6*N + 12*L*H*Q*T
#         flops_per_fwdbwd = flops_per_token * T
#         flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
#         # express our flops throughput as ratio of A100 bfloat16 peak flops
#         flops_achieved = flops_per_iter * (1.0/dt) # per second
#         flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
#         mfu = flops_achieved / flops_promised
#         # mfu = flops_achieved
#         return mfu

#     def getCheckpointPath(self):
#         full_path = os.path.join(self.model_path, self.checkpoint_folder_name)
#         return full_path

#     # file_name -> no extension, the needed one is added in the func. 
#     # def saveCheckpoint(self, file_name:str, epoch:int, optimizer:nn.Module, loss:int):
#     def saveCheckpoint(self, optimizer:nn.Module, val_loss:int, iter_num:int, non_eval_checkpoint=False):
#         checkpoint = {
#             MODEL_CONFIG: {
#                 "author": self.author,
#                 "vocab_size": self.vocab_size,
#                 "n_embd": self.n_embd,
#                 "n_head": self.n_head,
#                 "n_layer": self.n_layer,
#                 "block_size": self.block_size
#             },
#             OPT_CONFIG:{
#                 "weight_decay":self.opt_weight_decay,
#                 "learning_rate":self.opt_learning_rate,
#                 "betas":self.opt_betas,
#                 "optimizer":self.opt_type,
#                 "device_type":self.opt_device_type,
#             },
#             MODEL_STATE_DICT: self.state_dict(),
#             OPTIMIZER_STATE_DICT: optimizer.state_dict(),
#             ITER_NUM: iter_num,
#         }
#         # Helps with checkpoint coherence on replacement. When a threshold is crossed, eg. 9->10 or 99->100 the checkpoints
#         # will continue to stay sorted and won't have differing filename lengths
#         copy_iter_num = iter_num
#         trailing_zeros = 1
#         while (copy_iter_num >= 10):
#             trailing_zeros += 1
#             copy_iter_num = int(float(copy_iter_num)/10.0)
#         zeros_to_add = MAX_ZEROS_IN_CKPTFN-trailing_zeros
#         prefix = ""
#         while(zeros_to_add > 0):
#             prefix += "0"
#             zeros_to_add -= 1

#         save_file_name = f"ITER{prefix}{iter_num}_{val_loss:08.4f}val_loss" + "_" + nanoGPT.__name__ + "_" + self.author.replace(" ", "") + CHECKPOINT_EXT
#         if (non_eval_checkpoint):
#             save_file_name = f"ITER{prefix}{iter_num}_CHKPT{val_loss:08.4f}val_loss" + "_" + nanoGPT.__name__ + "_" + self.author.replace(" ", "") + CHECKPOINT_EXT

#         full_checkpoint_path = os.path.join(self.getCheckpointPath(), save_file_name)
#         torch.save(checkpoint, full_checkpoint_path)

#     def configure_optimizers(self, weight_decay, learning_rate, betas, optimizer_type, device_type):
#         self.opt_weight_decay = weight_decay
#         self.opt_learning_rate = learning_rate
#         self.opt_betas = betas
#         self.opt_type = optimizer_type
#         self.opt_device_type = device_type

#         # gather trainable params once
#         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

#         if optimizer_type == "muon":
#             print("Using muon optimizer")

#             hidden_weights = []
#             hidden_gains_biases = []
#             nonhidden_decay = []
#             nonhidden_no_decay = []

#             # params that should stay on Adam path, not Muon
#             nonhidden_keywords = (
#                 "embed", "embedding", "wte", "wpe",
#                 "head", "lm_head", "output", "classifier",
#             )

#             seen = set()

#             for name, p in param_dict.items():
#                 if id(p) in seen:
#                     continue
#                 seen.add(id(p))

#                 lname = name.lower()

#                 # embeddings / output heads stay on Adam path
#                 if any(k in lname for k in nonhidden_keywords):
#                     if p.ndim >= 2:
#                         nonhidden_decay.append(p)      # embeddings / output matrices
#                     else:
#                         nonhidden_no_decay.append(p)   # biases / norm scales
#                 # hidden matrix weights -> Muon
#                 elif p.ndim >= 2:
#                     hidden_weights.append(p)
#                 # hidden biases / norm gains -> Adam path
#                 else:
#                     hidden_gains_biases.append(p)

#             num_hidden_weights = sum(p.numel() for p in hidden_weights)
#             num_hidden_gains_biases = sum(p.numel() for p in hidden_gains_biases)
#             num_nonhidden_params = sum(p.numel() for p in nonhidden_decay)

#             print(f"num muon parameter tensors: {len(hidden_weights)}, with {num_hidden_weights:,} parameters")
#             print(f"num hidden gain/bias tensors: {len(hidden_gains_biases)}, with {num_hidden_gains_biases:,} parameters")
#             print(f"num nonhidden parameter tensors: {len(nonhidden_decay)}, with {num_nonhidden_params:,} parameters")

#             hidden_lr,nonhidden_lr = learning_rate[0], learning_rate[1]
#             param_groups = [
#                 dict(
#                     params=hidden_weights,
#                     use_muon=True,
#                     lr=hidden_lr,
#                     weight_decay=weight_decay,
#                 ),
#                 dict(
#                     params=hidden_gains_biases + nonhidden_no_decay,
#                     use_muon=False,
#                     lr=nonhidden_lr,
#                     betas=betas,
#                     weight_decay=0.0,
#                 ),
#                 dict(
#                     params=nonhidden_decay,
#                     use_muon=False,
#                     lr=nonhidden_lr,
#                     betas=betas,
#                     weight_decay=weight_decay,  # or smaller, e.g. 0.01
#                 ),
#             ]

#             use_distributed_muon = (
#                 torch.distributed.is_available()
#                 and torch.distributed.is_initialized()
#                 and torch.distributed.get_world_size() > 1
#             )

#             if use_distributed_muon:
#                 print("Using distributed MuonWithAuxAdam")
#                 optimizer = MuonWithAuxAdam(param_groups)
#             else:
#                 print("Using SingleDeviceMuonWithAuxAdam")
#                 optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

#             return optimizer

#         if optimizer_type == "adam":
#             # default AdamW path
#             decay_params = []
#             nodecay_params = []
#             seen = set()

#             for _, p in param_dict.items():
#                 if id(p) in seen:
#                     continue
#                 seen.add(id(p))

#                 if p.dim() >= 2:
#                     decay_params.append(p)
#                 else:
#                     nodecay_params.append(p)

#             optim_groups = [
#                 {"params": decay_params, "weight_decay": weight_decay},
#                 {"params": nodecay_params, "weight_decay": 0.0},
#             ]

#             num_decay_params = sum(p.numel() for p in decay_params)
#             num_nodecay_params = sum(p.numel() for p in nodecay_params)
#             print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
#             print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

#             fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
#             use_fused = fused_available and device_type == "cuda"
#             extra_args = dict(fused=True) if use_fused else dict()

#             optimizer = torch.optim.AdamW(
#                 optim_groups,
#                 lr=learning_rate,
#                 betas=betas,
#                 **extra_args,
#             )
#             print(f"using fused AdamW: {use_fused}")

#             return optimizer
        
#         raise ValueError(f"Invalid optimizer type: {optimizer_type}")

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
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                norm_first=True,
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # GPT-style init.
        self.apply(self._init_weights)

        # GPT-2-style residual projection scaling.
        #
        # In nn.TransformerEncoderLayer:
        # - self_attn.out_proj projects attention output back into residual stream
        # - linear2 projects MLP output back into residual stream
        #
        # These are scaled by 1 / sqrt(2 * n_layer).
        for block in self.blocks:
            nn.init.normal_(
                block.self_attn.out_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layer),
            )

            nn.init.normal_(
                block.linear2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layer),
            )

        # Tie token embedding and LM head weights after init.
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
        self.opt_device_type = 0
        self.opt_type = "muon"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
    checkpoint = None
    print(f"#Params: {model.num_parameters}")
    model.to(device)
    model.eval()
    return model
