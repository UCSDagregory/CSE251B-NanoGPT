import torch
import torch.nn as nn
import os
import inspect
from torch.nn import functional as F
from typing import Any

MODEL_CONFIG = "model_config"
OPT_CONFIG = "optimizer_config"
MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
CHECKPOINT_EXT = ".pt"
class nanoGPT(nn.Module):
    def __init__(self, model_folder_name:str, author:str="N/A", 
                 vocab_size=50257, n_embd=128, n_head=4, n_layer=2, block_size=1024):
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
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head,
                dim_feedforward=4 * n_embd, dropout=0.1,
                activation="gelu", batch_first=True,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight # set the output and input token embeddings to be the same as it can potentially save a lot of parameters without losing much accuracy

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

        self.num_parameters = sum(val.numel() for val in self.parameters())
        self.author = author
        self.checkpoint_path = os.path.join(os.getcwd(), model_folder_name, "checkpoints")

        self.opt_weight_decay = 0
        self.opt_learning_rate = 0
        self.opt_betas = 0
        self.opt_device_type = 0

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

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # return logits
        return logits, loss

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

    # file_name -> no extension, the needed one is added in the func. 
    # def saveCheckpoint(self, file_name:str, epoch:int, optimizer:nn.Module, loss:int):
    def saveCheckpoint(self, optimizer:nn.Module, val_loss:int):
        checkpoint = {
            MODEL_CONFIG: {
                "author": self.author,
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
                "block_size": self.block_size
            },
            OPT_CONFIG:{
                "weight_decay":self.opt_weight_decay,
                "learning_rate":self.opt_learning_rate,
                "betas":self.opt_betas,
                "device_type":self.opt_device_type,
            },
            MODEL_STATE_DICT: self.state_dict(),
            OPTIMIZER_STATE_DICT: optimizer.state_dict(),
        }
        save_file_name = f"{val_loss:08.4f}val_loss" + "_" + nanoGPT.__name__ + "_" + self.author.replace(" ", "") + CHECKPOINT_EXT
        full_checkpoint_path = os.path.join(self.checkpoint_path, save_file_name)
        torch.save(checkpoint, full_checkpoint_path)
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        self.opt_weight_decay = weight_decay
        self.opt_learning_rate = learning_rate
        self.opt_betas = betas
        self.opt_device_type = device_type

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

# file_name -> no extension, the needed one is added in the func. 
def loadFromCheckpoint(model_folder_name:str, checkpoint_file_path:str) -> tuple[nn.Module, Any, Any, Any]:
    load_path = os.path.join(os.getcwd(), model_folder_name, checkpoint_file_path+CHECKPOINT_EXT)
    checkpoint = torch.load(load_path, weights_only=True)
    model_args = [model_folder_name]
    model_saved_config = checkpoint[MODEL_CONFIG]
    for key in model_saved_config:
        model_args.append(model_saved_config[key])
    opt_args = []
    opt_saved_config = checkpoint[OPT_CONFIG]
    for key in opt_saved_config:
        opt_args.append(opt_saved_config[key])

    gpt_model = nanoGPT(*model_args)
    
    # optimizer = gpt_model.configure_optimizers(*opt_args)
    # gpt_model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    # optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT])
    # return gpt_model, optimizer
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

    # If you save config alongside weights, load it:
    # config = checkpoint["config"]
    # model = TinyGPT(**config)
    # model.load_state_dict(checkpoint["model_state_dict"])

    # Simple case: checkpoint is just the state_dict
    model = nanoGPT()  # Use your config here
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    print(f"#Params: {model.num_parameters}")


    # model.to(device)
    # model.eval()
    # return model


# --- Optional: quick sanity check ---

# if __name__ == "__main__":
#     print("Creating example model...")
#     model = nanoGPT()
#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"Parameters: {n_params:,}")

#     # Test forward pass
#     dummy_input = torch.randint(0, 50257, (2, 1024))
#     logits = model(dummy_input)
#     print(f"Input shape:  {dummy_input.shape}")
#     print(f"Output shape: {logits.shape}")
#     assert logits.shape == (2, 1024, 50257), "Output shape mismatch!"
#     print("Interface check passed.")

#     # Save example checkpoint
#     torch.save(model.state_dict(), "checkpoint.pt")
#     print("Saved example checkpoint.pt")
