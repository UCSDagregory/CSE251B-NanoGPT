"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import csv
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import train_helper_generic as train_helper
import train_helper_loader as th_loader
import argparse

import json

import json

TRAIN_HELPER_FILENAME = "train_helper.py"
OPT_FILENAME = "training_opt_params.json"
LEARNING_RATE = None
OPT_TYPE = "muon"

# Example to resume training from a checkpoint
# python train.py --device cuda --type resume --folder my_model --data_fd_name data/shakespeare_char --chpr checkpoints/077.5488val_loss_nanoGPT_DaginGregory.pt

# Example to train a model from scratch
# python train.py --device cuda --type scratch --folder my_model --data_fd_name data/shakespeare_char

def parseOptParams(file_path):
    global LEARNING_RATE
    global OPT_TYPE
    args = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for key in data:
            args.append(data[key])
        LEARNING_RATE = data['learning_rate']
        OPT_TYPE = data['optimizer']
    return args

parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True)
parser.add_argument("--type", required=True)
parser.add_argument("--folder", required=True)
parser.add_argument("--data_fd_name", required=True) # data_folder_name/folder_name
parser.add_argument("--chpn", required=False) # folder_name to save checkpoints, only for from scratch init
parser.add_argument("--chpr", required=False) # folder_name/checkpoint_file_name
parser.add_argument("--opt_name", required=False)
parser.add_argument("--data_collection", required=False)

args = parser.parse_args()

init_from:str = args.type

# Since each folder is self contained we must load the train_helper as a module to know how to create a model
model_folder_name = args.folder
model_path = os.path.join(os.getcwd(), model_folder_name)
train_helper_path = os.path.join(model_path, TRAIN_HELPER_FILENAME)

chkpt_folder_name_init = args.chpn
if (not args.chpn is None and init_from != 'scratch'):
    raise ValueError("Use --chpn file_name on only on from scratch creation to specify where to save checkpoints to.\n")

if (not args.chpr is None and init_from != 'resume'):
    raise ValueError("Use --chpr only on resume to tell the model where to save future checkpoints and which checkpoint to resume training from.\n")

impl_module = th_loader.loadModule(TRAIN_HELPER_FILENAME, train_helper_path, model_path)
train_helper.registerCreateModel(impl_module.createModel)

arg_opt_path = args.opt_name
if (arg_opt_path is None):
    arg_opt_path = OPT_FILENAME
opt_path = os.path.join(model_path, arg_opt_path)
parsed_opt_args = parseOptParams(opt_path)
if (LEARNING_RATE is None):
    raise ValueError("Something went wrong when parsing the optimizer.json, couldn't extract a valid learning rate.")

# eval_interval = 2000
eval_interval = 75 # How many iters until re-calculate val. loss
# eval_interval = 5
log_interval = 1
eval_iters = 25 # How many times to calculate val.loss per 'eval interval'
# eval_iters = 5
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
# iters_per_checkpoint = 5
iters_per_checkpoint = 15
# max_checkpoints_to_keep = 25
max_checkpoints_to_keep = 60

dataset = args.data_fd_name
effective_batch_size = 96
# Important note
# batch_size exists to be memory efficient, if there isn't enough space in the GPU memory it will spill over into SMEM (shared memory)
# which is significantly slower than it just living in VRAM
# batch_size should be as large as possible, (must be >= 1) s.t. everything fits in VRAM without spilling to SMEM, which will kill performance.
# sometimes the model is too large to all fit into VRAM which is ok, that's just the limit of the hardware

# Effective_batch_size is the compensation, in order to keep the gradients sufficiently clean and free of noise, we grab a total of effective_batch_size chunks 
# to train on from the data set, in batch_size chunks at a time. These then get sent to the GPU and the forward-backward starts.

# It used to ge gradient_accumulation_steps was separate and effective_batch_size was a calculated side effect of batch_size*gradient_accumulation_steps
# this is fine but it abstracts the reality that the number of effective batches per iteration is what matters for keeping the gradient clean.
# It's been changed s.t. the user now fixes effective_batch_size to a number they deem reasonable for clean gradients and batch_size to ensure training is as performant as possible
# gradient_accumulation_steps (micro batches) are now calculated from the two. The main issue is it's possible to construct non-evenly divisible batches so we simply round up

# batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
batch_size = 4 # if gradient_accumulation_steps > 1, this is the micro-batch size
gradient_accumulation_steps = int(float(effective_batch_size)/float(batch_size)) # used to simulate larger batch sizes
remainder = effective_batch_size%batch_size
if (remainder > 0):
    gradient_accumulation_steps += 1
print(f"Gradient accumulation steps:{gradient_accumulation_steps} | Total effective batch size:{gradient_accumulation_steps*batch_size}")
block_size = 1024 # Defined by project specs, DO NOT CHANGE

# max_iters = 600000 # total number of training iterations
max_iters = 10000 # total number of training iterations
# max_iters = 1500 # total number of training iterations
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
# warmup_iters = 200 # how many steps to warm up for
# warmup_iters = 25 # how many steps to warm up for
warmup_iters = max(25,int(0.05*float(max_iters))) # how many steps to warm up for
lr_decay_iters = int(max_iters*1.0) # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = args.device
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# compile = True # use PyTorch 2.0 to compile the model to be faster

print(f"Maximum unique training tokens{max_iters*effective_batch_size*block_size:.2e}\n\n")

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join(os.getcwd(), dataset)
batch_helper = th_loader.BatchHelper(block_size, batch_size, data_dir, device, device_type)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model = train_helper.createModel(model_folder_name, None, chkpt_folder_name_init)
    opt_args = parsed_opt_args
    opt_args.append(device_type)
    optimizer = model.configure_optimizers(*opt_args)
    model.to(device)

elif init_from == 'resume':
    checkpoint_file_path = args.chpr
    print(f"Resuming from a checkpoint:{checkpoint_file_path}")
    model, model_sd, opt_args, opt_sd, iter_num = train_helper.createModel(model_folder_name, checkpoint_file_path, None, from_scratch=False)
    max_iters += iter_num
    model.load_state_dict(model_sd)
    model.to(device)
    optimizer = model.configure_optimizers(*opt_args)
    optimizer.load_state_dict(opt_sd)

else:
    raise ValueError("Unknown input for --type")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            print(f"Loss({k}) estimate")
            X, Y = batch_helper.get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if OPT_TYPE == "adam":
        base_lr = LEARNING_RATE
    else:
        hidden_base_lr, nonhidden_base_lr = LEARNING_RATE

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        warmup_scale = (it + 1) / (warmup_iters + 1)

        if OPT_TYPE == "adam":
            return base_lr * warmup_scale

        return [
            hidden_base_lr * warmup_scale,
            nonhidden_base_lr * warmup_scale,
        ]

    # 2) if past decay horizon, clamp at min lr
    if it > lr_decay_iters:
        if OPT_TYPE == "adam":
            return min_lr

        return [min_lr, min_lr]

    # 3) cosine decay from base lr down to min lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    if OPT_TYPE == "adam":
        return min_lr + coeff * (base_lr - min_lr)

    hidden_lr = min_lr + coeff * (hidden_base_lr - min_lr)
    nonhidden_lr = min_lr + coeff * (nonhidden_base_lr - min_lr)
    return [hidden_lr, nonhidden_lr]

# training loop
X, Y = batch_helper.get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = iter_num # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

model_checkpoint_path = model.getCheckpointPath()

lossf = 0.0
tokens_parsed = 0.0
losses = {}
losses['val'] = 999.0
print(f"Warmup iters for:{warmup_iters}\n")
while True:
    total_tokens_processed = iter_num*effective_batch_size*block_size
    print(f"Tokens processed:{total_tokens_processed} | {total_tokens_processed:.2e}\n")

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else LEARNING_RATE
    # lr = get_lr(iter_num)

    if OPT_TYPE == "adam":
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    elif OPT_TYPE == "muon":
        hidden_lr, nonhidden_lr = lr
        optimizer.param_groups[0]["lr"] = hidden_lr
        optimizer.param_groups[1]["lr"] = nonhidden_lr
    else:
        raise ValueError("Something is incorrect with learning rate or optimizer type. Adam is a single scalar: ..., Muon is a list: [...]")

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num%eval_interval == 0 and master_process:
        losses = estimate_loss()
        train_ppl = math.exp(losses['train'])
        val_ppl = math.exp(losses['val'])
        tokens_seen = iter_num * tokens_per_iter
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"train ppl {train_ppl:.1f}, val ppl {val_ppl:.1f}")

    if iter_num%iters_per_checkpoint == 0 or (iter_num%eval_interval == 0 and losses['val'] < best_val_loss):
        best_val_loss = losses['val']
        if iter_num >= 0:
            print(f"Current val loss:{losses['val']:.4f}")
            current_checkpoints = len(os.listdir(model_checkpoint_path))
            if (current_checkpoints >= max_checkpoints_to_keep):
                files = os.listdir(model_checkpoint_path) # pseudo sorted because the val. loss is expected to decrease with time
                # chkpt_to_remove = os.path.join(model_checkpoint_path, files[len(files)-1])
                chkpt_to_remove = os.path.join(model_checkpoint_path, files[0]) # we remove the first since we store the iter num in the file name now since it's a better indicator of performance
                print(f"Removing checkpoint: {chkpt_to_remove}")
                os.remove(chkpt_to_remove)
            # model.saveCheckpoint(optimizer, losses['val'], iter_num)
            if (iter_num%eval_interval == 0):
                model.saveCheckpoint(optimizer, losses['val'], iter_num)
            else:
                model.saveCheckpoint(optimizer, lossf, iter_num)
            # if (not args.data_collection is None):
            # Just save to training data as a default
            with open(os.path.join(model_path, "training_data.txt"), mode='a') as f:
                f.write(f"{total_tokens_processed},{iter_num},{lossf:.4f}\n")
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = batch_helper.get_batch('train')
        # backward pass, with gradient scaling if training in fp16
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        tokens_seen = iter_num * tokens_per_iter

        # Format LR for logging (scalar for adam, first element for muon)
        lr_for_log = lr if OPT_TYPE == "adam" else lr[0]

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        # Log train metrics
        train_log_writer.writerow([
            iter_num,
            f"{lossf:.4f}",
            f"{lr_for_log:.6e}",
            f"{running_mfu:.4f}",
            f"{dt*1000:.2f}",
            tokens_seen,
        ])
        train_log_file.flush()

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# ---------------------------------------------------------------------------
# Close log files
# ---------------------------------------------------------------------------
train_log_file.close()
eval_log_file.close()

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
training_end_time = time.time()
total_time = training_end_time - training_start_time
total_tokens = iter_num * tokens_per_iter

# Format final LR for display
lr_display = f"{lr:.6e}" if OPT_TYPE == "adam" else f"[{lr[0]:.6e}, {lr[1]:.6e}]"

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"  Model parameters:      {n_params:,}")
print(f"  Total iterations:      {iter_num:,}")
print(f"  Total tokens seen:     {total_tokens:,}")
print(f"  Training time:         {total_time/3600:.2f} hours")
print(f"  Tokens/sec:            {total_tokens/total_time:,.0f}")
print(f"  Best val loss:         {best_val_loss:.4f}")
print(f"  Best val PPL:          {math.exp(best_val_loss):.2f}")
print(f"  Final learning rate:   {lr_display}")
print(f"  Optimizer:             {OPT_TYPE}")
print(f"  Batch size (micro):    {batch_size}")
print(f"  Grad accum steps:      {gradient_accumulation_steps}")
print(f"  Tokens per iter:       {tokens_per_iter:,}")
print(f"  Device:                {device}")
print(f"  Dtype:                 {dtype}")
print("="*60)

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for headless servers
    import matplotlib.pyplot as plt

    # --- Read eval CSV ---
    eval_iters_list, eval_train_loss, eval_val_loss = [], [], []
    eval_train_ppl, eval_val_ppl = [], []
    with open(eval_log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eval_iters_list.append(int(row["iter"]))
            eval_train_loss.append(float(row["train_loss"]))
            eval_val_loss.append(float(row["val_loss"]))
            eval_train_ppl.append(float(row["train_ppl"]))
            eval_val_ppl.append(float(row["val_ppl"]))

    # --- Read train CSV ---
    train_iters_list, train_loss_list, lr_list, mfu_list = [], [], [], []
    with open(train_log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_iters_list.append(int(row["iter"]))
            train_loss_list.append(float(row["train_loss"]))
            lr_list.append(float(row["lr"]))
            mfu_list.append(float(row["mfu"]))

    # --- Figure 1: Train & Val Loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_iters_list, train_loss_list, alpha=0.3, label="Train loss (per iter)", color="blue")
    ax.plot(eval_iters_list, eval_train_loss, 'o-', label="Train loss (eval)", color="blue", markersize=3)
    ax.plot(eval_iters_list, eval_val_loss, 'o-', label="Val loss (eval)", color="red", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "loss_curve.png"), dpi=150)
    print(f"Saved: {os.path.join(log_dir, 'loss_curve.png')}")

    # --- Figure 2: Val PPL ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eval_iters_list, eval_val_ppl, 'o-', label="Val PPL", color="red", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "val_ppl_curve.png"), dpi=150)
    print(f"Saved: {os.path.join(log_dir, 'val_ppl_curve.png')}")

    # --- Figure 3: Learning Rate Schedule ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_iters_list, lr_list, color="green")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Warmup + Cosine Decay)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "lr_schedule.png"), dpi=150)
    print(f"Saved: {os.path.join(log_dir, 'lr_schedule.png')}")

    # --- Figure 4: MFU ---
    fig, ax = plt.subplots(figsize=(10, 4))
    valid = [(i, m) for i, m in zip(train_iters_list, mfu_list) if m > 0]
    if valid:
        ax.plot([x[0] for x in valid], [x[1]*100 for x in valid], color="purple")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MFU (%)")
    ax.set_title("Model FLOPs Utilization")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "mfu_curve.png"), dpi=150)
    print(f"Saved: {os.path.join(log_dir, 'mfu_curve.png')}")

    plt.close('all')
    print("\nAll plots saved successfully.")

except ImportError:
    print("\nmatplotlib not installed -- skipping plot generation.")
    print("Install with: pip install matplotlib")
except Exception as e:
    print(f"\nPlot generation failed: {e}")
    print("CSV logs are still available for manual plotting.")

if ddp:
    destroy_process_group()