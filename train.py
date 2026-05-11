"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
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
import glob

TRAIN_HELPER_FILENAME = "train_helper.py"
OPT_FILENAME = "training_opt_params.json"
LEARNING_RATE = None
OPT_TYPE = "adam"

# ---------------- ARGPARSE ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True)
parser.add_argument("--type", required=True)
parser.add_argument("--folder", required=True)
parser.add_argument("--data_fd_name", required=True)
parser.add_argument("--chpn", required=False)
parser.add_argument("--chpr", required=False)
parser.add_argument("--opt_name", required=False)
args = parser.parse_args()

init_from = args.type
model_folder_name = args.folder
model_path = os.path.join(os.getcwd(), model_folder_name)

# ---------------- DEBUG GLOBAL INIT ----------------
print("\n" + "="*80)
print("🔍 TRAIN INITIAL DEBUG")
print("="*80)
print(f"cwd: {os.getcwd()}")
print(f"model_path: {model_path}")
print(f"dataset: {args.data_fd_name}")
print(f"device: {args.device}")
print("="*80)

# ---------------- LOAD MODEL HELPER ----------------
train_helper_path = os.path.join(model_path, TRAIN_HELPER_FILENAME)
impl_module = th_loader.loadModule(TRAIN_HELPER_FILENAME, train_helper_path, model_path)
train_helper.registerCreateModel(impl_module.createModel)

# ---------------- OPT ----------------
arg_opt_path = args.opt_name or OPT_FILENAME
opt_path = os.path.join(model_path, arg_opt_path)

parsed_opt_args = []
with open(opt_path, 'r') as file:
    data = json.load(file)
    for key in data:
        parsed_opt_args.append(data[key])
    LEARNING_RATE = data['learning_rate']
    OPT_TYPE = data['optimizer']

print("\n[OPT DEBUG]")
print(f"LR: {LEARNING_RATE}")
print(f"OPT_TYPE: {OPT_TYPE}")
print(f"parsed_opt_args: {parsed_opt_args}")

# ---------------- CONFIG ----------------
eval_interval = 100
log_interval = 10
eval_iters = 50
iters_per_checkpoint = 500
max_iters = 20400
warmup_iters = 500
lr_decay_iters = max_iters
min_lr = 6e-5
grad_clip = 1.0

dataset = args.data_fd_name
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 24

device = args.device
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# ---------------- DDP ----------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens/iter: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset)

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ---------------- DATA ----------------
data_dir = os.path.join(os.getcwd(), dataset)
train_files = sorted(glob.glob(os.path.join(data_dir, 'train*.bin')))

print("\n[DATA DEBUG]")
print(f"train files: {train_files}")
print(f"val exists: {os.path.exists(os.path.join(data_dir, 'val.bin'))}")

def get_batch(split):
    if split == 'train':
        data_file = train_files[torch.randint(len(train_files), (1,)).item()]
        data = np.memmap(data_file, dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

# ---------------- MODEL INIT ----------------
iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    print("Init scratch")
    model = train_helper.createModel(model_folder_name, None, args.chpn)
    opt_args = parsed_opt_args + [device_type]
    optimizer = model.configure_optimizers(*opt_args)
    model.to(device)

elif init_from == 'resume':
    ckpt = args.chpr
    print(f"\n[RESUME DEBUG] {ckpt}")

    model, model_sd, opt_args, opt_sd, iter_num = train_helper.createModel(
        model_folder_name, ckpt, None, from_scratch=False
    )

    print(f"loaded iter_num: {iter_num}")

    model.load_state_dict(model_sd)
    model.to(device)

    optimizer = model.configure_optimizers(*opt_args)
    optimizer.load_state_dict(opt_sd)

    print("\n[OPT STATE AFTER LOAD]")
    for i, g in enumerate(optimizer.param_groups):
        print(f"group {i}: lr={g['lr']} params={len(g['params'])}")

else:
    raise ValueError("bad init")

# ---------------- PARAM CHECK ----------------
n_params = sum(p.numel() for p in model.parameters())
print(f"\nmodel params: {n_params:,}")

# ---------------- SCALER ----------------
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

print("\n[AMP DEBUG]")
print(f"scaler enabled: {scaler.is_enabled()}")

# ---------------- DDP DEBUG ----------------
if ddp:
    print("\n[DDP DEBUG]")
    print(ddp_world_size, ddp_rank, ddp_local_rank)

model = DDP(model, device_ids=[ddp_local_rank]) if ddp else model

raw_model = model.module if ddp else model

# ---------------- LR ----------------
def get_lr(it):
    if it < warmup_iters:
        return LEARNING_RATE * (it + 1) / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (LEARNING_RATE - min_lr)

# ---------------- TRAIN LOOP ----------------
X, Y = get_batch('train')
t0 = time.time()

prev_val_loss = None

while True:

    lr = get_lr(iter_num)

    if iter_num % 50 == 0:
        print(f"\n[LR DEBUG] iter={iter_num} lr={lr}")

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = {}
        model.eval()
        for split in ['train', 'val']:
            ls = []
            for _ in range(eval_iters):
                x, y = get_batch(split)
                with ctx:
                    _, loss = model(x, y)
                ls.append(loss.item())
            losses[split] = sum(ls) / len(ls)
        model.train()

        train_ppl = math.exp(losses['train'])
        val_ppl = math.exp(losses['val'])

        print("\n[LOSS DEBUG]")
        print(losses)
        print(f"train ppl {train_ppl:.2f} val ppl {val_ppl:.2f}")

        if prev_val_loss is not None:
            print(f"Δval loss: {losses['val'] - prev_val_loss}")
        prev_val_loss = losses['val']

    for micro in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num > max_iters:
        break

    iter_num += 1

if ddp:
    destroy_process_group()