import os
from pathlib import Path
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import hashlib

# ---------------------------
# Setup directories
# ---------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

HF_ROOT = SCRIPT_DIR / "hf_cache"
HF_DATASETS = HF_ROOT / "datasets"
HF_HUB = HF_ROOT / "hub"
TMP_DIR = SCRIPT_DIR / "tmp"

for p in (HF_ROOT, HF_DATASETS, HF_HUB, TMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS)
os.environ["HF_HUB_CACHE"] = str(HF_HUB)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)

# ---------------------------
# Output files
# ---------------------------
train_file = SCRIPT_DIR / "train.bin"
val_file = SCRIPT_DIR / "val.bin"

# ---------------------------
# Tokenizer
# ---------------------------
enc = tiktoken.get_encoding("gpt2")

# ---------------------------
# Deterministic split function
# ---------------------------
def is_val(text):
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return h % 2000 == 0   # ≈0.05% validation

# ---------------------------
# Load dataset (STREAMING)
# ---------------------------
dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "python-edu",
    split="train",
    streaming=True
)

# ---------------------------
# Open output files
# ---------------------------
train_f = open(train_file, "wb")
val_f = open(val_file, "wb")

train_tokens = 0
val_tokens = 0

# ---------------------------
# Process stream
# ---------------------------
for example in tqdm(dataset, desc="Streaming + tokenizing"):
    text = example["text"]

    # tokenize
    ids = enc.encode(text)
    ids.append(enc.eot_token)

    arr = np.array(ids, dtype=np.uint32)

    # split
    if is_val(text):
        arr.tofile(val_f)
        val_tokens += len(arr)
    else:
        arr.tofile(train_f)
        train_tokens += len(arr)

# ---------------------------
# Cleanup
# ---------------------------
train_f.close()
val_f.close()

# ---------------------------
# Final counts
# ---------------------------
print(f"train has {train_tokens:,} tokens")
print(f"val has {val_tokens:,} tokens")
print(f"total tokens: {train_tokens + val_tokens:,}")