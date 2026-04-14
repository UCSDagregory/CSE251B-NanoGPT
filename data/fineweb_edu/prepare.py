# saves the fineweb-edu dataset to binary files next to this script
# streams data to avoid filling disk with parquet files

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Minimal cache directory
HF_ROOT = SCRIPT_DIR / "hf_cache"
for p in (HF_ROOT,):
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HF_DATASETS_CACHE"] = str(HF_ROOT / "datasets")
os.environ["HF_HUB_CACHE"] = str(HF_ROOT / "hub")

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# --- Config ---
MAX_DOCS = 2_000_000       # ~2B tokens, fits in Kaggle's 30GB disk + 13GB RAM
VAL_DOCS = 1_000            # small val set for quick eval during training
num_proc = 4                # lower proc count to save RAM

enc = tiktoken.get_encoding("gpt2")

def tokenize_doc(text):
    """Tokenize a single document, append EOT separator."""
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    return ids

if __name__ == '__main__':
    print(f"Streaming FineWeb-Edu, taking {MAX_DOCS:,} documents...")

    # Stream the dataset — nothing saved to disk during this step
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Collect documents into train and val lists of token IDs
    all_ids_train = []
    all_ids_val = []
    total_train_tokens = 0
    total_val_tokens = 0

    for i, example in enumerate(tqdm(dataset, total=MAX_DOCS, desc="streaming & tokenizing")):
        if i >= MAX_DOCS:
            break

        ids = tokenize_doc(example["text"])

        if i < VAL_DOCS:
            all_ids_val.append(ids)
            total_val_tokens += len(ids)
        else:
            all_ids_train.append(ids)
            total_train_tokens += len(ids)

    print(f"Train: {len(all_ids_train):,} docs, {total_train_tokens:,} tokens")
    print(f"Val:   {len(all_ids_val):,} docs, {total_val_tokens:,} tokens")

    # Write train.bin
    print("Writing train.bin...")
    train_file = SCRIPT_DIR / "train.bin"
    arr = np.memmap(train_file, dtype=np.uint16, mode="w+", shape=(total_train_tokens,))
    idx = 0
    for ids in tqdm(all_ids_train, desc="writing train.bin"):
        chunk = np.array(ids, dtype=np.uint16)
        arr[idx : idx + len(chunk)] = chunk
        idx += len(chunk)
    arr.flush()
    del arr, all_ids_train

    # Write val.bin
    print("Writing val.bin...")
    val_file = SCRIPT_DIR / "val.bin"
    arr = np.memmap(val_file, dtype=np.uint16, mode="w+", shape=(total_val_tokens,))
    idx = 0
    for ids in tqdm(all_ids_val, desc="writing val.bin"):
        chunk = np.array(ids, dtype=np.uint16)
        arr[idx : idx + len(chunk)] = chunk
        idx += len(chunk)
    arr.flush()
    del arr, all_ids_val

    # Clean up HF cache to free disk space before training
    import shutil
    if HF_ROOT.exists():
        shutil.rmtree(HF_ROOT, ignore_errors=True)
        print("Cleaned up HF cache.")

    print(f"\nDone! Files saved to {SCRIPT_DIR}")
    print(f"  train.bin: {train_file.stat().st_size / 1e9:.2f} GB")
    print(f"  val.bin:   {val_file.stat().st_size / 1e6:.1f} MB")