# saves the fineweb-edu dataset to binary files next to this script
# streams data and writes directly to disk — minimal RAM usage

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

HF_ROOT = SCRIPT_DIR / "hf_cache"
HF_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_ROOT)

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# --- Config ---
MAX_DOCS = 7_500_000       # ~2B tokens
VAL_DOCS = 3_750

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print(f"Streaming FineWeb-Edu, taking {MAX_DOCS:,} documents...")
    print("Writing directly to disk to avoid RAM issues.\n")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    train_file = open(SCRIPT_DIR / "train.bin", "wb")
    val_file = open(SCRIPT_DIR / "val.bin", "wb")

    total_train_tokens = 0
    total_val_tokens = 0

    for i, example in enumerate(tqdm(dataset, total=MAX_DOCS, desc="streaming")):
        if i >= MAX_DOCS:
            break

        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        chunk = np.array(ids, dtype=np.uint16)

        if i < VAL_DOCS:
            val_file.write(chunk.tobytes())
            total_val_tokens += len(ids)
        else:
            train_file.write(chunk.tobytes())
            total_train_tokens += len(ids)

    train_file.close()
    val_file.close()

    # Clean up HF cache
    import shutil
    if HF_ROOT.exists():
        shutil.rmtree(HF_ROOT, ignore_errors=True)

    print(f"\nDone!")
    print(f"Train: {total_train_tokens:,} tokens ({total_train_tokens * 2 / 1e9:.2f} GB)")
    print(f"Val:   {total_val_tokens:,} tokens ({total_val_tokens * 2 / 1e6:.1f} MB)")