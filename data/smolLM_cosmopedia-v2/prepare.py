import os
from pathlib import Path
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import hashlib

# ---------------------------
# Setup paths (your original is fine)
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
# Tokenizer
# ---------------------------
enc = tiktoken.get_encoding("gpt2")

# ---------------------------
# Deterministic split
# ---------------------------
def is_val(text):
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return h % 2000 == 0  # ~0.05%

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True 
    )

    # output files
    train_path = SCRIPT_DIR / "train.bin"
    val_path = SCRIPT_DIR / "val.bin"

    train_f = open(train_path, "wb")
    val_f = open(val_path, "wb")

    train_tokens = 0
    val_tokens = 0

    # ---------------------------
    # streaming loop
    # ---------------------------
    for example in tqdm(dataset, desc="Streaming + tokenizing"):

        # 🔍 handle unknown schema safely
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        elif "code" in example:
            text = example["code"]
        else:
            continue  # skip weird rows

        # tokenize (GPT-2)
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

    train_f.close()
    val_f.close()

    # ---------------------------
    # final counts
    # ---------------------------
    print(f"train has {train_tokens:,} tokens")
    print(f"val has {val_tokens:,} tokens")
    print(f"total tokens: {train_tokens + val_tokens:,}")