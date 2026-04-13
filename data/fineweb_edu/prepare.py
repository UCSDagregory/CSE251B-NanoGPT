# saves the fineweb-edu dataset to binary files next to this script
# uses the sample-10BT subset (~10B tokens, ~9.67M documents)

import os
from pathlib import Path

# Resolve this script's directory once
SCRIPT_DIR = Path(__file__).resolve().parent

# Put all Hugging Face intermediate/cache files under this directory
HF_ROOT = SCRIPT_DIR / "hf_cache"
HF_DATASETS = HF_ROOT / "datasets"
HF_HUB = HF_ROOT / "hub"
TMP_DIR = SCRIPT_DIR / "tmp"

for p in (HF_ROOT, HF_DATASETS, HF_HUB, TMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Must be set before importing datasets / huggingface libs
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS)
os.environ["HF_HUB_CACHE"] = str(HF_HUB)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

# validation split: hold out 0.05% for val (similar ratio to openwebtext)
val_fraction = 0.0005

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load the 10BT sample subset of FineWeb-Edu
    # This is ~9.67M documents and ~10B tokens — more than enough for 100M param models.
    # To use a smaller subset for faster iteration, change to "sample-100BT" or
    # add a .select(range(N)) call after loading.
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        num_proc=num_proc_load_dataset,
        cache_dir=str(HF_DATASETS),
    )

    # Create train/val split
    split_dataset = dataset.train_test_split(
        test_size=val_fraction, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)  # add <|endoftext|> separator between documents
        return {"ids": ids, "len": len(ids)}

    # Tokenize
    tokenized = split_dataset.map(
        process,
        remove_columns=dataset.column_names,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Write to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = SCRIPT_DIR / f"{split}.bin"
        dtype = np.uint16  # GPT-2 vocab is 50257, fits in uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True,
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()