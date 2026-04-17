import os
from pathlib import Path

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

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        num_proc=num_proc_load_dataset,
        cache_dir=str(HF_DATASETS),
        streaming=True
    )

    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = SCRIPT_DIR / f"{split}.bin"
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()

    for split in ("train", "val"):
        m = np.memmap(SCRIPT_DIR / f"{split}.bin", dtype=np.uint16, mode="r")
        print(f"{split} has {len(m):,} tokens")