import os
import argparse
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, DatasetDict


def parse_args():
    p = argparse.ArgumentParser(description="Prepare a Hugging Face text dataset into nanoGPT-style .bin files.")
    p.add_argument("--dataset", type=str, required=True,
                   help="Hugging Face dataset repo, e.g. openwebtext or wikitext")
    p.add_argument("--config", type=str, default=None,
                   help="Optional dataset config/subset, e.g. wikitext-103-raw-v1")
    p.add_argument("--text_column", type=str, default="text",
                   help="Column containing the text")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory where train.bin / val.bin / test.bin will be written")
    p.add_argument("--encoding", type=str, default="gpt2",
                   help="tiktoken encoding name, e.g. gpt2, cl100k_base")
    p.add_argument("--num_proc", type=int, default=8,
                   help="Workers for dataset.map()")
    p.add_argument("--num_proc_load_dataset", type=int, default=None,
                   help="Workers for load_dataset(); defaults to num_proc")
    p.add_argument("--train_split", type=str, default="train",
                   help="Which split to treat as train when making a val split from a single split dataset")
    p.add_argument("--val_split", type=str, default=None,
                   help="Existing validation split name if present, e.g. validation or val")
    p.add_argument("--test_split", type=str, default=None,
                   help="Existing test split name if present")
    p.add_argument("--val_size", type=float, default=0.0005,
                   help="Validation fraction if dataset only has one split")
    p.add_argument("--seed", type=int, default=2357,
                   help="Random seed for splitting")
    p.add_argument("--append_eot", action="store_true",
                   help="Append end-of-text token to each document")
    p.add_argument("--streaming", action="store_true",
                   help="Use streaming mode. Note: streaming mode is not compatible with this full .bin materialization workflow.")
    return p.parse_args()


def build_splits(dataset, args):
    split_names = list(dataset.keys())
    print(f"Available splits: {split_names}")

    if args.train_split in dataset and args.val_split and args.val_split in dataset:
        out = DatasetDict({
            "train": dataset[args.train_split],
            "val": dataset[args.val_split],
        })
        if args.test_split and args.test_split in dataset:
            out["test"] = dataset[args.test_split]
        return out

    for candidate in ["validation", "val"]:
        if args.train_split in dataset and candidate in dataset:
            out = DatasetDict({
                "train": dataset[args.train_split],
                "val": dataset[candidate],
            })
            if args.test_split and args.test_split in dataset:
                out["test"] = dataset[args.test_split]
            elif "test" in dataset:
                out["test"] = dataset["test"]
            return out

    if args.train_split in dataset:
        split_dataset = dataset[args.train_split].train_test_split(
            test_size=args.val_size,
            seed=args.seed,
            shuffle=True,
        )
        out = DatasetDict({
            "train": split_dataset["train"],
            "val": split_dataset["test"],
        })
        if args.test_split and args.test_split in dataset:
            out["test"] = dataset[args.test_split]
        elif "test" in dataset:
            out["test"] = dataset["test"]
        return out

    if len(split_names) == 1:
        only = split_names[0]
        split_dataset = dataset[only].train_test_split(
            test_size=args.val_size,
            seed=args.seed,
            shuffle=True,
        )
        return DatasetDict({
            "train": split_dataset["train"],
            "val": split_dataset["test"],
        })

    raise ValueError(
        f"Could not infer train/val splits from available splits: {split_names}. "
        f"Pass --train_split / --val_split / --test_split explicitly."
    )


def choose_dtype(enc):
    max_token = getattr(enc, "max_token_value", None)
    if max_token is None:
        return np.uint32
    if max_token < 2**16:
        return np.uint16
    if max_token < 2**32:
        return np.uint32
    return np.uint64


def main():
    args = parse_args()

    if args.streaming:
        raise ValueError("This script writes fully materialized .bin files and does not support --streaming.")

    if args.num_proc_load_dataset is None:
        args.num_proc_load_dataset = args.num_proc

    enc = tiktoken.get_encoding(args.encoding)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    if args.config is not None:
        dataset = load_dataset(
            args.dataset,
            args.config,
            num_proc=args.num_proc_load_dataset,
        )
    else:
        dataset = load_dataset(
            args.dataset,
            num_proc=args.num_proc_load_dataset,
        )

    split_dataset = build_splits(dataset, args)
    print(split_dataset)

    for split_name, dset in split_dataset.items():
        if args.text_column not in dset.column_names:
            raise ValueError(
                f"Split '{split_name}' does not contain text column '{args.text_column}'. "
                f"Available columns: {dset.column_names}"
            )

    def process(example):
        text = example[args.text_column]
        ids = enc.encode_ordinary(text)
        if args.append_eot:
            ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    print("Tokenizing...")
    tokenized = split_dataset.map(
        process,
        remove_columns=split_dataset[next(iter(split_dataset))].column_names,
        desc="tokenizing splits",
        num_proc=args.num_proc,
    )

    dtype = choose_dtype(enc)
    print(f"Using dtype: {dtype}")

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(args.output_dir, f"{split}.bin")
        print(f"Writing {split} -> {filename} ({arr_len} tokens)")

        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = min(1024, max(1, len(dset)))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {split}.bin"):
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True,
            ).with_format("numpy")

            if len(batch["ids"]) == 0:
                continue

            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()

    print(f"Done. Files written to: {args.output_dir}")


if __name__ == "__main__":
    main()