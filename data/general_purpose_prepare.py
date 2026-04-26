import os
import re
import json
import time
import math
import shutil
import random
import argparse
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import tiktoken
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import get_dataset_config_names


BYTES_PER_GB = 1024 ** 3
DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/parquet"

_WORKER_ENC = None
_WORKER_APPEND_EOT = False
_WORKER_DTYPE = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare a Hugging Face text dataset into sharded nanoGPT-style trainX.bin files and a monolithic val.bin."
    )
    p.add_argument("--dataset", type=str, required=True,
                   help="Hugging Face dataset repo, e.g. HuggingFaceFW/fineweb-edu")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory where trainX.bin / val.bin will be written")
    p.add_argument("--config", type=str, default=None,
                   help="Optional dataset config/subset. If omitted, the dataset default config is used.")
    p.add_argument("--text_column", type=str, default="text",
                   help="Column containing the text")
    p.add_argument("--encoding", type=str, default="gpt2",
                   help="tiktoken encoding name, e.g. gpt2, cl100k_base")
    p.add_argument("--train_split", type=str, default="train",
                   help="Which split to treat as train")
    p.add_argument("--val_split", type=str, default=None,
                   help="Existing validation split name if present, e.g. validation or val")
    p.add_argument("--test_split", type=str, default=None,
                   help="Unused here; accepted for compatibility with the original script")
    p.add_argument("--val_size", type=float, default=0.0005,
                   help="Validation fraction if dataset has no validation split")
    p.add_argument("--seed", type=int, default=2357,
                   help="Random seed for deterministic holdout splitting")
    p.add_argument("--append_eot", action="store_true",
                   help="Append end-of-text token to each document")
    p.add_argument("--hf_token", type=str, default=None,
                   help="Optional Hugging Face access token for authenticated downloads")
    p.add_argument("--shard_size", type=float, default=30.0,
                   help="Maximum size per training shard in GB. Default: 30")
    p.add_argument("--data_batch", type=float, default=2.0,
                   help="Amount of raw source data staged per cycle, expressed as a multiple of shard_size. Default: 2")
    p.add_argument("--row_group_batch_size", type=int, default=4096,
                   help="Number of rows per parquet batch while tokenizing. Default: 4096")
    p.add_argument("--download_timeout", type=int, default=120,
                   help="HTTP timeout in seconds for metadata and download requests. Default: 120")
    p.add_argument("--num_proc", type=int, default=1,
                   help="Number of worker processes for parallel tokenization. Default: 1")
    p.add_argument("--heartbeat_interval", type=float, default=15.0,
                   help="Seconds between progress heartbeat logs during tokenization. Default: 15")
    args = p.parse_args()
    print(f"Num of processors used to tokenize: {args.num_proc}")
    return args


def choose_dtype(enc):
    max_token = getattr(enc, "max_token_value", None)
    if max_token is None:
        return np.uint32
    if max_token < 2**16:
        return np.uint16
    if max_token < 2**32:
        return np.uint32
    return np.uint64


def configure_hf_cache(output_dir, hf_token=None):
    hf_cache_dir = os.path.join(output_dir, "hf_cache")
    hub_cache_dir = os.path.join(hf_cache_dir, "hub")
    datasets_cache_dir = os.path.join(hf_cache_dir, "datasets")
    modules_cache_dir = os.path.join(hf_cache_dir, "modules")
    token_path = os.path.join(hf_cache_dir, "token")

    os.makedirs(hub_cache_dir, exist_ok=True)
    os.makedirs(datasets_cache_dir, exist_ok=True)
    os.makedirs(modules_cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = hub_cache_dir
    os.environ["HF_DATASETS_CACHE"] = datasets_cache_dir
    os.environ["HF_MODULES_CACHE"] = modules_cache_dir
    os.environ["HF_TOKEN_PATH"] = token_path

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    return hf_cache_dir


def make_session(hf_token: Optional[str]) -> requests.Session:
    session = requests.Session()
    if hf_token:
        session.headers.update({"Authorization": f"Bearer {hf_token}"})
    session.headers.update({"User-Agent": "hf-local-sharded-tokenizer/1.0"})
    return session


def resolve_config(dataset: str, requested_config: Optional[str], hf_token: Optional[str]) -> str:
    configs = list(get_dataset_config_names(path=dataset, token=hf_token))
    if not configs:
        raise ValueError(f"No dataset configs found for {dataset}")

    if requested_config is not None:
        if requested_config not in configs:
            raise ValueError(
                f"Requested config '{requested_config}' not found for {dataset}. "
                f"Available configs include: {configs[:20]}{'...' if len(configs) > 20 else ''}"
            )
        return requested_config

    if "default" in configs:
        return "default"
    if len(configs) == 1:
        return configs[0]

    raise ValueError(
        f"Dataset {dataset} has multiple configs and no --config was given. "
        f"Examples: {configs[:20]}{'...' if len(configs) > 20 else ''}"
    )


def fetch_parquet_metadata(session: requests.Session, dataset: str, timeout: int) -> Dict:
    resp = session.get(DATASETS_SERVER_URL, params={"dataset": dataset}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def normalize_parquet_entries(parquet_json: Dict, target_config: str) -> List[Dict]:
    entries = parquet_json.get("parquet_files")
    if entries is None:
        raise ValueError("Unexpected /parquet response format: missing 'parquet_files'")

    out = []
    for entry in entries:
        config = entry.get("config") or entry.get("subset")
        split = entry.get("split")
        url = entry.get("url")
        filename = entry.get("filename")
        size = entry.get("size")

        if config != target_config:
            continue
        if not split or not url or not filename:
            continue

        out.append({
            "config": config,
            "split": split,
            "url": url,
            "filename": filename,
            "size": int(size) if size is not None else None,
        })

    if not out:
        raise ValueError(
            f"No parquet files found for config '{target_config}'. "
            f"This implementation expects parquet metadata from the dataset viewer."
        )

    return out


def choose_splits(entries: List[Dict], train_split: str, val_split: Optional[str]) -> Tuple[str, Optional[str]]:
    available_splits = sorted(set(e["split"] for e in entries))
    print(f"Available splits for selected config: {available_splits}")

    if train_split not in available_splits:
        if len(available_splits) == 1:
            train_split = available_splits[0]
        else:
            raise ValueError(
                f"Requested train split '{train_split}' not found. Available splits: {available_splits}"
            )

    resolved_val = None
    if val_split is not None:
        if val_split not in available_splits:
            raise ValueError(
                f"Requested val split '{val_split}' not found. Available splits: {available_splits}"
            )
        resolved_val = val_split
    else:
        for candidate in ("validation", "val"):
            if candidate in available_splits:
                resolved_val = candidate
                break

    return train_split, resolved_val


def group_entries_by_split(entries: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for entry in entries:
        grouped.setdefault(entry["split"], []).append(entry)

    for split, lst in grouped.items():
        lst.sort(key=lambda x: x["filename"])
    return grouped


def human_gib(num_bytes: int) -> str:
    return f"{num_bytes / BYTES_PER_GB:.2f} GiB"


def human_count(num: float) -> str:
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if abs_num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return f"{num:.0f}"


def extract_train_idx(path_or_name: str) -> Optional[int]:
    m = re.fullmatch(r"train(\d+)\.bin", os.path.basename(path_or_name))
    if not m:
        return None
    return int(m.group(1))


class TrainShardWriter:
    def __init__(self, output_dir: str, dtype, max_shard_bytes: int, start_shard_idx: int = 0):
        self.output_dir = output_dir
        self.dtype = np.dtype(dtype)
        self.max_shard_bytes = int(max_shard_bytes)
        self.current_file = None
        self.current_bytes = 0
        self.current_shard_idx = start_shard_idx - 1
        self.total_tokens = 0
        self.total_bytes = 0
        self._open_new_shard()

    @property
    def current_shard_name(self) -> str:
        return f"train{self.current_shard_idx}.bin"

    def _open_new_shard(self):
        if self.current_file is not None:
            self.current_file.flush()
            self.current_file.close()

        self.current_shard_idx += 1
        filename = os.path.join(self.output_dir, f"train{self.current_shard_idx}.bin")
        self.current_file = open(filename, "ab")
        self.current_bytes = os.path.getsize(filename) if os.path.exists(filename) else 0
        print(f"Opened {filename} (existing_bytes={self.current_bytes})")

    def write_tokens(self, token_array: np.ndarray):
        if token_array.size == 0:
            return

        pos = 0
        itemsize = self.dtype.itemsize
        while pos < token_array.size:
            remaining_bytes = self.max_shard_bytes - self.current_bytes
            if remaining_bytes <= 0:
                self._open_new_shard()
                remaining_bytes = self.max_shard_bytes

            tokens_fit = remaining_bytes // itemsize
            if tokens_fit <= 0:
                self._open_new_shard()
                continue

            take = min(tokens_fit, token_array.size - pos)
            chunk = token_array[pos:pos + take]
            chunk.tofile(self.current_file)

            self.current_bytes += chunk.nbytes
            self.total_bytes += chunk.nbytes
            self.total_tokens += chunk.size
            pos += take

    def flush(self):
        if self.current_file is not None:
            self.current_file.flush()
            os.fsync(self.current_file.fileno())

    def close(self):
        if self.current_file is not None:
            self.current_file.flush()
            self.current_file.close()
            self.current_file = None


class SingleBinWriter:
    def __init__(self, filename: str, dtype):
        self.filename = filename
        self.dtype = np.dtype(dtype)
        self.file = open(filename, "wb")
        self.total_tokens = 0
        self.total_bytes = 0

    def write_tokens(self, token_array: np.ndarray):
        if token_array.size == 0:
            return
        token_array.tofile(self.file)
        self.total_tokens += token_array.size
        self.total_bytes += token_array.nbytes

    def flush(self):
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None


def encode_text(enc, text: str, append_eot: bool, dtype) -> np.ndarray:
    ids = enc.encode_ordinary(text)
    if append_eot:
        ids.append(enc.eot_token)
    return np.asarray(ids, dtype=dtype)


def init_tokenizer_worker(encoding_name: str, append_eot: bool, dtype_name: str):
    global _WORKER_ENC, _WORKER_APPEND_EOT, _WORKER_DTYPE
    _WORKER_ENC = tiktoken.get_encoding(encoding_name)
    _WORKER_APPEND_EOT = append_eot
    _WORKER_DTYPE = np.dtype(dtype_name)


def encode_text_worker(text: str) -> np.ndarray:
    global _WORKER_ENC, _WORKER_APPEND_EOT, _WORKER_DTYPE
    if text is None:
        return np.empty(0, dtype=_WORKER_DTYPE)
    if not isinstance(text, str):
        text = str(text)

    ids = _WORKER_ENC.encode_ordinary(text)
    if _WORKER_APPEND_EOT:
        ids.append(_WORKER_ENC.eot_token)
    return np.asarray(ids, dtype=_WORKER_DTYPE)


def make_token_pool(args, dtype):
    if args.num_proc <= 1:
        return None
    return mp.Pool(
        processes=args.num_proc,
        initializer=init_tokenizer_worker,
        initargs=(args.encoding, args.append_eot, np.dtype(dtype).name),
    )


def select_stage_batch(entries: List[Dict], start_idx: int, max_stage_bytes: int) -> Tuple[List[Dict], int]:
    if start_idx >= len(entries):
        return [], start_idx

    batch = []
    total = 0
    idx = start_idx

    while idx < len(entries):
        entry = entries[idx]
        size = entry["size"]
        if size is None:
            if not batch:
                batch.append(entry)
                idx += 1
            break

        if not batch and size > max_stage_bytes:
            batch.append(entry)
            idx += 1
            break

        if total + size > max_stage_bytes:
            break

        batch.append(entry)
        total += size
        idx += 1

    if not batch:
        batch.append(entries[start_idx])
        idx = start_idx + 1

    return batch, idx


def staged_file_path(staging_dir: str, entry: Dict) -> str:
    base = os.path.basename(entry["filename"])
    return os.path.join(staging_dir, base)


def download_file(session: requests.Session, url: str, dest_path: str, timeout: int):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"

    with session.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        desc = f"Downloading {os.path.basename(dest_path)}"
        with open(tmp_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=False,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    os.replace(tmp_path, dest_path)


def download_stage_batch(session: requests.Session, batch_entries: List[Dict], staging_dir: str, timeout: int) -> List[str]:
    local_paths = []
    batch_bytes = sum(e["size"] or 0 for e in batch_entries)
    print(
        f"Downloading staged batch: {len(batch_entries)} files, "
        f"declared_size={human_gib(batch_bytes)}"
    )
    for entry in batch_entries:
        local_path = staged_file_path(staging_dir, entry)
        download_file(session, entry["url"], local_path, timeout)
        local_paths.append(local_path)
    return local_paths


def cleanup_staging_dir(staging_dir: str):
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)


def cleanup_future_train_shards(output_dir: str, committed_shard_idx: int):
    for name in os.listdir(output_dir):
        idx = extract_train_idx(name)
        if idx is not None and idx > committed_shard_idx:
            path = os.path.join(output_dir, name)
            print(f"Removing uncommitted shard from previous interrupted run: {path}")
            os.remove(path)


def checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, "download_checkpoint.txt")


def write_checkpoint(output_dir: str, checkpoint: Dict):
    path = checkpoint_path(output_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, sort_keys=True)


def read_checkpoint(output_dir: str) -> Optional[Dict]:
    path = checkpoint_path(output_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def locate_resume_index(entries: List[Dict], checkpoint: Dict) -> int:
    last_filename = checkpoint.get("last_completed_filename")
    if last_filename is None:
        return 0

    for i, entry in enumerate(entries):
        if entry["filename"] == last_filename:
            return i + 1

    raise ValueError(
        f"Checkpoint refers to filename '{last_filename}', but it was not found in the selected config/split."
    )


def maybe_log_tokenization_heartbeat(
    args,
    log_prefix: str,
    current_file: str,
    batch_idx: int,
    file_started_at: float,
    last_log_at: float,
    file_rows_scanned: int,
    file_rows_written: int,
    file_text_bytes: int,
    file_tokens: int,
) -> float:
    now = time.time()
    if now - last_log_at < args.heartbeat_interval:
        return last_log_at

    elapsed = max(now - file_started_at, 1e-9)
    rows_per_sec = file_rows_scanned / elapsed
    tokens_per_sec = file_tokens / elapsed
    print(
        f"[{log_prefix} heartbeat] file={os.path.basename(current_file)} | "
        f"batch={batch_idx} | scanned_rows={human_count(file_rows_scanned)} | "
        f"written_rows={human_count(file_rows_written)} | "
        f"text_scanned={human_gib(file_text_bytes)} | tokens_written={human_count(file_tokens)} | "
        f"rows_per_sec={human_count(rows_per_sec)} | tokens_per_sec={human_count(tokens_per_sec)}"
    )
    return now


def tokenize_parquet_file(
    local_path: str,
    writer,
    args,
    enc,
    dtype,
    log_prefix: str,
    progress_state: Dict,
    routing_mode: str,
    rng: Optional[random.Random] = None,
    token_pool=None,
):
    parquet = pq.ParquetFile(local_path)
    rows_before = progress_state["rows"]
    tokens_before = progress_state["tokens"]
    text_bytes_before = progress_state["text_bytes"]

    file_started_at = time.time()
    last_log_at = file_started_at
    file_rows_scanned = 0
    file_rows_written = 0
    file_tokens = 0
    file_text_bytes = 0
    batch_idx = 0

    batch_iter = parquet.iter_batches(
        batch_size=args.row_group_batch_size,
        columns=[args.text_column],
        use_threads=True,
    )

    for record_batch in batch_iter:
        batch_idx += 1
        col = record_batch.column(0)
        py_list = col.to_pylist()

        selected_texts = []

        for text in py_list:
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)

            if routing_mode == "val_only":
                keep = True
            elif routing_mode == "train_only":
                keep = True
            elif routing_mode == "holdout_split":
                assert rng is not None
                keep = rng.random() < args.val_size
            elif routing_mode == "holdout_train_pass":
                assert rng is not None
                keep = not (rng.random() < args.val_size)
            elif routing_mode == "holdout_val_pass":
                assert rng is not None
                keep = rng.random() < args.val_size
            else:
                raise ValueError(f"Unknown routing_mode: {routing_mode}")

            progress_state["rows"] += 1
            progress_state["text_bytes"] += len(text.encode("utf-8"))
            file_rows_scanned += 1
            file_text_bytes += len(text.encode("utf-8"))

            if keep:
                selected_texts.append(text)

        if selected_texts:
            if token_pool is None:
                for text in selected_texts:
                    token_array = encode_text(enc, text, args.append_eot, dtype)
                    writer.write_tokens(token_array)
                    progress_state["tokens"] += token_array.size
                    file_tokens += token_array.size
                    file_rows_written += 1
            else:
                chunksize = max(
                    1,
                    min(
                        256,
                        len(selected_texts) // (args.num_proc * 4) if len(selected_texts) > args.num_proc else 1,
                    ),
                )
                encoded_batches = token_pool.map(encode_text_worker, selected_texts, chunksize=chunksize)
                for token_array in encoded_batches:
                    writer.write_tokens(token_array)
                    progress_state["tokens"] += token_array.size
                    file_tokens += token_array.size
                    file_rows_written += 1

        last_log_at = maybe_log_tokenization_heartbeat(
            args=args,
            log_prefix=log_prefix,
            current_file=local_path,
            batch_idx=batch_idx,
            file_started_at=file_started_at,
            last_log_at=last_log_at,
            file_rows_scanned=file_rows_scanned,
            file_rows_written=file_rows_written,
            file_text_bytes=file_text_bytes,
            file_tokens=file_tokens,
        )

    elapsed = max(time.time() - file_started_at, 1e-9)
    rows_done = progress_state["rows"] - rows_before
    tokens_done = progress_state["tokens"] - tokens_before
    text_bytes_done = progress_state["text_bytes"] - text_bytes_before
    print(
        f"{log_prefix}: processed {os.path.basename(local_path)} "
        f"(rows={rows_done}, written_rows={file_rows_written}, text={human_gib(text_bytes_done)}, "
        f"tokens={tokens_done}, elapsed={elapsed:.1f}s, tokens_per_sec={human_count(tokens_done / elapsed)})"
    )


def summarize_remaining(entries: List[Dict], next_index: int) -> Tuple[int, int]:
    remaining_files = max(0, len(entries) - next_index)
    remaining_bytes = sum((e["size"] or 0) for e in entries[next_index:])
    return remaining_files, remaining_bytes


def build_train_shards(
    session: requests.Session,
    train_entries: List[Dict],
    args,
    enc,
    dtype,
    output_dir: str,
    max_shard_bytes: int,
    max_stage_bytes: int,
):
    ckpt = read_checkpoint(output_dir)
    if ckpt is None:
        start_index = 0
        start_shard_idx = 0
        ckpt = {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.train_split,
            "last_completed_filename": None,
            "current_train_shard": "train0.bin",
        }
        write_checkpoint(output_dir, ckpt)
        print("Created new download_checkpoint.txt")
    else:
        if ckpt.get("dataset") != args.dataset or ckpt.get("config") != args.config or ckpt.get("split") != args.train_split:
            raise ValueError(
                "Existing download_checkpoint.txt does not match the requested dataset/config/train_split."
            )
        start_index = locate_resume_index(train_entries, ckpt)
        shard_name = ckpt.get("current_train_shard", "train0.bin")
        shard_idx = extract_train_idx(shard_name)
        if shard_idx is None:
            raise ValueError(f"Invalid current_train_shard in checkpoint: {shard_name}")
        cleanup_future_train_shards(output_dir, shard_idx)
        start_shard_idx = shard_idx + 1
        print(
            f"Resuming training from source file index {start_index}/{len(train_entries)}. "
            f"Last committed shard: {shard_name}"
        )

    staging_dir = os.path.join(output_dir, "_staging_train")
    cleanup_staging_dir(staging_dir)

    writer = TrainShardWriter(
        output_dir=output_dir,
        dtype=dtype,
        max_shard_bytes=max_shard_bytes,
        start_shard_idx=start_shard_idx,
    )

    progress_state = {"rows": 0, "tokens": 0, "text_bytes": 0}
    cumulative_downloaded_bytes = 0

    next_index = start_index
    batch_number = 0
    token_pool = make_token_pool(args, dtype)

    try:
        while next_index < len(train_entries):
            batch_entries, new_next_index = select_stage_batch(train_entries, next_index, max_stage_bytes)
            batch_number += 1

            remaining_files, remaining_bytes = summarize_remaining(train_entries, next_index)
            batch_declared_bytes = sum((e["size"] or 0) for e in batch_entries)

            print(
                f"[train batch {batch_number}] files {next_index}..{new_next_index - 1} / {len(train_entries) - 1} | "
                f"batch_size={human_gib(batch_declared_bytes)} | "
                f"remaining_before_batch={remaining_files} files, {human_gib(remaining_bytes)}"
            )

            local_paths = download_stage_batch(session, batch_entries, staging_dir, args.download_timeout)
            cumulative_downloaded_bytes += batch_declared_bytes

            for entry, local_path in zip(batch_entries, local_paths):
                print(
                    f"[train tokenize] file={os.path.basename(local_path)} | mode=train_only | "
                    f"workers={args.num_proc} | row_group_batch_size={args.row_group_batch_size}"
                )
                tokenize_parquet_file(
                    local_path=local_path,
                    writer=writer,
                    args=args,
                    enc=enc,
                    dtype=dtype,
                    log_prefix="train",
                    progress_state=progress_state,
                    routing_mode="train_only",
                    token_pool=token_pool,
                )
                writer.flush()

                ckpt = {
                    "dataset": args.dataset,
                    "config": args.config,
                    "split": args.train_split,
                    "last_completed_filename": entry["filename"],
                    "current_train_shard": writer.current_shard_name,
                }
                write_checkpoint(output_dir, ckpt)
                print(
                    f"Checkpoint updated: last_completed_filename={entry['filename']} | "
                    f"current_train_shard={writer.current_shard_name}"
                )

            cleanup_staging_dir(staging_dir)
            next_index = new_next_index

            remaining_files, remaining_bytes = summarize_remaining(train_entries, next_index)
            print(
                f"[train batch {batch_number}] done | downloaded_total={human_gib(cumulative_downloaded_bytes)} | "
                f"text_processed_total={human_gib(progress_state['text_bytes'])} | "
                f"tokens_total={progress_state['tokens']} | "
                f"current_shard={writer.current_shard_name} | "
                f"remaining={remaining_files} files, {human_gib(remaining_bytes)}"
            )

    finally:
        cleanup_staging_dir(staging_dir)
        writer.close()
        if token_pool is not None:
            token_pool.close()
            token_pool.join()

    print(
        f"Training complete: train_shards=0..{extract_train_idx(writer.current_shard_name)} | "
        f"text_processed={human_gib(progress_state['text_bytes'])} | tokens={progress_state['tokens']}"
    )


def build_single_bin_from_entries(
    session: requests.Session,
    entries: List[Dict],
    args,
    enc,
    dtype,
    output_filename: str,
    max_stage_bytes: int,
    routing_mode: str,
):
    writer = SingleBinWriter(output_filename, dtype)
    staging_dir = os.path.join(os.path.dirname(output_filename), "_staging_val")
    cleanup_staging_dir(staging_dir)

    progress_state = {"rows": 0, "tokens": 0, "text_bytes": 0}
    cumulative_downloaded_bytes = 0
    next_index = 0
    batch_number = 0
    rng = random.Random(args.seed)
    token_pool = make_token_pool(args, dtype)

    try:
        while next_index < len(entries):
            batch_entries, new_next_index = select_stage_batch(entries, next_index, max_stage_bytes)
            batch_number += 1
            batch_declared_bytes = sum((e["size"] or 0) for e in batch_entries)

            remaining_files, remaining_bytes = summarize_remaining(entries, next_index)
            print(
                f"[{os.path.basename(output_filename)} batch {batch_number}] files {next_index}..{new_next_index - 1} / {len(entries) - 1} | "
                f"batch_size={human_gib(batch_declared_bytes)} | "
                f"remaining_before_batch={remaining_files} files, {human_gib(remaining_bytes)}"
            )

            local_paths = download_stage_batch(session, batch_entries, staging_dir, args.download_timeout)
            cumulative_downloaded_bytes += batch_declared_bytes

            for entry, local_path in zip(batch_entries, local_paths):
                print(
                    f"[{os.path.basename(output_filename)} tokenize] file={os.path.basename(local_path)} | "
                    f"mode={routing_mode} | workers={args.num_proc} | "
                    f"row_group_batch_size={args.row_group_batch_size}"
                )
                tokenize_parquet_file(
                    local_path=local_path,
                    writer=writer,
                    args=args,
                    enc=enc,
                    dtype=dtype,
                    log_prefix=os.path.basename(output_filename),
                    progress_state=progress_state,
                    routing_mode=routing_mode,
                    rng=rng if routing_mode.startswith("holdout_") else None,
                    token_pool=token_pool,
                )
                writer.flush()

            cleanup_staging_dir(staging_dir)
            next_index = new_next_index

            remaining_files, remaining_bytes = summarize_remaining(entries, next_index)
            print(
                f"[{os.path.basename(output_filename)} batch {batch_number}] done | "
                f"downloaded_total={human_gib(cumulative_downloaded_bytes)} | "
                f"text_processed_total={human_gib(progress_state['text_bytes'])} | "
                f"tokens_total={progress_state['tokens']} | "
                f"remaining={remaining_files} files, {human_gib(remaining_bytes)}"
            )

    finally:
        cleanup_staging_dir(staging_dir)
        writer.close()
        if token_pool is not None:
            token_pool.close()
            token_pool.join()

    print(
        f"Wrote {output_filename} | text_processed={human_gib(progress_state['text_bytes'])} | "
        f"tokens={progress_state['tokens']} | output_bytes={human_gib(writer.total_bytes)}"
    )


def main():
    args = parse_args()

    if args.shard_size <= 0:
        raise ValueError("--shard_size must be > 0")
    if args.data_batch <= 0:
        raise ValueError("--data_batch must be > 0")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError("--val_size must be between 0 and 1")
    if args.num_proc <= 0:
        raise ValueError("--num_proc must be >= 1")
    if args.heartbeat_interval <= 0:
        raise ValueError("--heartbeat_interval must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)
    hf_cache_dir = configure_hf_cache(args.output_dir, args.hf_token)

    session = make_session(args.hf_token)
    args.config = resolve_config(args.dataset, args.config, args.hf_token)

    parquet_json = fetch_parquet_metadata(session, args.dataset, args.download_timeout)
    entries = normalize_parquet_entries(parquet_json, args.config)
    train_split, resolved_val_split = choose_splits(entries, args.train_split, args.val_split)

    split_entries = group_entries_by_split(entries)
    train_entries = split_entries[train_split]
    val_entries = split_entries[resolved_val_split] if resolved_val_split is not None else None

    enc = tiktoken.get_encoding(args.encoding)
    dtype = choose_dtype(enc)
    max_shard_bytes = int(args.shard_size * BYTES_PER_GB)
    max_stage_bytes = int(args.data_batch * max_shard_bytes)

    total_train_bytes = sum((e["size"] or 0) for e in train_entries)
    print(f"Hugging Face cache directory: {hf_cache_dir}")
    print(f"Using config: {args.config}")
    print(f"Using dtype: {dtype}")
    print(f"Tokenization workers: {args.num_proc}")
    print(f"Heartbeat interval: {args.heartbeat_interval:.1f}s")
    print(f"Max training shard size: {args.shard_size:.2f} GiB")
    print(f"Max staged raw-data batch size: {human_gib(max_stage_bytes)}")
    print(f"Training split '{train_split}': {len(train_entries)} files, declared_size={human_gib(total_train_bytes)}")

    build_train_shards(
        session=session,
        train_entries=train_entries,
        args=args,
        enc=enc,
        dtype=dtype,
        output_dir=args.output_dir,
        max_shard_bytes=max_shard_bytes,
        max_stage_bytes=max_stage_bytes,
    )

    val_filename = os.path.join(args.output_dir, "val.bin")
    if resolved_val_split is not None and val_entries is not None:
        total_val_bytes = sum((e["size"] or 0) for e in val_entries)
        print(
            f"Building val.bin from explicit validation split '{resolved_val_split}': "
            f"{len(val_entries)} files, declared_size={human_gib(total_val_bytes)}"
        )
        build_single_bin_from_entries(
            session=session,
            entries=val_entries,
            args=args,
            enc=enc,
            dtype=dtype,
            output_filename=val_filename,
            max_stage_bytes=max_stage_bytes,
            routing_mode="val_only",
        )
    else:
        print(
            f"No separate validation split found. Building val.bin from a deterministic second pass "
            f"over train split '{train_split}' with val_size={args.val_size}."
        )
        build_single_bin_from_entries(
            session=session,
            entries=train_entries,
            args=args,
            enc=enc,
            dtype=dtype,
            output_filename=val_filename,
            max_stage_bytes=max_stage_bytes,
            routing_mode="holdout_val_pass",
        )

    print(f"Done. Output written to: {args.output_dir}")


if __name__ == "__main__":
    main()