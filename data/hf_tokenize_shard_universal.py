#!/usr/bin/env python3
"""
Tokenize and shard Hugging Face text datasets into nanoGPT-style .bin files.

Works with:
- Dataset Viewer Parquet files when available.
- datasets.load_dataset(..., streaming=True) fallback for repos whose viewer is disabled
  because they require a loading script / trust_remote_code.
- Single text column datasets.
- Multi-column text datasets such as title + abstract.
- Existing validation split or deterministic holdout validation without train/val leakage.

Examples:
  python hf_tokenize_shard_universal.py --dataset gfissore/arxiv-abstracts-2021 \
    --output_dir /data/arxiv --text_columns title abstract --append_eot
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import requests
import tiktoken
from tqdm import tqdm

try:
    from huggingface_hub import HfApi, hf_hub_url
except Exception:  # pragma: no cover
    HfApi = None
    hf_hub_url = None

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

try:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
except Exception as e:  # pragma: no cover
    raise SystemExit("Install dependencies first: pip install datasets tiktoken pyarrow requests tqdm numpy") from e

BYTES_PER_GB = 1024 ** 3
DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/parquet"

_WORKER_ENC = None
_WORKER_APPEND_EOT = False
_WORKER_DTYPE = None


@dataclass
class ParquetEntry:
    config: str
    split: str
    url: str
    filename: str
    size: Optional[int]


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare HF text datasets into trainX.bin shards and val.bin."
    )
    p.add_argument("--dataset", required=True, help="HF dataset repo, e.g. allenai/peS2o")
    p.add_argument("--output_dir", required=True, help="Directory for trainX.bin, val.bin, and metadata")
    p.add_argument("--config", default=None, help="Dataset config/subset, e.g. v2 for allenai/peS2o")
    p.add_argument("--backend", choices=["auto", "parquet", "streaming"], default="auto",
                   help="auto tries parquet first, then streaming. Use streaming for loading-script datasets.")
    p.add_argument("--parquet_revision", default=None,
                   help="Optional Hub revision/branch containing .parquet files, e.g. refs/pr/12 for allenai/peS2o when the Dataset Viewer /parquet API returns 501.")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Pass trust_remote_code=True to datasets.load_dataset/get_dataset_config_names.")
    p.add_argument("--list_configs", action="store_true", help="Print configs and exit.")

    p.add_argument("--text_column", default=None,
                   help="Single text column. Kept for compatibility; equivalent to --text_columns COLUMN.")
    p.add_argument("--text_columns", nargs="*", default=None,
                   help="One or more columns to concatenate into a document, e.g. --text_columns title abstract")
    p.add_argument("--text_separator", default="\n\n", help="Separator used when joining multiple text columns")
    p.add_argument("--skip_missing_text_columns", action="store_true",
                   help="Skip missing requested text columns instead of raising.")

    p.add_argument("--encoding", default="gpt2", help="tiktoken encoding, e.g. gpt2, cl100k_base")
    p.add_argument("--append_eot", action="store_true", help="Append EOT token to each document")
    p.add_argument("--hf_token", default=None, help="HF token for private/gated data")

    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split", default=None, help="Existing validation split: validation, valid, val, etc.")
    p.add_argument("--val_size", type=float, default=0.0005,
                   help="Validation fraction when no validation split exists")
    p.add_argument("--seed", type=int, default=2357,
                   help="Seed for deterministic holdout hashing")

    p.add_argument("--shard_size", type=float, default=12.0, help="Max train shard size in GiB")
    p.add_argument("--data_batch", type=float, default=2.0,
                   help="Raw staged parquet batch size = data_batch * shard_size")
    p.add_argument("--row_group_batch_size", type=int, default=4096,
                   help="Rows per Parquet batch")
    p.add_argument("--download_timeout", type=int, default=120)
    p.add_argument("--num_proc", type=int, default=1)
    p.add_argument("--heartbeat_interval", type=float, default=15.0)

    p.add_argument("--max_train_docs", type=int, default=None,
                   help="Optional cap for debugging/subsampling train docs after routing")
    p.add_argument("--max_val_docs", type=int, default=None,
                   help="Optional cap for debugging/subsampling validation docs")
    p.add_argument("--max_train_download_gb", type=float, default=None,
                   help="Parquet backend only: cap selected train Parquet files by cumulative declared download size in GiB")
    p.add_argument("--max_val_download_gb", type=float, default=None,
                   help="Parquet backend only: cap selected validation Parquet files by cumulative declared download size in GiB")
    p.add_argument("--parquet_subset_strategy", choices=("random", "strided", "head"), default="random",
                   help="How to choose Parquet files when a max_*_download_gb cap is set. random gives the least order-biased subset; strided spreads across sorted filenames; head keeps the old first-N behavior.")
    p.add_argument("--max_stream_text_gb", type=float, default=None,
                   help="Streaming backend only: stop after writing this many GiB of UTF-8 text for each output pass")
    p.add_argument("--dry_run", action="store_true",
                   help="Resolve metadata, estimate downloadable bytes where possible, then exit")
    args = p.parse_args()

    if args.text_column and args.text_columns:
        raise ValueError("Use either --text_column or --text_columns, not both.")
    if args.text_column:
        args.text_columns = [args.text_column]
    if not args.text_columns:
        args.text_columns = ["text"]

    if args.shard_size <= 0:
        raise ValueError("--shard_size must be > 0")
    if args.data_batch <= 0:
        raise ValueError("--data_batch must be > 0")
    if not 0.0 < args.val_size < 1.0:
        raise ValueError("--val_size must be between 0 and 1")
    if args.num_proc <= 0:
        raise ValueError("--num_proc must be >= 1")
    for name in ("max_train_download_gb", "max_val_download_gb", "max_stream_text_gb"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be > 0 when provided")
    return args


def human_bytes(num: Optional[int]) -> str:
    if num is None:
        return "unknown"
    num = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PiB"


def human_count(num: float) -> str:
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if abs_num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return f"{num:.0f}"


def configure_hf_cache(output_dir: str, hf_token: Optional[str]) -> str:
    hf_cache_dir = os.path.join(output_dir, "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_cache_dir, "modules")
    os.environ["HF_TOKEN_PATH"] = os.path.join(hf_cache_dir, "token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    return hf_cache_dir


def make_session(hf_token: Optional[str]) -> requests.Session:
    s = requests.Session()
    if hf_token:
        s.headers.update({"Authorization": f"Bearer {hf_token}"})
    s.headers.update({"User-Agent": "hf-tokenize-shard-universal/2.0"})
    return s


def choose_dtype(enc):
    max_token = getattr(enc, "max_token_value", None)
    if max_token is None:
        return np.uint32
    if max_token < 2**16:
        return np.uint16
    if max_token < 2**32:
        return np.uint32
    return np.uint64


def resolve_config(dataset: str, requested_config: Optional[str], token: Optional[str], trust_remote_code: bool) -> Optional[str]:
    # Important for datasets like allenai/peS2o on recent versions of `datasets`:
    # `get_dataset_config_names()` may fail because old dataset scripts are no longer
    # supported locally, even though the Dataset Viewer Parquet API still has usable
    # converted parquet metadata. If the user explicitly supplied --config, trust it
    # and let the selected backend validate it against the actual parquet metadata.
    if requested_config is not None:
        return requested_config

    try:
        configs = list(get_dataset_config_names(dataset, token=token, trust_remote_code=trust_remote_code))
    except TypeError:
        configs = list(get_dataset_config_names(dataset, token=token))
    except RuntimeError as e:
        msg = str(e)
        if "Dataset scripts are no longer supported" in msg or "trust_remote_code" in msg:
            raise ValueError(
                f"Could not inspect configs for {dataset} with your installed datasets package. "
                "Pass --config explicitly, for example --config v2 for allenai/peS2o."
            ) from e
        raise

    if not configs:
        return None
    if "default" in configs:
        return "default"
    if len(configs) == 1:
        return configs[0]
    raise ValueError(f"Dataset has multiple configs; pass --config. Available configs: {configs[:50]}")


def fetch_parquet_metadata(session: requests.Session, dataset: str, timeout: int) -> Dict:
    r = session.get(DATASETS_SERVER_URL, params={"dataset": dataset}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _head_size(session: requests.Session, url: str, timeout: int) -> Optional[int]:
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        if r.ok and r.headers.get("content-length"):
            return int(r.headers["content-length"])
    except Exception:
        pass
    return None


def infer_config_split_from_parquet_path(path: str, target_config: Optional[str]) -> Optional[Tuple[str, str]]:
    parts = path.strip("/").split("/")
    if len(parts) < 2 or not path.endswith(".parquet"):
        return None

    # Common converted layout: <config>/<split>/<file>.parquet, or for peS2o: v2/train-00000-of-00068.parquet
    if target_config is not None:
        if parts[0] != target_config:
            return None
        if len(parts) >= 3:
            return parts[0], parts[1]
        stem = os.path.basename(path)
        if "-" in stem:
            return parts[0], stem.split("-", 1)[0]
        return None

    if len(parts) >= 3:
        return parts[0], parts[1]
    stem = os.path.basename(path)
    if "-" in stem:
        return parts[0], stem.split("-", 1)[0]
    return "default", parts[-2]


def list_parquet_entries_from_hub_revision(session: requests.Session, dataset: str, target_config: Optional[str],
                                           revision: str, token: Optional[str], timeout: int) -> List[ParquetEntry]:
    if HfApi is None or hf_hub_url is None:
        raise RuntimeError("huggingface_hub is required for --parquet_revision fallback. Install/update: pip install -U huggingface_hub")

    api = HfApi(token=token)
    items = []
    try:
        items = list(api.list_repo_tree(repo_id=dataset, repo_type="dataset", revision=revision, recursive=True, expand=True))
        paths = [(getattr(x, "path", None), getattr(x, "size", None), getattr(x, "type", None)) for x in items]
    except Exception:
        files = api.list_repo_files(repo_id=dataset, repo_type="dataset", revision=revision)
        paths = [(x, None, "file") for x in files]

    out: List[ParquetEntry] = []
    for path, size, typ in paths:
        if not path or not path.endswith(".parquet"):
            continue
        if typ not in (None, "file"):
            continue
        parsed = infer_config_split_from_parquet_path(path, target_config)
        if parsed is None:
            continue
        cfg, split = parsed
        url = hf_hub_url(repo_id=dataset, filename=path, repo_type="dataset", revision=revision)
        if size is None:
            size = _head_size(session, url, timeout)
        out.append(ParquetEntry(cfg, split, url, path, int(size) if size is not None else None))

    if not out:
        raise ValueError(f"No Parquet files found on {dataset}@{revision} for config={target_config!r}")
    return out


def normalize_parquet_entries(parquet_json: Dict, target_config: Optional[str]) -> List[ParquetEntry]:
    entries = parquet_json.get("parquet_files")
    if entries is None:
        raise ValueError("/parquet response missing parquet_files")
    out: List[ParquetEntry] = []
    for e in entries:
        cfg = e.get("config") or e.get("subset") or "default"
        if target_config is not None and cfg != target_config:
            continue
        split = e.get("split")
        url = e.get("url")
        filename = e.get("filename")
        if not split or not url or not filename:
            continue
        out.append(ParquetEntry(cfg, split, url, filename, int(e["size"]) if e.get("size") is not None else None))
    if not out:
        raise ValueError("No Parquet entries for selected config")
    return out


def choose_splits(available: Sequence[str], train_split: str, val_split: Optional[str]) -> Tuple[str, Optional[str]]:
    available = sorted(set(available))
    if train_split not in available:
        if len(available) == 1:
            train_split = available[0]
        else:
            raise ValueError(f"Requested train split '{train_split}' not found. Available splits: {available}")
    if val_split:
        if val_split not in available:
            raise ValueError(f"Requested val split '{val_split}' not found. Available splits: {available}")
        return train_split, val_split
    for cand in ("validation", "valid", "val", "dev"):
        if cand in available:
            return train_split, cand
    return train_split, None


def get_streaming_splits(dataset: str, config: Optional[str], token: Optional[str], trust_remote_code: bool) -> List[str]:
    try:
        return list(get_dataset_split_names(dataset, config, token=token, trust_remote_code=trust_remote_code))
    except Exception:
        return ["train"]


def example_to_text(example: Dict, columns: Sequence[str], sep: str, skip_missing: bool) -> str:
    parts = []
    for col in columns:
        if col not in example:
            if skip_missing:
                continue
            raise KeyError(f"Missing text column '{col}'. Available columns include: {list(example.keys())[:30]}")
        v = example.get(col)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v if x is not None)
        elif isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False, sort_keys=True)
        elif not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if v:
            parts.append(v)
    return sep.join(parts)


def stable_holdout_keep(dataset: str, split: str, idx: int, text: str, seed: int, val_size: float, want_val: bool) -> bool:
    key = f"{seed}|{dataset}|{split}|{idx}|{text[:512]}".encode("utf-8", errors="ignore")
    h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    frac = h / float(2**64)
    is_val = frac < val_size
    return is_val if want_val else not is_val


def init_tokenizer_worker(encoding_name: str, append_eot: bool, dtype_name: str):
    global _WORKER_ENC, _WORKER_APPEND_EOT, _WORKER_DTYPE
    _WORKER_ENC = tiktoken.get_encoding(encoding_name)
    _WORKER_APPEND_EOT = append_eot
    _WORKER_DTYPE = np.dtype(dtype_name)


def encode_worker(text: str) -> np.ndarray:
    ids = _WORKER_ENC.encode_ordinary(text)
    if _WORKER_APPEND_EOT:
        ids.append(_WORKER_ENC.eot_token)
    return np.asarray(ids, dtype=_WORKER_DTYPE)


def encode_local(enc, text: str, append_eot: bool, dtype) -> np.ndarray:
    ids = enc.encode_ordinary(text)
    if append_eot:
        ids.append(enc.eot_token)
    return np.asarray(ids, dtype=dtype)


def make_pool(args, dtype):
    if args.num_proc <= 1:
        return None
    return mp.Pool(args.num_proc, initializer=init_tokenizer_worker,
                   initargs=(args.encoding, args.append_eot, np.dtype(dtype).name))


class TrainShardWriter:
    def __init__(self, output_dir: str, dtype, max_shard_bytes: int, start_idx: int = 0):
        self.output_dir = output_dir
        self.dtype = np.dtype(dtype)
        self.max_shard_bytes = int(max_shard_bytes)
        self.current_idx = start_idx - 1
        self.file = None
        self.current_bytes = 0
        self.total_tokens = 0
        self.total_bytes = 0
        self._open_new()

    @property
    def current_name(self) -> str:
        return f"train{self.current_idx}.bin"

    def _open_new(self):
        if self.file is not None:
            self.file.flush(); self.file.close()
        self.current_idx += 1
        path = os.path.join(self.output_dir, self.current_name)
        self.file = open(path, "ab")
        self.current_bytes = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"Opened {path} existing_bytes={self.current_bytes}")

    def write_tokens(self, arr: np.ndarray):
        if arr.size == 0:
            return
        arr = arr.astype(self.dtype, copy=False)
        pos = 0
        itemsize = self.dtype.itemsize
        while pos < arr.size:
            remaining = self.max_shard_bytes - self.current_bytes
            if remaining <= 0:
                self._open_new(); remaining = self.max_shard_bytes
            take = min(remaining // itemsize, arr.size - pos)
            if take <= 0:
                self._open_new(); continue
            chunk = arr[pos:pos + take]
            chunk.tofile(self.file)
            self.current_bytes += chunk.nbytes
            self.total_bytes += chunk.nbytes
            self.total_tokens += chunk.size
            pos += take

    def flush(self):
        if self.file is not None:
            self.file.flush(); os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.flush(); self.file.close(); self.file = None


class SingleBinWriter:
    def __init__(self, filename: str, dtype):
        self.filename = filename
        self.dtype = np.dtype(dtype)
        self.file = open(filename, "wb")
        self.total_tokens = 0
        self.total_bytes = 0

    def write_tokens(self, arr: np.ndarray):
        if arr.size == 0:
            return
        arr = arr.astype(self.dtype, copy=False)
        arr.tofile(self.file)
        self.total_tokens += arr.size
        self.total_bytes += arr.nbytes

    def flush(self):
        self.file.flush(); os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.flush(); self.file.close(); self.file = None


def write_checkpoint(output_dir: str, data: Dict):
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def encode_and_write_texts(texts: List[str], writer, args, enc, dtype, pool) -> int:
    tokens = 0
    if not texts:
        return 0
    if pool is None:
        for text in texts:
            arr = encode_local(enc, text, args.append_eot, dtype)
            writer.write_tokens(arr)
            tokens += arr.size
    else:
        chunksize = max(1, min(256, len(texts) // max(1, args.num_proc * 4) or 1))
        for arr in pool.map(encode_worker, texts, chunksize=chunksize):
            writer.write_tokens(arr)
            tokens += arr.size
    return tokens


def stream_examples(dataset: str, config: Optional[str], split: str, args) -> Iterator[Dict]:
    kwargs = dict(split=split, streaming=True, token=args.hf_token, trust_remote_code=args.trust_remote_code)
    ds = load_dataset(dataset, config, **kwargs) if config else load_dataset(dataset, **kwargs)
    for ex in ds:
        yield ex


def process_streaming_split(dataset: str, config: Optional[str], source_split: str, writer, args, enc, dtype,
                            route: str, max_docs: Optional[int], max_text_bytes: Optional[int] = None) -> Dict:
    pool = make_pool(args, dtype)
    stats = {"rows_seen": 0, "docs_written": 0, "text_bytes": 0, "tokens": 0, "started_at": time.time()}
    last_log = time.time()
    batch: List[str] = []
    try:
        for idx, ex in enumerate(stream_examples(dataset, config, source_split, args)):
            text = example_to_text(ex, args.text_columns, args.text_separator, args.skip_missing_text_columns)
            if not text:
                continue
            stats["rows_seen"] += 1
            keep = True
            if route == "holdout_train":
                keep = stable_holdout_keep(dataset, source_split, idx, text, args.seed, args.val_size, want_val=False)
            elif route == "holdout_val":
                keep = stable_holdout_keep(dataset, source_split, idx, text, args.seed, args.val_size, want_val=True)
            elif route in ("all_train", "all_val"):
                keep = True
            else:
                raise ValueError(f"unknown route {route}")
            if not keep:
                continue
            if max_docs is not None and stats["docs_written"] >= max_docs:
                break
            text_nbytes = len(text.encode("utf-8"))
            if max_text_bytes is not None and stats["text_bytes"] >= max_text_bytes:
                break
            if max_text_bytes is not None and stats["text_bytes"] + text_nbytes > max_text_bytes and stats["docs_written"] > 0:
                break
            batch.append(text)
            stats["docs_written"] += 1
            stats["text_bytes"] += text_nbytes
            if len(batch) >= args.row_group_batch_size:
                stats["tokens"] += encode_and_write_texts(batch, writer, args, enc, dtype, pool)
                batch.clear(); writer.flush()
            now = time.time()
            if now - last_log >= args.heartbeat_interval:
                elapsed = max(now - stats["started_at"], 1e-9)
                print(f"[{route}] split={source_split} rows_seen={human_count(stats['rows_seen'])} "
                      f"docs_written={human_count(stats['docs_written'])} tokens={human_count(stats['tokens'])} "
                      f"docs/sec={human_count(stats['docs_written'] / elapsed)}")
                last_log = now
        if batch:
            stats["tokens"] += encode_and_write_texts(batch, writer, args, enc, dtype, pool)
            writer.flush()
    finally:
        if pool is not None:
            pool.close(); pool.join()
    return stats


def group_parquet_entries(entries: List[ParquetEntry]) -> Dict[str, List[ParquetEntry]]:
    grouped: Dict[str, List[ParquetEntry]] = {}
    for e in entries:
        grouped.setdefault(e.split, []).append(e)
    for s in grouped:
        grouped[s].sort(key=lambda e: e.filename)
    return grouped


def _take_until_declared_cap(candidates: List[ParquetEntry], max_bytes: int) -> List[ParquetEntry]:
    selected: List[ParquetEntry] = []
    total = 0
    for entry in candidates:
        size = entry.size
        if size is None:
            if not selected:
                selected.append(entry)
            continue
        if selected and total + size > max_bytes:
            continue
        selected.append(entry)
        total += size
        if total >= max_bytes:
            break
    return selected


def _strided_order(entries: List[ParquetEntry]) -> List[ParquetEntry]:
    n = len(entries)
    if n <= 2:
        return list(entries)
    # Visit sorted files from across the entire range instead of consuming only the prefix.
    # Example for 68 files: 0, 67, 34, 17, 51, 8, 25, 42, 59, ...
    order = []
    seen = set()
    step = max(1, n // 2)
    while math.gcd(step, n) != 1 and step > 1:
        step -= 1
    idx = 0
    for _ in range(n):
        if idx not in seen:
            seen.add(idx)
            order.append(entries[idx])
        idx = (idx + step) % n
    return order


def cap_parquet_entries_by_declared_download(
    entries: List[ParquetEntry],
    max_gb: Optional[float],
    strategy: str = "random",
    seed: int = 2357,
) -> List[ParquetEntry]:
    """Keep whole Parquet files until the cumulative declared size reaches max_gb.

    Because Parquet objects are downloaded whole, the cap is file-granular. The default
    strategy is random to avoid the distribution bias that can come from taking only the
    first N sorted shards. Returned entries are sorted by filename for reproducible staging
    and checkpoints after selection.
    """
    if max_gb is None:
        return entries
    max_bytes = int(max_gb * BYTES_PER_GB)
    if not entries:
        return entries

    sized_total = sum(e.size or 0 for e in entries)
    if sized_total and sized_total <= max_bytes:
        return entries

    if strategy == "head":
        candidates = list(entries)
    elif strategy == "strided":
        candidates = _strided_order(entries)
    elif strategy == "random":
        candidates = list(entries)
        rng = random.Random(seed)
        rng.shuffle(candidates)
    else:
        raise ValueError(f"Unknown parquet subset strategy: {strategy}")

    selected = _take_until_declared_cap(candidates, max_bytes)
    if not selected:
        selected = [candidates[0]]
    selected.sort(key=lambda e: e.filename)
    return selected


def download_file(session: requests.Session, url: str, dest: str, timeout: int):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
                                        desc=f"Downloading {os.path.basename(dest)}", leave=False) as pbar:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk); pbar.update(len(chunk))
    os.replace(tmp, dest)


def remove_with_retries(path: str, attempts: int = 8, delay: float = 0.25):
    """Remove a file on Windows even if pyarrow/AV briefly holds the handle."""
    for i in range(attempts):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            gc.collect()
            if i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


def process_parquet_entries(session, entries: List[ParquetEntry], writer, args, enc, dtype, route: str,
                            source_split: str, max_docs: Optional[int]) -> Dict:
    if pq is None:
        raise RuntimeError("pyarrow is required for parquet backend: pip install pyarrow")
    pool = make_pool(args, dtype)
    staging = os.path.join(args.output_dir, f"_staging_{route}")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    stats = {"rows_seen": 0, "docs_written": 0, "text_bytes": 0, "tokens": 0, "downloaded_bytes": 0, "started_at": time.time()}
    try:
        for file_idx, entry in enumerate(entries):
            local = os.path.join(staging, os.path.basename(entry.filename))
            download_file(session, entry.url, local, args.download_timeout)
            if entry.size:
                stats["downloaded_bytes"] += entry.size
            batch_texts: List[str] = []
            with open(local, "rb") as parquet_handle:
                pf = pq.ParquetFile(parquet_handle)
                for rb in pf.iter_batches(batch_size=args.row_group_batch_size, use_threads=True):
                    cols = rb.column_names
                    pycols = {name: rb.column(cols.index(name)).to_pylist() for name in args.text_columns if name in cols}
                    missing = [c for c in args.text_columns if c not in pycols]
                    if missing and not args.skip_missing_text_columns:
                        raise KeyError(f"Missing text columns {missing}; available columns: {cols}")
                    n = rb.num_rows
                    for i in range(n):
                        ex = {c: pycols[c][i] for c in pycols}
                        text = example_to_text(ex, args.text_columns, args.text_separator, args.skip_missing_text_columns)
                        if not text:
                            continue
                        global_idx = stats["rows_seen"]
                        stats["rows_seen"] += 1
                        keep = True
                        if route == "holdout_train":
                            keep = stable_holdout_keep(args.dataset, source_split, global_idx, text, args.seed, args.val_size, False)
                        elif route == "holdout_val":
                            keep = stable_holdout_keep(args.dataset, source_split, global_idx, text, args.seed, args.val_size, True)
                        if not keep:
                            continue
                        if max_docs is not None and stats["docs_written"] >= max_docs:
                            break
                        batch_texts.append(text)
                        stats["docs_written"] += 1
                        stats["text_bytes"] += len(text.encode("utf-8"))
                        if len(batch_texts) >= args.row_group_batch_size:
                            stats["tokens"] += encode_and_write_texts(batch_texts, writer, args, enc, dtype, pool)
                            batch_texts.clear(); writer.flush()
                    if max_docs is not None and stats["docs_written"] >= max_docs:
                        break
                del pf
            if batch_texts:
                stats["tokens"] += encode_and_write_texts(batch_texts, writer, args, enc, dtype, pool)
                batch_texts.clear(); writer.flush()
            remove_with_retries(local)
            print(f"[{route}] completed {file_idx + 1}/{len(entries)} files; docs={human_count(stats['docs_written'])}; tokens={human_count(stats['tokens'])}")
            if max_docs is not None and stats["docs_written"] >= max_docs:
                break
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if pool is not None:
            pool.close(); pool.join()
    return stats


def parquet_plan(args, session, config) -> Optional[Tuple[List[ParquetEntry], str, Optional[str]]]:
    errors = []
    try:
        pj = fetch_parquet_metadata(session, args.dataset, args.download_timeout)
        entries = normalize_parquet_entries(pj, config)
        if pj.get("partial"):
            print("WARNING: Dataset Viewer reports partial Parquet conversion. Use --backend streaming for the full dataset.")
        train_split, val_split = choose_splits([e.split for e in entries], args.train_split, args.val_split)
        return entries, train_split, val_split
    except Exception as e:
        errors.append(f"Dataset Viewer /parquet failed: {type(e).__name__}: {e}")

    # Some script-based datasets, notably allenai/peS2o, return 501 from /parquet while
    # converted Parquet files are still available on Hub revisions/PR refs.
    revisions: List[str] = []
    if args.parquet_revision:
        revisions.append(args.parquet_revision)
    elif args.dataset == "allenai/peS2o":
        revisions.extend(["refs/pr/12", "refs/pr/13", "refs/convert/parquet"])
    else:
        revisions.append("refs/convert/parquet")

    for revision in revisions:
        try:
            entries = list_parquet_entries_from_hub_revision(
                session=session, dataset=args.dataset, target_config=config, revision=revision,
                token=args.hf_token, timeout=args.download_timeout,
            )
            print(f"Using direct Hub Parquet files from revision: {revision}")
            train_split, val_split = choose_splits([e.split for e in entries], args.train_split, args.val_split)
            return entries, train_split, val_split
        except Exception as e:
            errors.append(f"Hub revision {revision} failed: {type(e).__name__}: {e}")

    if args.backend == "parquet":
        raise RuntimeError("Parquet backend unavailable. " + " | ".join(errors))
    print("Parquet backend unavailable; falling back to streaming. " + " | ".join(errors))
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    hf_cache = configure_hf_cache(args.output_dir, args.hf_token)
    session = make_session(args.hf_token)

    if args.list_configs:
        try:
            cfgs = get_dataset_config_names(args.dataset, token=args.hf_token, trust_remote_code=args.trust_remote_code)
        except TypeError:
            cfgs = get_dataset_config_names(args.dataset, token=args.hf_token)
        except RuntimeError:
            # Fallback for script-based legacy datasets: infer configs from Dataset Viewer parquet metadata.
            pj = fetch_parquet_metadata(session, args.dataset, args.download_timeout)
            cfgs = sorted({e.get("config") or e.get("subset") for e in pj.get("parquet_files", []) if (e.get("config") or e.get("subset"))})
        print("\n".join(cfgs))
        return

    config = resolve_config(args.dataset, args.config, args.hf_token, args.trust_remote_code)
    enc = tiktoken.get_encoding(args.encoding)
    dtype = choose_dtype(enc)
    max_shard_bytes = int(args.shard_size * BYTES_PER_GB)

    backend = args.backend
    plan = None
    if backend in ("auto", "parquet"):
        plan = parquet_plan(args, session, config)
        if plan is not None:
            backend = "parquet"
    if backend == "auto":
        backend = "streaming"

    if backend == "parquet":
        entries, train_split, val_split = plan
        grouped = group_parquet_entries(entries)
        grouped[train_split] = cap_parquet_entries_by_declared_download(
            grouped[train_split], args.max_train_download_gb, args.parquet_subset_strategy, args.seed
        )
        if val_split:
            grouped[val_split] = cap_parquet_entries_by_declared_download(
                grouped[val_split], args.max_val_download_gb, args.parquet_subset_strategy, args.seed + 1
            )
        total_train_download = sum(e.size or 0 for e in grouped[train_split])
        total_val_download = sum(e.size or 0 for e in grouped[val_split]) if val_split else 0
        print(json.dumps({
            "dataset": args.dataset, "config": config, "backend": "parquet",
            "parquet_revision": args.parquet_revision,
            "train_split": train_split, "val_split": val_split,
            "declared_train_download_bytes": total_train_download,
            "declared_val_download_bytes": total_val_download,
            "train_parquet_files": len(grouped[train_split]),
            "val_parquet_files": len(grouped[val_split]) if val_split else 0,
            "max_train_download_gb": args.max_train_download_gb,
            "max_val_download_gb": args.max_val_download_gb,
            "parquet_subset_strategy": args.parquet_subset_strategy,
            "text_columns": args.text_columns, "hf_cache": hf_cache,
        }, indent=2))
        if args.dry_run:
            return
        writer = TrainShardWriter(args.output_dir, dtype, max_shard_bytes)
        try:
            train_route = "all_train" if val_split else "holdout_train"
            train_stats = process_parquet_entries(session, grouped[train_split], writer, args, enc, dtype, train_route, train_split, args.max_train_docs)
        finally:
            writer.close()
        val_writer = SingleBinWriter(os.path.join(args.output_dir, "val.bin"), dtype)
        try:
            if val_split:
                val_stats = process_parquet_entries(session, grouped[val_split], val_writer, args, enc, dtype, "all_val", val_split, args.max_val_docs)
            else:
                val_stats = process_parquet_entries(session, grouped[train_split], val_writer, args, enc, dtype, "holdout_val", train_split, args.max_val_docs)
        finally:
            val_writer.close()
    else:
        splits = get_streaming_splits(args.dataset, config, args.hf_token, args.trust_remote_code)
        train_split, val_split = choose_splits(splits, args.train_split, args.val_split)
        print(json.dumps({
            "dataset": args.dataset, "config": config, "backend": "streaming",
            "train_split": train_split, "val_split": val_split,
            "download_bytes": "unknown for streaming/loading-script datasets; run will stream source files",
            "max_stream_text_gb": args.max_stream_text_gb,
            "text_columns": args.text_columns, "hf_cache": hf_cache,
        }, indent=2))
        if args.dry_run:
            return
        max_stream_text_bytes = int(args.max_stream_text_gb * BYTES_PER_GB) if args.max_stream_text_gb is not None else None
        writer = TrainShardWriter(args.output_dir, dtype, max_shard_bytes)
        try:
            train_route = "all_train" if val_split else "holdout_train"
            train_stats = process_streaming_split(args.dataset, config, train_split, writer, args, enc, dtype, train_route, args.max_train_docs, max_stream_text_bytes)
        finally:
            writer.close()
        val_writer = SingleBinWriter(os.path.join(args.output_dir, "val.bin"), dtype)
        try:
            if val_split:
                val_stats = process_streaming_split(args.dataset, config, val_split, val_writer, args, enc, dtype, "all_val", args.max_val_docs, max_stream_text_bytes)
            else:
                val_stats = process_streaming_split(args.dataset, config, train_split, val_writer, args, enc, dtype, "holdout_val", args.max_val_docs, max_stream_text_bytes)
        finally:
            val_writer.close()

    metadata = {
        "dataset": args.dataset,
        "config": config,
        "backend": backend,
        "parquet_revision": args.parquet_revision,
        "text_columns": args.text_columns,
        "encoding": args.encoding,
        "append_eot": args.append_eot,
        "dtype": np.dtype(dtype).name,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "output_dir": args.output_dir,
    }
    write_checkpoint(args.output_dir, metadata)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
