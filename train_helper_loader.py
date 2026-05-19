from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import pickle
import queue
import random
import re
import threading
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import sys
import importlib

def loadModule(module_name, file_path, module_dir):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_BIN_RE = re.compile(r"^train([0-9]+)\.bin$")
DEFAULT_SOURCE_CACHE_MAX_BYTES = 10 * 1024**3
DEFAULT_RAW_BUFFER_MAX_BYTES = 10 * 1024**3
STOP = object()


def _now() -> float:
    return time.time()


def _parse_size(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    s = str(value).strip().lower().replace("_", "")
    units = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
        "tb": 1000**4,
        "tib": 1024**4,
    }
    for unit, mult in sorted(units.items(), key=lambda kv: len(kv[0]), reverse=True):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * mult)
    return int(float(s))


def _cpu_count() -> int:
    try:
        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1


def _auto_tokenizer_workers() -> int:
    c = _cpu_count()
    if c <= 4:
        return max(1, c - 1)
    return max(1, min(32, c - 2))


@dataclass
class TokenChunk:
    tokens: np.ndarray
    source_name: str
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceFile:
    filename: str
    size_bytes: Optional[int] = None
    estimated_tokens: Optional[int] = None

    def to_state(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state(cls, state: str | Dict[str, Any], dtype: np.dtype) -> "SourceFile":
        if isinstance(state, str):
            return cls(filename=state)
        size = state.get("size_bytes")
        return cls(
            filename=state["filename"],
            size_bytes=None if size is None else int(size),
            estimated_tokens=state.get("estimated_tokens"),
        )


class ByteBoundedQueue:
    def __init__(self, max_bytes: int, max_items: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        self.max_bytes = int(max_bytes)
        self.max_items = int(max_items)
        self._items: List[Tuple[Any, int]] = []
        self._bytes = 0
        self._closed = False
        self._cv = threading.Condition()

    @property
    def current_bytes(self) -> int:
        with self._cv:
            return self._bytes

    @property
    def current_items(self) -> int:
        with self._cv:
            return len(self._items)

    def put(self, item: Any, size_bytes: int) -> bool:
        size_bytes = max(0, int(size_bytes))
        with self._cv:
            while not self._closed:
                fits_bytes = self._bytes + size_bytes <= self.max_bytes or not self._items
                fits_items = len(self._items) < self.max_items
                if fits_bytes and fits_items:
                    self._items.append((item, size_bytes))
                    self._bytes += size_bytes
                    self._cv.notify_all()
                    return True
                self._cv.wait(timeout=0.25)
            return False

    def get(self) -> Any:
        with self._cv:
            while True:
                if self._items:
                    item, size_bytes = self._items.pop(0)
                    self._bytes -= size_bytes
                    self._cv.notify_all()
                    return item
                if self._closed:
                    return STOP
                self._cv.wait(timeout=0.25)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


@dataclass
class RawTextBatch:
    texts: List[str]
    raw_bytes: int
    first_example_index: int
    last_example_index: int
    source_name: str = ""
    source_epoch: int = 0
    tokenizer_encoding: str = "gpt2"
    append_eot: bool = True


@dataclass
class TokenizedBatch:
    token_arrays: List[List[int]]
    raw_bytes: int
    examples: int
    first_example_index: int
    last_example_index: int
    source_name: str = ""
    source_epoch: int = 0


class SourceBackend(ABC):
    def __init__(self, name: str, weight: float, backend: str, dtype: np.dtype, logger: logging.Logger):
        self.name = name
        self.weight = float(weight)
        self.backend = backend
        self.dtype = dtype
        self.logger = logger
        self.debt_tokens = 0.0
        self.tokens_emitted = 0

    @abstractmethod
    def next_token_chunk(self, target_tokens: int) -> TokenChunk:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HFBinSource(SourceBackend):
    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        dtype: np.dtype,
        logger: logging.Logger,
        hf_token: Optional[str],
        source_cache_dir: Path,
        source_cache_max_bytes: int,
        allow_single_oversized_source: bool,
        max_single_source_download_bytes: Optional[int],
        block_size: int,
    ):
        name = cfg.get("name") or cfg.get("repo_id") or cfg.get("repo")
        weight = float(cfg.get("weight", 1.0))
        super().__init__(name=name, weight=weight, backend="hf_bin", dtype=dtype, logger=logger)
        self.repo_id = cfg.get("repo_id") or cfg.get("repo") or cfg.get("name")
        if not self.repo_id:
            raise ValueError(f"hf_bin source {name!r} missing repo_id/repo")
        self.revision = cfg.get("revision")
        self.repo_type = cfg.get("repo_type", "dataset")
        self.hf_token = hf_token
        self.source_cache_dir = source_cache_dir
        self.source_cache_max_bytes = int(source_cache_max_bytes)
        self.allow_single_oversized_source = bool(allow_single_oversized_source)
        self.max_single_source_download_bytes = max_single_source_download_bytes
        self.block_size = int(block_size)
        self.source_files: List[Dict[str, Any]] = []
        self.shuffled_queue: List[str] = []
        self.current_file: Optional[str] = None
        self.current_file_size_bytes: Optional[int] = None
        self.current_token_offset = 0
        self.rng = random.Random(int(cfg.get("seed", 1234)))
        self.heartbeat_interval_seconds = max(1.0, float(cfg.get("heartbeat_interval_seconds", cfg.get("heartbeat_interval", 30.0))))
        self._active_source_path: Optional[Path] = None

    def next_token_chunk(self, target_tokens: int) -> TokenChunk:
        while True:
            self._ensure_current_file()
            assert self.current_file is not None
            source_file = self.current_file
            source_size = self.current_file_size_bytes
            path = Path(self._download_source_file())
            self._active_source_path = path.resolve()
            try:
                data = np.memmap(path, dtype=self.dtype, mode="r")
                try:
                    start = int(self.current_token_offset)
                    remaining = len(data) - start
                    if remaining <= self.block_size:
                        self._advance_file()
                        continue
                    n = min(int(target_tokens), int(remaining))
                    if n <= self.block_size:
                        self._advance_file()
                        continue
                    end = start + n
                    tokens = np.asarray(data[start:end], dtype=self.dtype).copy()
                    self.current_token_offset = end
                    if self.current_token_offset + self.block_size >= len(data):
                        self._advance_file()
                    self.tokens_emitted += int(tokens.size)
                    return TokenChunk(
                        tokens=tokens,
                        source_name=self.name,
                        backend=self.backend,
                        metadata={
                            "repo_id": self.repo_id,
                            "revision": self.revision,
                            "source_file": source_file,
                            "source_size_bytes": source_size,
                            "source_start_token": start,
                            "source_end_token": end,
                            "exact_position": True,
                        },
                    )
                finally:
                    del data
            finally:
                self._active_source_path = None
                self._prune_source_cache(force=True)

    def _advance_file(self) -> None:
        self.current_file = None
        self.current_file_size_bytes = None
        self.current_token_offset = 0

    def _ensure_current_file(self) -> None:
        if not self.source_files:
            self.source_files = [sf.to_state() for sf in self._discover_source_files()]
        if self.current_file is not None:
            return
        selected = self._pop_next_scheduled_file()
        self.current_file = selected.filename
        self.current_file_size_bytes = selected.size_bytes
        self.current_token_offset = 0

    def _pop_next_scheduled_file(self) -> SourceFile:
        by_name = {sf["filename"]: SourceFile.from_state(sf, self.dtype) for sf in self.source_files}
        while True:
            if not self.shuffled_queue:
                self.shuffled_queue = [sf["filename"] for sf in self.source_files]
                self.rng.shuffle(self.shuffled_queue)
                self.logger.info("hf_bin source=%s refilled shuffled FIFO files=%d", self.name, len(self.shuffled_queue))
            filename = self.shuffled_queue.pop(0)
            selected = by_name.get(filename)
            if selected is None:
                self.logger.warning("hf_bin source=%s skipping stale scheduled file=%s", self.name, filename)
                continue
            if (
                self.max_single_source_download_bytes is not None
                and selected.size_bytes is not None
                and selected.size_bytes > self.max_single_source_download_bytes
            ):
                raise RuntimeError(
                    f"Scheduled source file {selected.filename} is {selected.size_bytes} bytes, "
                    f"larger than max_single_source_download_bytes={self.max_single_source_download_bytes}."
                )
            if (
                not self.allow_single_oversized_source
                and selected.size_bytes is not None
                and selected.size_bytes > self.source_cache_max_bytes
            ):
                raise RuntimeError(
                    f"Scheduled source file {selected.filename} is {selected.size_bytes} bytes, "
                    f"larger than source_cache_max_bytes={self.source_cache_max_bytes}."
                )
            return selected

    def _discover_source_files(self) -> List[SourceFile]:
        try:
            from huggingface_hub import HfApi, list_repo_files
        except ImportError as e:
            raise ImportError("hf_bin requires huggingface_hub") from e

        self.logger.info("hf_bin discovering repo=%s revision=%s", self.repo_id, self.revision)
        source_files: List[SourceFile] = []
        try:
            info = HfApi(token=self.hf_token).repo_info(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                files_metadata=True,
            )
            for s in getattr(info, "siblings", []) or []:
                filename = getattr(s, "rfilename", None) or getattr(s, "filename", None)
                if not filename or not TRAIN_BIN_RE.match(Path(filename).name):
                    continue
                size = getattr(s, "size", None)
                size = None if size is None else int(size)
                source_files.append(SourceFile(filename=filename, size_bytes=size, estimated_tokens=None if size is None else size // self.dtype.itemsize))
        except Exception as e:
            self.logger.warning("hf_bin size discovery failed repo=%s err=%r; using filename listing", self.repo_id, e)
            files = list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type, revision=self.revision, token=self.hf_token)
            source_files = [SourceFile(filename=f) for f in files if TRAIN_BIN_RE.match(Path(f).name)]
        if not source_files:
            raise FileNotFoundError(f"No train[0-9]*.bin files found in {self.repo_id}")
        def sort_key(sf: SourceFile) -> Tuple[int, str]:
            m = TRAIN_BIN_RE.match(Path(sf.filename).name)
            return (int(m.group(1)) if m else 0, sf.filename)
        return sorted(source_files, key=sort_key)

    def _download_source_file(self) -> str:
        assert self.current_file is not None
        self._prune_source_cache_for_incoming(self.current_file_size_bytes)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("hf_bin requires huggingface_hub") from e
        local_dir = self._repo_cache_dir()
        local_dir.mkdir(parents=True, exist_ok=True)
        t0 = _now()
        self.logger.info(
            "hf_bin heartbeat source=%s action=ensure_cached repo=%s file=%s size=%s revision=%s",
            self.name,
            self.repo_id,
            self.current_file,
            self.current_file_size_bytes,
            self.revision,
        )
        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.current_file,
            repo_type=self.repo_type,
            revision=self.revision,
            token=self.hf_token,
            local_dir=str(local_dir),
        )
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = None
        self.logger.info(
            "hf_bin heartbeat source=%s action=cached repo=%s file=%s elapsed=%.1fs path=%s bytes=%s",
            self.name,
            self.repo_id,
            self.current_file,
            _now() - t0,
            path,
            size,
        )
        return path

    def _repo_cache_dir(self) -> Path:
        revision = self.revision or "default"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", f"{self.repo_type}__{self.repo_id}__{revision}")
        return self.source_cache_dir / safe

    def _source_cache_size_bytes(self) -> int:
        total = 0
        if not self.source_cache_dir.exists():
            return 0
        for p in self.source_cache_dir.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def _prune_source_cache_for_incoming(self, incoming_size: Optional[int]) -> None:
        if incoming_size is None:
            self._prune_source_cache(force=True)
            return
        if self._source_cache_size_bytes() + int(incoming_size) <= self.source_cache_max_bytes:
            return
        self._prune_source_cache(force=True, target_free_bytes=int(incoming_size))

    def _prune_source_cache(self, *, force: bool, target_free_bytes: int = 0) -> None:
        if not force or not self.source_cache_dir.exists():
            return
        active = {self._active_source_path.resolve()} if self._active_source_path else set()
        files: List[Tuple[float, int, Path]] = []
        total = 0
        for p in self.source_cache_dir.rglob("*"):
            if not p.is_file():
                continue
            try:
                rp = p.resolve()
                size = p.stat().st_size
                total += size
            except OSError:
                continue
            if rp not in active:
                files.append((p.stat().st_mtime, size, p))
        target_max = self.source_cache_max_bytes
        if target_free_bytes > self.source_cache_max_bytes and self.allow_single_oversized_source:
            target_max = 0
        elif target_free_bytes:
            target_max = max(0, self.source_cache_max_bytes - target_free_bytes)
        if total <= target_max:
            return
        files.sort(key=lambda x: x[0])
        deleted = 0
        deleted_bytes = 0
        for _, size, p in files:
            if total <= target_max:
                break
            try:
                p.unlink()
                total -= size
                deleted += 1
                deleted_bytes += size
            except OSError as e:
                self.logger.warning("hf_bin could not prune cache file=%s err=%r", p, e)
        self._remove_empty_dirs(self.source_cache_dir)
        if deleted:
            self.logger.info("hf_bin pruned source cache deleted=%d bytes=%d", deleted, deleted_bytes)

    def _remove_empty_dirs(self, root: Path) -> None:
        for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "weight": self.weight,
            "debt_tokens": self.debt_tokens,
            "tokens_emitted": self.tokens_emitted,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "repo_type": self.repo_type,
            "source_files": self.source_files,
            "shuffled_queue": self.shuffled_queue,
            "current_file": self.current_file,
            "current_file_size_bytes": self.current_file_size_bytes,
            "current_token_offset": self.current_token_offset,
            "rng_state": base64.b64encode(pickle.dumps(self.rng.getstate(), protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii"),
            "state_exactness": "exact_file_token_offset",
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.debt_tokens = float(state.get("debt_tokens", self.debt_tokens))
        self.tokens_emitted = int(state.get("tokens_emitted", 0))
        self.source_files = list(state.get("source_files", self.source_files))
        self.shuffled_queue = list(state.get("shuffled_queue", self.shuffled_queue))
        self.current_file = state.get("current_file", self.current_file)
        self.current_file_size_bytes = state.get("current_file_size_bytes", self.current_file_size_bytes)
        self.current_token_offset = int(state.get("current_token_offset", self.current_token_offset))
        if state.get("rng_state"):
            self.rng.setstate(pickle.loads(base64.b64decode(state["rng_state"].encode("ascii"))))



class RawTextManager:
    """Shared raw-text runtime for all raw sources.

    All hf_raw_text sources share one byte-bounded raw queue, one global token
    batch budget, and one tokenizer worker pool. Individual sources own their
    persistent reader/iterator state, but they do not each get a 10GiB buffer or
    a private worker pool.
    """

    def __init__(
        self,
        *,
        dtype: np.dtype,
        logger: logging.Logger,
        raw_buffer_bytes: int,
        raw_queue_max_batches: int,
        token_queue_max_batches: int,
        tokenizer_workers: int,
        fail_fast: bool = True,
    ):
        self.dtype = dtype
        self.logger = logger
        self.raw_buffer_bytes = int(raw_buffer_bytes)
        self.raw_queue_max_batches = int(raw_queue_max_batches)
        self.token_queue_max_batches = int(token_queue_max_batches)
        self.tokenizer_workers = int(tokenizer_workers)
        self.fail_fast = bool(fail_fast)
        self.raw_queue = ByteBoundedQueue(self.raw_buffer_bytes, self.raw_queue_max_batches)
        self._token_slots = threading.Semaphore(self.token_queue_max_batches)
        self._source_token_queues: Dict[str, "queue.Queue[Any]"] = {}
        self._threads: List[threading.Thread] = []
        self._started = False
        self._closed = False
        self._lock = threading.RLock()
        self.errors: List[str] = []
        self.raw_batches_submitted = 0
        self.token_batches_produced = 0

    def register_source(self, source_name: str) -> "queue.Queue[Any]":
        with self._lock:
            if source_name not in self._source_token_queues:
                self._source_token_queues[source_name] = queue.Queue()
            return self._source_token_queues[source_name]

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._closed = False
            for i in range(self.tokenizer_workers):
                t = threading.Thread(target=self._tokenizer_loop, args=(i,), name=f"raw-tokenizer-global-{i}", daemon=True)
                self._threads.append(t)
                t.start()
            self.logger.info(
                "RawTextManager started tokenizer_workers=%d raw_buffer_bytes=%d raw_queue_max_batches=%d token_queue_max_batches=%d",
                self.tokenizer_workers,
                self.raw_buffer_bytes,
                self.raw_queue_max_batches,
                self.token_queue_max_batches,
            )

    def submit(self, batch: RawTextBatch, size_bytes: int) -> bool:
        if self._closed:
            return False
        self.start()
        ok = self.raw_queue.put(batch, size_bytes)
        if ok:
            self.raw_batches_submitted += 1
        return ok

    def get_tokenized(self, source_name: str, timeout: float = 1.0) -> Any:
        q = self.register_source(source_name)
        item = q.get(timeout=timeout)
        if item is not STOP:
            self._token_slots.release()
        return item

    def _tokenizer_loop(self, worker_id: int) -> None:
        enc_cache: Dict[str, Any] = {}
        eot_cache: Dict[str, Optional[int]] = {}
        try:
            import tiktoken
        except Exception as e:
            self.errors.append(f"worker {worker_id}: could not import tiktoken: {e!r}")
            self.logger.exception("RawTextManager tokenizer import failed worker=%d", worker_id)
            return

        while not self._closed:
            item = self.raw_queue.get()
            if item is STOP:
                break
            assert isinstance(item, RawTextBatch)
            try:
                encoding_name = getattr(item, "tokenizer_encoding", "gpt2")
                append_eot = bool(getattr(item, "append_eot", True))
                if encoding_name not in enc_cache:
                    enc_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
                    eot_cache[encoding_name] = getattr(enc_cache[encoding_name], "eot_token", None)
                enc = enc_cache[encoding_name]
                eot = eot_cache[encoding_name]
                if append_eot and eot is None:
                    raise ValueError(f"append_eot=True but tokenizer {encoding_name!r} has no eot_token")
                # Do not call encode_ordinary_batch() inside these worker threads.
                # tiktoken's batch API creates/schedules its own internal futures;
                # during shutdown that can raise "cannot schedule new futures after
                # interpreter shutdown", and during normal operation it creates
                # nested thread pools on top of our global tokenizer worker pool.
                # The global RawTextManager workers already provide parallelism.
                token_arrays = [enc.encode_ordinary(t) for t in item.texts]
                if append_eot:
                    token_arrays = [list(a) + [int(eot)] for a in token_arrays]
                else:
                    token_arrays = [list(a) for a in token_arrays]

                self._token_slots.acquire()
                q = self.register_source(item.source_name)
                q.put(TokenizedBatch(
                    token_arrays=token_arrays,
                    raw_bytes=item.raw_bytes,
                    examples=len(item.texts),
                    first_example_index=item.first_example_index,
                    last_example_index=item.last_example_index,
                    source_name=item.source_name,
                    source_epoch=item.source_epoch,
                ))
                self.token_batches_produced += 1
            except Exception as e:
                # If Python is already shutting down, tokenizer failures are
                # teardown noise, not a data/read/tokenization failure that should
                # corrupt the committed loader state.
                if sys.is_finalizing() or "interpreter shutdown" in str(e):
                    self.logger.warning(
                        "RawTextManager tokenizer stopped during interpreter shutdown worker=%d source=%s error=%r",
                        worker_id,
                        getattr(item, "source_name", None),
                        e,
                    )
                    break
                msg = f"worker {worker_id}: {e!r}"
                self.errors.append(msg)
                self.logger.exception("RawTextManager tokenizer failed worker=%d source=%s", worker_id, getattr(item, "source_name", None))
                if self.fail_fast:
                    self.close(wait=False)
                    break

    def close(self, *, wait: bool = False, timeout: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            self.raw_queue.close()
            for q in self._source_token_queues.values():
                try:
                    q.put_nowait(STOP)
                except Exception:
                    pass
        if wait:
            current = threading.current_thread()
            for t in list(self._threads):
                if t is not current and t.is_alive():
                    t.join(timeout=timeout)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "raw_buffer_bytes": self.raw_buffer_bytes,
            "raw_queue_max_batches": self.raw_queue_max_batches,
            "token_queue_max_batches": self.token_queue_max_batches,
            "tokenizer_workers": self.tokenizer_workers,
            "raw_batches_submitted": self.raw_batches_submitted,
            "token_batches_produced": self.token_batches_produced,
            "errors": list(self.errors[-20:]),
            "global_shared_raw_resources": True,
        }



class HFRawTextSource(SourceBackend):
    """Demand-driven persistent raw HF text source.

    The dataset object/iterator is kept around after first use, but raw reading is
    demand-driven by next_token_chunk(). This prevents inactive small sources from
    filling the global raw/token buffers while still sharing one RawTextManager
    and tokenizer worker pool across all raw sources.

    Exhaustion is always repeat: when the iterator ends, the source_epoch is
    incremented and a new iterator is opened.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        dtype: np.dtype,
        logger: logging.Logger,
        hf_token: Optional[str],
        raw_manager: RawTextManager,
    ):
        name = cfg.get("name") or cfg.get("dataset") or cfg.get("repo_id") or "hf_raw_text"
        weight = float(cfg.get("weight", 1.0))
        super().__init__(name=name, weight=weight, backend="hf_raw_text", dtype=dtype, logger=logger)
        self.dataset = cfg.get("dataset") or cfg.get("repo_id") or cfg.get("path")
        if not self.dataset:
            raise ValueError(f"hf_raw_text source {name!r} missing dataset/repo_id/path")
        self.dataset_config = cfg.get("dataset_config") or cfg.get("name_config") or cfg.get("config")
        self.split = cfg.get("split", "train")
        self.text_column = cfg.get("text_column", "text")
        self.hf_token = hf_token
        # Some HF datasets use custom dataset loading code and require
        # trust_remote_code=True. Keep it source-local so only explicitly
        # trusted datasets execute remote code.
        self.trust_remote_code = cfg.get("trust_remote_code", None)
        self.load_dataset_kwargs = dict(cfg.get("load_dataset_kwargs", {}))
        self.tokenizer_encoding = cfg.get("tokenizer") or cfg.get("tokenizer_encoding", "gpt2")
        self.append_eot = bool(cfg.get("append_eot", True))
        self.shuffle_buffer = int(cfg.get("shuffle_buffer", 0))
        self.seed = int(cfg.get("seed", 1234))
        self.read_batch_examples = int(cfg.get("read_batch_examples", 128))
        self.read_batch_bytes = _parse_size(cfg.get("read_batch_bytes"), 8 * 1024**2)
        self.max_raw_doc_bytes = None if cfg.get("max_raw_doc_bytes") is None else _parse_size(cfg.get("max_raw_doc_bytes"), 0)
        self.streaming = bool(cfg.get("streaming", True))
        self.fail_fast = bool(cfg.get("fail_fast", True))
        self.prefetch_batches = int(cfg.get("prefetch_batches", cfg.get("raw_prefetch_batches_per_source", 8)))
        self.heartbeat_interval_seconds = max(1.0, float(cfg.get("heartbeat_interval_seconds", cfg.get("heartbeat_interval", 30.0))))
        self.exhaustion_policy = "repeat"
        self.raw_manager = raw_manager
        self.token_queue = raw_manager.register_source(self.name)
        self._closed = False
        self._started = False
        self._iterator: Optional[Any] = None
        self._dataset_obj: Optional[Any] = None
        self._inflight_batches = 0
        self._lock = threading.RLock()

        self.source_epoch = 0
        self.examples_read = 0
        self.examples_tokenized = 0
        self.examples_emitted = 0
        self.raw_bytes_read = 0
        self.raw_bytes_emitted = 0
        self.raw_batches = 0
        self.token_batches = 0
        self.load_dataset_seconds_total = 0.0
        self.load_dataset_calls = 0
        self.last_example_index_read = -1
        self.last_example_index_emitted = -1
        self.errors: List[str] = []

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self.raw_manager.start()
            self._open_iterator()
            self.logger.info(
                "hf_raw_text source=%s initialized dataset=%s config=%s exhaustion_policy=repeat global_raw_manager=True prefetch_batches=%d",
                self.name,
                self.dataset,
                self.dataset_config,
                self.prefetch_batches,
            )

    def _open_iterator(self) -> None:
        from datasets import load_dataset

        kwargs: Dict[str, Any] = {"path": self.dataset, "split": self.split, "streaming": self.streaming}
        if self.dataset_config:
            kwargs["name"] = self.dataset_config
        if self.hf_token:
            kwargs["token"] = self.hf_token
        if self.trust_remote_code is not None:
            kwargs["trust_remote_code"] = bool(self.trust_remote_code)

        # Allow advanced per-source HF datasets options without changing loader code,
        # but do not allow them to silently override the core source identity fields.
        reserved = {"path", "name", "split", "streaming", "token"}
        bad_keys = sorted(k for k in self.load_dataset_kwargs if k in reserved)
        if bad_keys:
            raise ValueError(
                f"hf_raw_text source {self.name!r} load_dataset_kwargs may not override reserved keys {bad_keys}. "
                "Use the top-level source fields dataset/dataset_config/split/streaming instead."
            )
        kwargs.update(self.load_dataset_kwargs)

        t0 = _now()
        self.logger.info(
            "hf_raw_text heartbeat source=%s action=load_dataset_start dataset=%s config=%s split=%s epoch=%d trust_remote_code=%s extra_kwargs=%s",
            self.name,
            self.dataset,
            self.dataset_config,
            self.split,
            self.source_epoch,
            kwargs.get("trust_remote_code", None),
            sorted(self.load_dataset_kwargs.keys()),
        )
        try:
            ds = load_dataset(**kwargs)
        except ValueError as e:
            msg = str(e)
            if "trust_remote_code" in msg:
                raise RuntimeError(
                    f"hf_raw_text source {self.name!r} failed to load dataset {self.dataset!r}. "
                    "This dataset appears to require custom Hugging Face dataset code. "
                    "If you trust this dataset repository, add trust_remote_code=true "
                    "to this source config. Original error: " + msg
                ) from e
            raise RuntimeError(
                f"hf_raw_text source {self.name!r} failed in load_dataset(dataset={self.dataset!r}, "
                f"config={self.dataset_config!r}, split={self.split!r}). Original error: {msg}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"hf_raw_text source {self.name!r} failed in load_dataset(dataset={self.dataset!r}, "
                f"config={self.dataset_config!r}, split={self.split!r}). Original error: {e!r}"
            ) from e
        elapsed = _now() - t0
        self.load_dataset_seconds_total += elapsed
        self.load_dataset_calls += 1
        self.logger.info(
            "hf_raw_text heartbeat source=%s action=load_dataset_done elapsed=%.1fs calls=%d",
            self.name,
            elapsed,
            self.load_dataset_calls,
        )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed + self.source_epoch, buffer_size=self.shuffle_buffer)
        self._dataset_obj = ds
        self._iterator = iter(ds)
        self.last_example_index_read = -1

    def _restart_after_exhaustion(self) -> None:
        old_epoch = self.source_epoch
        self.source_epoch += 1
        self._iterator = None
        self._dataset_obj = None
        self.logger.info("hf_raw_text source=%s exhausted epoch=%d; repeating epoch=%d", self.name, old_epoch, self.source_epoch)
        self._open_iterator()

    def _read_next_raw_batch_and_submit(self) -> bool:
        if self._closed:
            return False
        if self._iterator is None:
            self._open_iterator()
        texts: List[str] = []
        raw_bytes = 0
        first_idx: Optional[int] = None
        while len(texts) < self.read_batch_examples and raw_bytes < self.read_batch_bytes:
            try:
                ex = next(self._iterator)  # type: ignore[arg-type]
                self.last_example_index_read += 1
                idx = self.last_example_index_read
            except StopIteration:
                if texts:
                    break
                self._restart_after_exhaustion()
                continue
            text = ex.get(self.text_column) if isinstance(ex, dict) else None
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if not text:
                continue
            b = len(text.encode("utf-8", errors="ignore"))
            if self.max_raw_doc_bytes is not None and b > self.max_raw_doc_bytes:
                continue
            if first_idx is None:
                first_idx = idx
            texts.append(text)
            raw_bytes += b
            self.examples_read += 1
            self.raw_bytes_read += b
        if not texts:
            return False
        ok = self.raw_manager.submit(RawTextBatch(
            texts=texts,
            raw_bytes=raw_bytes,
            first_example_index=first_idx if first_idx is not None else self.last_example_index_read,
            last_example_index=self.last_example_index_read,
            source_name=self.name,
            source_epoch=self.source_epoch,
            tokenizer_encoding=self.tokenizer_encoding,
            append_eot=self.append_eot,
        ), raw_bytes)
        if ok:
            self.raw_batches += 1
            self._inflight_batches += 1
        return ok

    def next_token_chunk(self, target_tokens: int) -> TokenChunk:
        self.start()
        arrays: List[List[int]] = []
        raw_bytes = 0
        examples = 0
        first_idx: Optional[int] = None
        last_idx = -1
        first_epoch: Optional[int] = None
        last_epoch: Optional[int] = None
        token_count = 0
        chunk_start_time = _now()
        last_heartbeat = chunk_start_time
        queue_empty_count = 0
        while token_count < target_tokens or not arrays:
            if self.errors and self.fail_fast:
                raise RuntimeError(f"Raw source {self.name} has errors: {self.errors[-3:]}")
            if self.raw_manager.errors and self.fail_fast:
                raise RuntimeError(f"RawTextManager has errors: {self.raw_manager.errors[-3:]}")
            while self._inflight_batches < self.prefetch_batches and self.token_queue.qsize() == 0:
                if not self._read_next_raw_batch_and_submit():
                    break
            try:
                item = self.raw_manager.get_tokenized(self.name, timeout=0.25)
            except queue.Empty:
                queue_empty_count += 1
                # Keep the pipeline fed for the selected source only.
                self._read_next_raw_batch_and_submit()
                now = _now()
                if now - last_heartbeat >= self.heartbeat_interval_seconds:
                    self.logger.info(
                        "hf_raw_text heartbeat source=%s action=waiting_for_tokens target=%d collected=%d elapsed=%.1fs epoch=%d examples_read=%d examples_emitted=%d raw_batches=%d token_batches=%d inflight=%d source_token_q=%d raw_q_items=%d raw_q_gib=%.2f manager_errors=%d queue_empty=%d",
                        self.name, target_tokens, token_count, now - chunk_start_time, self.source_epoch,
                        self.examples_read, self.examples_emitted, self.raw_batches, self.token_batches,
                        self._inflight_batches, self.token_queue.qsize(), self.raw_manager.raw_queue.current_items,
                        self.raw_manager.raw_queue.current_bytes / (1024**3), len(self.raw_manager.errors), queue_empty_count,
                    )
                    last_heartbeat = now
                continue
            if item is STOP:
                continue
            assert isinstance(item, TokenizedBatch)
            self._inflight_batches = max(0, self._inflight_batches - 1)
            self.token_batches += 1
            self.examples_tokenized += item.examples
            arrays.extend(item.token_arrays)
            raw_bytes += item.raw_bytes
            examples += item.examples
            token_count += sum(len(a) for a in item.token_arrays)
            first_idx = item.first_example_index if first_idx is None else min(first_idx, item.first_example_index)
            last_idx = max(last_idx, item.last_example_index)
            first_epoch = item.source_epoch if first_epoch is None else min(first_epoch, item.source_epoch)
            last_epoch = item.source_epoch if last_epoch is None else max(last_epoch, item.source_epoch)
            now = _now()
            if now - last_heartbeat >= self.heartbeat_interval_seconds:
                self.logger.info(
                    "hf_raw_text heartbeat source=%s action=collecting_tokens target=%d collected=%d pct=%.2f elapsed=%.1fs epoch=%d examples_read=%d examples_emitted=%d raw_batches=%d token_batches=%d inflight=%d source_token_q=%d raw_q_items=%d raw_q_gib=%.2f",
                    self.name, target_tokens, token_count, 100.0 * min(1.0, token_count / max(1, target_tokens)), now - chunk_start_time,
                    self.source_epoch, self.examples_read, self.examples_emitted, self.raw_batches, self.token_batches,
                    self._inflight_batches, self.token_queue.qsize(), self.raw_manager.raw_queue.current_items,
                    self.raw_manager.raw_queue.current_bytes / (1024**3),
                )
                last_heartbeat = now

        flat: List[int] = []
        for a in arrays:
            flat.extend(a)
        token_np = np.asarray(flat, dtype=self.dtype)
        self.tokens_emitted += int(token_np.size)
        self.examples_emitted += examples
        self.raw_bytes_emitted += raw_bytes
        self.last_example_index_emitted = max(self.last_example_index_emitted, last_idx)
        self.logger.info(
            "hf_raw_text heartbeat source=%s action=chunk_ready tokens=%d target=%d elapsed=%.1fs examples=%d raw_bytes=%d epoch_start=%s epoch_end=%s",
            self.name, int(token_np.size), target_tokens, _now() - chunk_start_time, examples, raw_bytes, first_epoch, last_epoch,
        )
        return TokenChunk(
            tokens=token_np,
            source_name=self.name,
            backend=self.backend,
            metadata={
                "dataset": self.dataset,
                "dataset_config": self.dataset_config,
                "split": self.split,
                "text_column": self.text_column,
                "tokenizer": self.tokenizer_encoding,
                "append_eot": self.append_eot,
                "examples_start": first_idx,
                "examples_end": last_idx,
                "examples": examples,
                "raw_bytes": raw_bytes,
                "source_epoch_start": first_epoch,
                "source_epoch_end": last_epoch,
                "exhaustion_policy": "repeat",
                "state_exactness": "best_effort_stream_position",
            },
        )

    def close(self) -> None:
        self._closed = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "weight": self.weight,
            "debt_tokens": self.debt_tokens,
            "tokens_emitted": self.tokens_emitted,
            "dataset": self.dataset,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "text_column": self.text_column,
            "trust_remote_code": self.trust_remote_code,
            "load_dataset_kwargs": dict(self.load_dataset_kwargs),
            "tokenizer_encoding": self.tokenizer_encoding,
            "append_eot": self.append_eot,
            "shuffle_buffer": self.shuffle_buffer,
            "seed": self.seed,
            "source_epoch": self.source_epoch,
            "exhaustion_policy": "repeat",
            "examples_read": self.examples_read,
            "examples_tokenized": self.examples_tokenized,
            "examples_emitted": self.examples_emitted,
            "raw_bytes_read": self.raw_bytes_read,
            "raw_bytes_emitted": self.raw_bytes_emitted,
            "last_example_index_read": self.last_example_index_read,
            "last_example_index_emitted": self.last_example_index_emitted,
            "load_dataset_seconds_total": self.load_dataset_seconds_total,
            "load_dataset_calls": self.load_dataset_calls,
            "inflight_batches": self._inflight_batches,
            "prefetch_batches": self.prefetch_batches,
            "errors": list(self.errors[-20:]),
            "state_exactness": "best_effort_stream_position",
            "global_shared_raw_resources": True,
            "note": "Generated train.bin/reserve.bin are exact if present. Raw stream replay is best-effort if pools are lost.",
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.debt_tokens = float(state.get("debt_tokens", self.debt_tokens))
        self.tokens_emitted = int(state.get("tokens_emitted", 0))
        self.source_epoch = int(state.get("source_epoch", 0))
        self.examples_read = int(state.get("examples_read", 0))
        self.examples_tokenized = int(state.get("examples_tokenized", 0))
        self.examples_emitted = int(state.get("examples_emitted", 0))
        self.raw_bytes_read = int(state.get("raw_bytes_read", 0))
        self.raw_bytes_emitted = int(state.get("raw_bytes_emitted", 0))
        self.last_example_index_read = int(state.get("last_example_index_read", -1))
        self.last_example_index_emitted = int(state.get("last_example_index_emitted", -1))
        self.trust_remote_code = state.get("trust_remote_code", self.trust_remote_code)
        self.load_dataset_kwargs = dict(state.get("load_dataset_kwargs", self.load_dataset_kwargs))
        self.load_dataset_seconds_total = float(state.get("load_dataset_seconds_total", 0.0))
        self.load_dataset_calls = int(state.get("load_dataset_calls", 0))


class BatchHelper:
    STATE_SCHEMA_VERSION = 2

    def __init__(
        self,
        block_size: int,
        batch_size: int,
        data_dir: str | os.PathLike[str],
        device: str | torch.device,
        device_type: str,
        *,
        dataset_config_name: Optional[str] = None,
        config_dir: Optional[str | os.PathLike[str]] = None,
        hf_token: Optional[str] = None,
        dtype: str = "uint16",
        pool_size_bytes: int = 2 * 1024**3,
        pool_reuse_factor: float = 1.0,
        emergency_reuse_factor: float = 2.0,
        pool_chunk_tokens: int = 4_194_304,
        async_pool_builder: bool = True,
        start_reserve_builder: bool = True,
        block_on_missing_train: bool = True,
        eager_initial_pool_build: bool = False,
        missing_pool_policy: str = "rebuild",
        source_cache_dir: Optional[str | os.PathLike[str]] = None,
        source_cache_max_bytes: int = DEFAULT_SOURCE_CACHE_MAX_BYTES,
        allow_single_oversized_source: bool = True,
        max_single_source_download_bytes: Optional[int] = None,
        raw_buffer_bytes: Optional[str | int] = None,
        raw_queue_max_batches: int = 256,
        raw_token_queue_max_batches: int = 64,
        raw_tokenizer_workers: str | int = "auto",
        heartbeat_interval_seconds: float = 30.0,
        data_log_path: Optional[str | os.PathLike[str]] = None,
        data_log_max_bytes: int = 10_000_000,
        data_log_backup_count: int = 5,
        data_log_to_console: bool = False,
        disable_hf_progress: bool = True,
        hash_pools: bool = False,
        restore_torch_rng: bool = False,
        seed: Optional[int] = None,
        save_hf_token_in_state: bool = False,
        **legacy_kwargs: Any,
    ):
        if missing_pool_policy not in {"fail", "promote", "rebuild"}:
            raise ValueError("missing_pool_policy must be one of fail/promote/rebuild")
        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.device_type = device_type
        self.dtype = np.dtype(dtype)
        self.pool_size_bytes = int(pool_size_bytes)
        self.pool_reuse_factor = float(pool_reuse_factor)
        self.emergency_reuse_factor = float(emergency_reuse_factor)
        self.pool_chunk_tokens = int(pool_chunk_tokens)
        self.async_pool_builder = bool(async_pool_builder)
        self.start_reserve_builder = bool(start_reserve_builder)
        self.block_on_missing_train = bool(block_on_missing_train)
        self.eager_initial_pool_build = bool(eager_initial_pool_build)
        self.missing_pool_policy = missing_pool_policy
        self.hf_token = hf_token
        self.hash_pools = bool(hash_pools)
        self.restore_torch_rng = bool(restore_torch_rng)
        self.seed = seed
        self.save_hf_token_in_state = bool(save_hf_token_in_state)
        self.class_dir = self._default_class_dir()
        self.config_dir = Path(config_dir) if config_dir is not None else self.class_dir
        self.source_cache_dir = Path(source_cache_dir) if source_cache_dir is not None else self.class_dir / ".batch_helper_source_cache"
        self.source_cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_cache_max_bytes = int(source_cache_max_bytes)
        self.allow_single_oversized_source = bool(allow_single_oversized_source)
        self.max_single_source_download_bytes = max_single_source_download_bytes
        self.raw_buffer_bytes = _parse_size(raw_buffer_bytes, DEFAULT_RAW_BUFFER_MAX_BYTES)
        self.raw_queue_max_batches = int(raw_queue_max_batches)
        self.raw_token_queue_max_batches = int(raw_token_queue_max_batches)
        self.raw_tokenizer_workers = _auto_tokenizer_workers() if str(raw_tokenizer_workers) == "auto" else max(1, int(raw_tokenizer_workers))
        self.heartbeat_interval_seconds = max(1.0, float(heartbeat_interval_seconds))
        self.raw_manager: Optional[RawTextManager] = None
        self.data_log_path = Path(data_log_path) if data_log_path is not None else self.class_dir / "pool_builder.log"
        self.data_log_max_bytes = int(data_log_max_bytes)
        self.data_log_backup_count = int(data_log_backup_count)
        self.data_log_to_console = bool(data_log_to_console)
        self.disable_hf_progress = bool(disable_hf_progress)
        self.logger = self._configure_logger()
        self._configure_external_output()
        if legacy_kwargs:
            self._warn(f"Ignoring unsupported legacy kwargs: {sorted(legacy_kwargs)}")

        self.train_path = self.data_dir / "train.bin"
        self.val_path = self.data_dir / "val.bin"
        self.reserve_path = self.data_dir / "reserve.bin"
        self.reserve_tmp_path = self.data_dir / "reserve.tmp"
        self.train_meta_path = self.data_dir / "train.meta.json"
        self.reserve_meta_path = self.data_dir / "reserve.meta.json"
        self.state_path = self.data_dir / "pool_state.json"

        self.dataset_config_name = dataset_config_name
        self.dataset_config: Optional[Dict[str, Any]] = None
        self.dataset_config_sha256: Optional[str] = None
        self.sources: List[SourceBackend] = []
        self.pooled_mode = dataset_config_name is not None
        self.rng = random.Random(seed)
        self.pool_generation = 0
        self.active_pool_tokens_used = 0
        self._active_meta: Optional[Dict[str, Any]] = None
        self._reserve_meta: Optional[Dict[str, Any]] = None
        self._warned_emergency_reuse = False
        self._pool_lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._builder_future: Optional[Future] = None
        self._last_builder_error: Optional[str] = None
        self._committed_source_states: List[Dict[str, Any]] = []
        self._committed_raw_manager_state: Optional[Dict[str, Any]] = None

        if self.pooled_mode:
            self.dataset_config, self.dataset_config_sha256 = self._load_dataset_config(dataset_config_name)
            self._configure_raw_backend_from_config(self.dataset_config)
            self.raw_manager = RawTextManager(
                dtype=self.dtype,
                logger=self.logger,
                raw_buffer_bytes=self.raw_buffer_bytes,
                raw_queue_max_batches=self.raw_queue_max_batches,
                token_queue_max_batches=self.raw_token_queue_max_batches,
                tokenizer_workers=self.raw_tokenizer_workers,
            )
            self.sources = self._build_sources(self.dataset_config)
            self._normalize_source_weights()
            self._cleanup_abandoned_tmp_files()
            self._load_existing_pool_metadata()
            self._commit_current_source_state(reason="initialization")
            if self.eager_initial_pool_build:
                self._ensure_train_pool_ready_for_batch(reason="initialization")
            if self.start_reserve_builder:
                self._start_reserve_builder_if_needed()

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            if self.pooled_mode:
                self._ensure_train_pool_ready_for_batch(reason="pre_batch")
                self._maybe_swap_before_train_batch()
                self._ensure_train_pool_ready_for_batch(reason="post_swap")
                x, y = self._sample_train_batch()
                self.active_pool_tokens_used += self.batch_size * self.block_size
                if self.start_reserve_builder:
                    self._start_reserve_builder_if_needed()
                return self._move_batch_to_device(x, y)
            return self._get_static_local_batch("train")
        if split == "val":
            return self._get_static_local_batch("val")
        raise ValueError(f"Unknown split: {split}")

    def _configure_raw_backend_from_config(self, config: Dict[str, Any]) -> None:
        rb = config.get("raw_backend") or config.get("raw_text_backend") or {}
        if not isinstance(rb, dict):
            raise ValueError("raw_backend must be an object when provided")
        if "global_raw_buffer_bytes" in rb:
            self.raw_buffer_bytes = _parse_size(rb.get("global_raw_buffer_bytes"), self.raw_buffer_bytes)
        if "raw_buffer_bytes" in rb:
            self.raw_buffer_bytes = _parse_size(rb.get("raw_buffer_bytes"), self.raw_buffer_bytes)
        if "raw_queue_max_batches" in rb:
            self.raw_queue_max_batches = int(rb["raw_queue_max_batches"])
        if "global_token_queue_max_batches" in rb:
            self.raw_token_queue_max_batches = int(rb["global_token_queue_max_batches"])
        if "token_queue_max_batches" in rb:
            self.raw_token_queue_max_batches = int(rb["token_queue_max_batches"])
        if "tokenizer_workers" in rb:
            tw = rb["tokenizer_workers"]
            self.raw_tokenizer_workers = _auto_tokenizer_workers() if str(tw) == "auto" else max(1, int(tw))
        if "heartbeat_interval_seconds" in rb:
            self.heartbeat_interval_seconds = max(1.0, float(rb["heartbeat_interval_seconds"]))
        elif "heartbeat_interval" in rb:
            self.heartbeat_interval_seconds = max(1.0, float(rb["heartbeat_interval"]))
        self.logger.info(
            "Configured global raw backend raw_buffer_bytes=%d raw_queue_max_batches=%d token_queue_max_batches=%d tokenizer_workers=%d heartbeat_interval=%.1fs",
            self.raw_buffer_bytes,
            self.raw_queue_max_batches,
            self.raw_token_queue_max_batches,
            self.raw_tokenizer_workers,
            self.heartbeat_interval_seconds,
        )

    def _build_sources(self, config: Dict[str, Any]) -> List[SourceBackend]:
        entries = config.get("sources") or config.get("repos") or config.get("datasets")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Dataset config must contain a non-empty sources/repos/datasets list")
        sources: List[SourceBackend] = []
        for idx, raw in enumerate(entries):
            cfg = dict(raw)
            backend = cfg.get("backend", "hf_bin")
            cfg.setdefault("name", cfg.get("repo_id") or cfg.get("dataset") or f"source_{idx}")
            cfg.setdefault("heartbeat_interval_seconds", self.heartbeat_interval_seconds)
            if backend == "hf_bin":
                sources.append(HFBinSource(
                    cfg,
                    dtype=self.dtype,
                    logger=self.logger,
                    hf_token=self.hf_token,
                    source_cache_dir=self.source_cache_dir,
                    source_cache_max_bytes=self.source_cache_max_bytes,
                    allow_single_oversized_source=self.allow_single_oversized_source,
                    max_single_source_download_bytes=self.max_single_source_download_bytes,
                    block_size=self.block_size,
                ))
            elif backend == "hf_raw_text":
                if self.raw_manager is None:
                    self.raw_manager = RawTextManager(
                        dtype=self.dtype,
                        logger=self.logger,
                        raw_buffer_bytes=self.raw_buffer_bytes,
                        raw_queue_max_batches=self.raw_queue_max_batches,
                        token_queue_max_batches=self.raw_token_queue_max_batches,
                        tokenizer_workers=self.raw_tokenizer_workers,
                    )
                sources.append(HFRawTextSource(cfg, dtype=self.dtype, logger=self.logger, hf_token=self.hf_token, raw_manager=self.raw_manager))
            else:
                raise ValueError(f"Unsupported source backend: {backend}")
        return sources

    def _normalize_source_weights(self) -> None:
        total = sum(max(0.0, s.weight) for s in self.sources)
        if total <= 0:
            raise ValueError("Sum of source weights must be > 0")
        for s in self.sources:
            s.weight = float(s.weight) / total

    def _ensure_train_pool_ready_for_batch(self, *, reason: str) -> None:
        if not self.pooled_mode:
            return
        if self.train_path.exists():
            if self._active_meta is None:
                self._active_meta = self._meta_for_existing_pool(self.train_path, self.pool_generation)
            return
        if not self.block_on_missing_train:
            raise FileNotFoundError(f"Missing train.bin: {self.train_path}")
        self._wait_for_running_builder_if_any(reason=reason)
        if self.train_path.exists():
            return
        if self.reserve_path.exists():
            self._promote_reserve_to_train()
            return
        if self.missing_pool_policy in {"fail", "promote"}:
            raise FileNotFoundError(f"Missing train.bin and reserve.bin: {self.data_dir}")
        self._build_reserve_pool_sync()
        self._promote_reserve_to_train()

    def _maybe_swap_before_train_batch(self) -> None:
        self._poll_builder_future(raise_on_error=False)
        if self.active_pool_tokens_used < self._active_pool_token_budget():
            return
        if self.reserve_path.exists():
            self._promote_reserve_to_train()
            self._warned_emergency_reuse = False
            return
        self._start_reserve_builder_if_needed()
        if self.active_pool_tokens_used < self._active_pool_emergency_budget():
            if not self._warned_emergency_reuse:
                self._warn("Reserve not ready; continuing under emergency reuse budget")
                self._warned_emergency_reuse = True
            return
        self._warn("Reserve not ready after emergency budget; continuing to avoid downtime")

    def _start_reserve_builder_if_needed(self) -> None:
        if not self.pooled_mode:
            return
        self._poll_builder_future(raise_on_error=False)
        with self._pool_lock:
            if self.reserve_path.exists():
                return
            if self._builder_future is not None and not self._builder_future.done():
                return
            if self.async_pool_builder:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="BatchHelperPoolBuilder")
                self._builder_future = self._executor.submit(self._build_reserve_pool_sync)
            else:
                self._build_reserve_pool_sync()

    def _wait_for_running_builder_if_any(self, *, reason: str) -> None:
        f = self._builder_future
        if f is None or f.done():
            self._poll_builder_future(raise_on_error=True)
            return
        self.logger.info("Waiting for reserve builder reason=%s", reason)
        try:
            f.result()
        finally:
            if self._builder_future is f:
                self._builder_future = None

    def _poll_builder_future(self, *, raise_on_error: bool) -> None:
        f = self._builder_future
        if f is None or not f.done():
            return
        self._builder_future = None
        try:
            f.result()
        except Exception as e:
            self._last_builder_error = repr(e)
            self.logger.exception("Reserve builder failed")
            if raise_on_error:
                raise
            self._warn(f"Reserve builder failed: {e!r}")

    def _snapshot_source_state(self) -> Dict[str, Any]:
        return {
            "sources": [s.state_dict() for s in self.sources],
            "raw_manager": self.raw_manager.state_dict() if self.raw_manager is not None else None,
        }

    def _commit_current_source_state(self, *, reason: str) -> None:
        snap = self._snapshot_source_state()
        self._committed_source_states = [dict(s) for s in snap["sources"]]
        self._committed_raw_manager_state = dict(snap["raw_manager"]) if snap.get("raw_manager") is not None else None
        self.logger.info("Committed source state reason=%s sources=%d", reason, len(self._committed_source_states))

    def _restore_source_state(self, snapshot: Dict[str, Any], *, reason: str) -> None:
        by_name = {s.name: s for s in self.sources}
        for ss in snapshot.get("sources", []):
            name = ss.get("name")
            if name in by_name:
                by_name[name].load_state_dict(ss)
        # Any in-flight raw batches/token batches belong to the failed/discarded reserve.tmp.
        # Drop them by recreating the shared raw manager and rebinding raw sources.
        self._reset_raw_manager_runtime(reason=reason)
        self.logger.info("Restored pre-build source state reason=%s", reason)

    def _reset_raw_manager_runtime(self, *, reason: str) -> None:
        if self.raw_manager is None:
            return
        try:
            self.raw_manager.close()
        except Exception:
            pass
        self.raw_manager = RawTextManager(
            dtype=self.dtype,
            logger=self.logger,
            raw_buffer_bytes=self.raw_buffer_bytes,
            raw_queue_max_batches=self.raw_queue_max_batches,
            token_queue_max_batches=self.raw_token_queue_max_batches,
            tokenizer_workers=self.raw_tokenizer_workers,
        )
        for src in self.sources:
            if isinstance(src, HFRawTextSource):
                src.raw_manager = self.raw_manager
                src.token_queue = self.raw_manager.register_source(src.name)
                src._inflight_batches = 0
                src._started = False
                src._closed = False
                src._iterator = None
                src._dataset_obj = None
        self.logger.info("Reset shared raw manager runtime reason=%s", reason)

    def _build_reserve_pool_sync(self) -> None:
        if not self.pooled_mode:
            return
        with self._pool_lock:
            if self.reserve_path.exists():
                return
            generation = self.pool_generation + 1
            prebuild_snapshot = self._snapshot_source_state()

        self.reserve_tmp_path.unlink(missing_ok=True)
        tmp_meta_path = self.reserve_tmp_path.with_suffix(".tmp.meta.json")
        tmp_meta_path.unlink(missing_ok=True)
        target_tokens = self.pool_size_bytes // self.dtype.itemsize
        total_tokens = 0
        segments: List[Dict[str, Any]] = []
        actual_by_source = {s.name: 0 for s in self.sources}
        build_start_time = _now()
        last_build_heartbeat = build_start_time
        self.logger.info("Building reserve generation=%d target_tokens=%d sources=%d heartbeat_interval=%.1fs", generation, target_tokens, len(self.sources), self.heartbeat_interval_seconds)

        try:
            with self.reserve_tmp_path.open("wb") as out:
                while total_tokens + self.block_size + 1 <= target_tokens:
                    for src in self.sources:
                        src.debt_tokens += self.pool_chunk_tokens * src.weight
                    src = max(self.sources, key=lambda s: s.debt_tokens)
                    remaining = target_tokens - total_tokens
                    target = min(self.pool_chunk_tokens, remaining)
                    chunk = src.next_token_chunk(target)
                    if chunk.tokens.size <= self.block_size:
                        continue
                    start = total_tokens
                    chunk.tokens.tofile(out)
                    n = int(chunk.tokens.size)
                    total_tokens += n
                    src.debt_tokens -= n
                    actual_by_source[src.name] = actual_by_source.get(src.name, 0) + n
                    segments.append({
                        "start_token": start,
                        "length_tokens": n,
                        "source_name": chunk.source_name,
                        "backend": chunk.backend,
                        "metadata": chunk.metadata,
                    })
                    now = _now()
                    if now - last_build_heartbeat >= self.heartbeat_interval_seconds:
                        elapsed = now - build_start_time
                        self.logger.info(
                            "reserve heartbeat generation=%d tokens=%d/%d pct=%.2f elapsed=%.1fs rate_tokens_per_s=%.0f current_source=%s backend=%s segments=%d actual_by_source=%s",
                            generation, total_tokens, target_tokens, 100.0 * total_tokens / max(1, target_tokens), elapsed,
                            total_tokens / max(1e-9, elapsed), chunk.source_name, chunk.backend, len(segments), actual_by_source,
                        )
                        last_build_heartbeat = now
                out.flush(); os.fsync(out.fileno())

            if total_tokens <= self.block_size:
                raise RuntimeError("Reserve build produced too few tokens")

            meta = {
                "schema_version": self.STATE_SCHEMA_VERSION,
                "generation": generation,
                "created_at_unix": _now(),
                "dtype": str(self.dtype),
                "token_count": total_tokens,
                "bytes": self.reserve_tmp_path.stat().st_size,
                "block_size": self.block_size,
                "dataset_config_name": self.dataset_config_name,
                "dataset_config_sha256": self.dataset_config_sha256,
                "segments": segments,
                "actual_tokens_by_source": actual_by_source,
                "source_states_after_build": [s.state_dict() for s in self.sources],
                "sha256": self._file_sha256(self.reserve_tmp_path) if self.hash_pools else None,
            }
            self._atomic_write_json(tmp_meta_path, meta)

            with self._pool_lock:
                if self.reserve_path.exists():
                    self.reserve_tmp_path.unlink(missing_ok=True)
                    tmp_meta_path.unlink(missing_ok=True)
                    self._restore_source_state(prebuild_snapshot, reason="reserve_discarded_existing_reserve")
                    return
                self.reserve_tmp_path.replace(self.reserve_path)
                tmp_meta_path.replace(self.reserve_meta_path)
                self._reserve_meta = meta
                self._commit_current_source_state(reason="reserve_completed")
                self.logger.info("Reserve ready generation=%d tokens=%d", generation, total_tokens)
        except Exception:
            self.reserve_tmp_path.unlink(missing_ok=True)
            tmp_meta_path.unlink(missing_ok=True)
            self._restore_source_state(prebuild_snapshot, reason="reserve_build_failed")
            raise

    def _promote_reserve_to_train(self) -> None:
        if not self.reserve_path.exists():
            raise FileNotFoundError(f"Missing reserve.bin: {self.reserve_path}")
        meta = self._read_json_if_exists(self.reserve_meta_path) or self._meta_for_existing_pool(self.reserve_path, self.pool_generation + 1)
        stamp = f"{int(_now())}.{os.getpid()}"
        old_train = self.data_dir / f"train.retired.{stamp}.bin"
        old_meta = self.data_dir / f"train.retired.{stamp}.meta.json"
        if self.train_path.exists(): self.train_path.replace(old_train)
        if self.train_meta_path.exists(): self.train_meta_path.replace(old_meta)
        self.reserve_path.replace(self.train_path)
        if self.reserve_meta_path.exists(): self.reserve_meta_path.replace(self.train_meta_path)
        else: self._atomic_write_json(self.train_meta_path, meta)
        self._active_meta = meta
        self._reserve_meta = None
        self.pool_generation = int(meta.get("generation", self.pool_generation + 1))
        self.active_pool_tokens_used = 0
        for p in (old_train, old_meta):
            try:
                if p.exists(): p.unlink()
            except OSError: pass

    def _sample_train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.train_path.exists():
            self._ensure_train_pool_ready_for_batch(reason="sample_missing_train")
        data = np.memmap(self.train_path, dtype=self.dtype, mode="r")
        try:
            meta = self._active_meta or self._read_json_if_exists(self.train_meta_path)
            segments = [] if meta is None else list(meta.get("segments", []))
            if segments:
                return self._sample_from_segmented_memmap(data, segments)
            return self._sample_from_memmap(data)
        finally:
            del data

    def _get_static_local_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.train_path if split == "train" else self.val_path
        if not path.exists():
            raise FileNotFoundError(f"Missing {split}: {path}")
        data = np.memmap(path, dtype=self.dtype, mode="r")
        try:
            x, y = self._sample_from_memmap(data)
        finally:
            del data
        return self._move_batch_to_device(x, y)

    def _sample_from_memmap(self, data: np.memmap) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(data) <= self.block_size:
            raise ValueError(f"Data too small for block_size: {len(data)}")
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(np.asarray(data[int(i):int(i)+self.block_size], dtype=np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(np.asarray(data[int(i)+1:int(i)+1+self.block_size], dtype=np.int64)) for i in ix])
        return x, y

    def _sample_from_segmented_memmap(self, data: np.memmap, segments: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        valid, counts = [], []
        for seg in segments:
            start, length = int(seg["start_token"]), int(seg["length_tokens"])
            c = length - self.block_size
            if c > 0:
                valid.append((start, length)); counts.append(c)
        if not valid:
            raise ValueError("No valid segments for sampling")
        cumsum = np.cumsum(np.asarray(counts, dtype=np.int64))
        draws = torch.randint(int(cumsum[-1]), (self.batch_size,), dtype=torch.long).numpy()
        starts = []
        for d in draws:
            si = int(np.searchsorted(cumsum, d, side="right"))
            prev = int(cumsum[si-1]) if si > 0 else 0
            starts.append(valid[si][0] + int(d) - prev)
        x = torch.stack([torch.from_numpy(np.asarray(data[i:i+self.block_size], dtype=np.int64)) for i in starts])
        y = torch.stack([torch.from_numpy(np.asarray(data[i+1:i+1+self.block_size], dtype=np.int64)) for i in starts])
        return x, y

    def _move_batch_to_device(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.device_type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def constructor_args_dict(self) -> Dict[str, Any]:
        return {
            "block_size": self.block_size, "batch_size": self.batch_size, "data_dir": str(self.data_dir),
            "device": str(self.device), "device_type": self.device_type, "dataset_config_name": self.dataset_config_name,
            "config_dir": str(self.config_dir), "hf_token": self.hf_token if self.save_hf_token_in_state else None,
            "dtype": str(self.dtype), "pool_size_bytes": self.pool_size_bytes, "pool_reuse_factor": self.pool_reuse_factor,
            "emergency_reuse_factor": self.emergency_reuse_factor, "pool_chunk_tokens": self.pool_chunk_tokens,
            "async_pool_builder": self.async_pool_builder, "start_reserve_builder": self.start_reserve_builder,
            "block_on_missing_train": self.block_on_missing_train, "eager_initial_pool_build": self.eager_initial_pool_build,
            "missing_pool_policy": self.missing_pool_policy, "source_cache_dir": str(self.source_cache_dir),
            "source_cache_max_bytes": self.source_cache_max_bytes, "allow_single_oversized_source": self.allow_single_oversized_source,
            "max_single_source_download_bytes": self.max_single_source_download_bytes,
            "raw_buffer_bytes": self.raw_buffer_bytes,
            "raw_queue_max_batches": self.raw_queue_max_batches,
            "raw_token_queue_max_batches": self.raw_token_queue_max_batches,
            "raw_tokenizer_workers": self.raw_tokenizer_workers,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "data_log_path": str(self.data_log_path),
            "data_log_max_bytes": self.data_log_max_bytes, "data_log_backup_count": self.data_log_backup_count,
            "data_log_to_console": self.data_log_to_console, "disable_hf_progress": self.disable_hf_progress,
            "hash_pools": self.hash_pools, "restore_torch_rng": self.restore_torch_rng, "seed": self.seed,
            "save_hf_token_in_state": self.save_hf_token_in_state,
        }

    def state_dict(self, *, include_torch_rng: Optional[bool] = None, wait_for_builder: bool = False) -> Dict[str, Any]:
        include_torch_rng = self.restore_torch_rng if include_torch_rng is None else include_torch_rng
        if wait_for_builder: self._wait_for_running_builder_if_any(reason="checkpoint")
        else: self._poll_builder_future(raise_on_error=False)
        state = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "mode": "pooled_backend" if self.pooled_mode else "static_local_bin",
            "constructor_args": self.constructor_args_dict(),
            "config": {"dataset_config": self.dataset_config, "dataset_config_sha256": self.dataset_config_sha256},
            "pool_runtime": {"pool_generation": self.pool_generation, "active_pool_tokens_used": self.active_pool_tokens_used},
            "active_pool": self._active_meta if self.train_path.exists() else None,
            "reserve_pool": self._reserve_meta if self.reserve_path.exists() else None,
            "sources": self._committed_source_states if (self._builder_future is not None and not self._builder_future.done()) else [s.state_dict() for s in self.sources],
            "raw_manager": self._committed_raw_manager_state if (self._builder_future is not None and not self._builder_future.done()) else (self.raw_manager.state_dict() if self.raw_manager is not None else None),
            "rng": {"python_random_state": base64.b64encode(pickle.dumps(self.rng.getstate(), protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")},
            "builder": {"running": self._builder_future is not None and not self._builder_future.done(), "last_error": self._last_builder_error, "state_policy": "committed_only_while_builder_running"},
        }
        if include_torch_rng:
            state["rng"]["torch_cpu_rng_state"] = torch.get_rng_state().tolist()
            if torch.cuda.is_available(): state["rng"]["torch_cuda_rng_state_all"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
        return state

    def load_state_dict(self, state: Dict[str, Any], *, strict: bool = False, restore_torch_rng: Optional[bool] = None) -> None:
        if state.get("schema_version") not in {1, 2}:
            raise ValueError(f"Unsupported state schema_version={state.get('schema_version')}")
        rt = state.get("pool_runtime", {})
        self.pool_generation = int(rt.get("pool_generation", self.pool_generation))
        self.active_pool_tokens_used = int(rt.get("active_pool_tokens_used", 0))
        by_name = {s.name: s for s in self.sources}
        for ss in state.get("sources", []):
            if ss.get("name") in by_name:
                by_name[ss["name"]].load_state_dict(ss)
        if state.get("rng", {}).get("python_random_state"):
            self.rng.setstate(pickle.loads(base64.b64decode(state["rng"]["python_random_state"].encode("ascii"))))
        self._load_existing_pool_metadata()
        self._commit_current_source_state(reason="load_state_dict")
        if strict:
            if state.get("active_pool") is not None and not self.train_path.exists():
                raise FileNotFoundError(f"Strict load requires train.bin: {self.train_path}")
            if state.get("reserve_pool") is not None and not self.reserve_path.exists():
                raise FileNotFoundError(f"Strict load requires reserve.bin: {self.reserve_path}")
        if restore_torch_rng or (restore_torch_rng is None and self.restore_torch_rng):
            cpu = state.get("rng", {}).get("torch_cpu_rng_state")
            if cpu is not None: torch.set_rng_state(torch.tensor(cpu, dtype=torch.uint8))
        if self.pooled_mode and self.start_reserve_builder:
            self._start_reserve_builder_if_needed()

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], *, strict: bool = False, restore_torch_rng: Optional[bool] = None, **runtime_overrides: Any) -> "BatchHelper":
        args = dict(state["constructor_args"])
        args.update({k: v for k, v in runtime_overrides.items() if v is not None})
        start = bool(args.get("start_reserve_builder", True)); eager = bool(args.get("eager_initial_pool_build", False))
        args["start_reserve_builder"] = False; args["eager_initial_pool_build"] = False
        helper = cls(**args)
        helper.load_state_dict(state, strict=strict, restore_torch_rng=restore_torch_rng)
        helper.start_reserve_builder = start; helper.eager_initial_pool_build = eager
        if helper.pooled_mode and helper.start_reserve_builder: helper._start_reserve_builder_if_needed()
        return helper

    @staticmethod
    def dataloader_sidecar_path(checkpoint_path: str | os.PathLike[str]) -> Path:
        return Path(f"{checkpoint_path}.dataloader.json")

    def save_checkpoint_sidecar(self, checkpoint_path: str | os.PathLike[str], *, include_torch_rng: Optional[bool] = None, wait_for_builder: bool = False) -> Path:
        p = self.dataloader_sidecar_path(checkpoint_path)
        payload = {"schema_version": self.STATE_SCHEMA_VERSION, "kind": "BatchHelperCheckpointSidecar", "checkpoint_file": Path(checkpoint_path).name, "saved_at_unix": _now(), "items": [self.state_dict(include_torch_rng=include_torch_rng, wait_for_builder=wait_for_builder)]}
        self._atomic_write_json(p, payload)
        return p

    @classmethod
    def load_checkpoint_sidecar_items(cls, checkpoint_path: str | os.PathLike[str]) -> Optional[List[Any]]:
        p = cls.dataloader_sidecar_path(checkpoint_path)
        if not p.exists(): return None
        with p.open("r", encoding="utf-8") as f: payload = json.load(f)
        items = payload.get("items")
        if not isinstance(items, list): raise ValueError(f"Invalid sidecar: {p}")
        return items

    @classmethod
    def from_checkpoint_sidecar(cls, checkpoint_path: str | os.PathLike[str], *, strict: bool = False, restore_torch_rng: Optional[bool] = None, **runtime_overrides: Any) -> Optional["BatchHelper"]:
        items = cls.load_checkpoint_sidecar_items(checkpoint_path)
        if items is None: return None
        if not items: raise ValueError(f"Dataloader sidecar contains no items: {cls.dataloader_sidecar_path(checkpoint_path)}")
        return cls.from_state_dict(items[0], strict=strict, restore_torch_rng=restore_torch_rng, **runtime_overrides)

    def close(self, *, wait: bool = False) -> None:
        for s in self.sources: s.close()
        if self.raw_manager is not None:
            self.raw_manager.close(wait=wait)
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
            self._executor = None; self._builder_future = None

    def _load_dataset_config(self, name: str) -> Tuple[Dict[str, Any], str]:
        p = Path(name)
        if not p.is_absolute(): p = self.config_dir / name
        raw = p.read_bytes()
        return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()

    def _load_existing_pool_metadata(self) -> None:
        self._active_meta = None; self._reserve_meta = None
        if self.train_path.exists(): self._active_meta = self._read_json_if_exists(self.train_meta_path) or self._meta_for_existing_pool(self.train_path, self.pool_generation)
        if self.reserve_path.exists(): self._reserve_meta = self._read_json_if_exists(self.reserve_meta_path) or self._meta_for_existing_pool(self.reserve_path, self.pool_generation + 1)

    def _meta_for_existing_pool(self, path: Path, generation: int) -> Dict[str, Any]:
        size = path.stat().st_size
        return {"schema_version": self.STATE_SCHEMA_VERSION, "generation": int(generation), "path": path.name, "dtype": str(self.dtype), "token_count": size // self.dtype.itemsize, "bytes": size, "block_size": self.block_size, "segments": [], "sha256": self._file_sha256(path) if self.hash_pools else None}

    def _active_pool_token_budget(self) -> int:
        return max(1, int(self._active_token_count() * self.pool_reuse_factor))
    def _active_pool_emergency_budget(self) -> int:
        return max(1, int(self._active_token_count() * self.emergency_reuse_factor))
    def _active_token_count(self) -> int:
        if self._active_meta and self._active_meta.get("token_count"): return int(self._active_meta["token_count"])
        if self.train_path.exists(): return self.train_path.stat().st_size // self.dtype.itemsize
        return self.pool_size_bytes // self.dtype.itemsize

    def _cleanup_abandoned_tmp_files(self) -> None:
        self.reserve_tmp_path.unlink(missing_ok=True); self.reserve_tmp_path.with_suffix(".tmp.meta.json").unlink(missing_ok=True)
    def _read_json_if_exists(self, p: Path) -> Optional[Dict[str, Any]]:
        if not p.exists(): return None
        with p.open("r", encoding="utf-8") as f: return json.load(f)
    def _atomic_write_json(self, p: Path, obj: Dict[str, Any]) -> None:
        tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        tmp.replace(p)
    def _file_sha256(self, p: Path, chunk_size: int = 16*1024*1024) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b: break
                h.update(b)
        return h.hexdigest()
    def _default_class_dir(self) -> Path:
        try: return Path(__file__).resolve().parent
        except NameError: return Path.cwd()
    def _configure_logger(self) -> logging.Logger:
        self.data_log_path.parent.mkdir(parents=True, exist_ok=True)
        lg = logging.getLogger(f"{__name__}.BatchHelper.{id(self)}"); lg.setLevel(logging.INFO); lg.propagate = False
        for h in list(lg.handlers): lg.removeHandler(h); h.close()
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = RotatingFileHandler(self.data_log_path, maxBytes=self.data_log_max_bytes, backupCount=self.data_log_backup_count, encoding="utf-8"); fh.setFormatter(fmt); lg.addHandler(fh)
        if self.data_log_to_console:
            ch = logging.StreamHandler(); ch.setFormatter(fmt); lg.addHandler(ch)
        return lg
    def _configure_external_output(self) -> None:
        if not self.disable_hf_progress: return
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1"); os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        try:
            from huggingface_hub import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception: pass
    def _warn(self, msg: str) -> None:
        self.logger.warning(msg)
        if self.data_log_to_console: warnings.warn(msg, stacklevel=2)
