import importlib.util
import sys
import os
import hashlib

import torch
import numpy as np


def loadModule(module_name, file_path, module_dir):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchHelper:
    def __init__(
        self,
        block_size,
        batch_size,
        data_dir,
        device,
        device_type,
        shards_per_pool=4,
        steps_per_pool=100,
        streaming=False,
        streaming_text_column="text",
        streaming_tokenizer=None,
        streaming_encoding="gpt2",
        streaming_buffer_tokens=1_000_000,
        streaming_append_eot=False,
        streaming_config_name=None,
        streaming_val_split="validation",
        streaming_holdout_validation=True,
        streaming_val_size=0.005,
    ):
        if data_dir is None:
            raise ValueError("data_dir must not be None")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if shards_per_pool <= 0:
            raise ValueError("shards_per_pool must be > 0")
        if steps_per_pool <= 0:
            raise ValueError("steps_per_pool must be > 0")
        if streaming_buffer_tokens <= block_size + 1:
            raise ValueError("streaming_buffer_tokens must be larger than block_size + 1")
        if not (0.0 < streaming_val_size < 1.0):
            raise ValueError("streaming_val_size must be between 0 and 1")

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = device_type

        self.SHARDS_PER_POOL = shards_per_pool
        self.STEPS_PER_POOL = steps_per_pool

        # Persistent train-pool state for local sharded .bin mode.
        self._train_pool_shards = None
        self._train_pool_steps_used = 0

        # Streaming mode configuration.
        self.streaming = streaming
        self.streaming_text_column = streaming_text_column
        self.streaming_encoding = streaming_encoding
        self.streaming_buffer_tokens = streaming_buffer_tokens
        self.streaming_append_eot = streaming_append_eot
        self.streaming_config_name = streaming_config_name
        self.streaming_val_split = streaming_val_split
        self.streaming_holdout_validation = streaming_holdout_validation
        self.streaming_val_size = streaming_val_size

        # Persistent streaming state.
        self._streaming_datasets = {}
        self._streaming_iterators = {}
        self._streaming_buffers = {}

        if self.streaming:
            self.data_dir = self._normalize_hf_dataset_ref(data_dir)

            if streaming_tokenizer is None:
                try:
                    import tiktoken
                except ImportError as e:
                    raise ImportError(
                        "streaming=True requires tiktoken unless you pass "
                        "streaming_tokenizer explicitly. Install with: pip install tiktoken"
                    ) from e

                self.streaming_tokenizer = tiktoken.get_encoding(streaming_encoding)
            else:
                self.streaming_tokenizer = streaming_tokenizer

            if not hasattr(self.streaming_tokenizer, "encode_ordinary"):
                raise TypeError(
                    "streaming_tokenizer must support encode_ordinary(text) "
                    "to match the existing preprocessing pipeline."
                )
        else:
            self.data_dir = data_dir
            self.streaming_tokenizer = streaming_tokenizer

    # -------------------------------------------------------------------------
    # Public batch API
    # -------------------------------------------------------------------------

    def get_batch(self, split):
        if self.streaming:
            return self._get_streaming_batch(split)

        return self._get_local_batch(split)

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _move_batch_to_device(self, x, y):
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y

    # -------------------------------------------------------------------------
    # Local .bin / memmap path
    # -------------------------------------------------------------------------

    def _list_train_shards(self):
        """Return all shard files matching trainX.bin, sorted by X."""
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Invalid data_dir: {self.data_dir}")

        shard_files = []
        for fname in os.listdir(self.data_dir):
            if fname.startswith("train") and fname.endswith(".bin") and fname != "train.bin":
                shard_id = fname[len("train"):-len(".bin")]
                if shard_id.isdigit():
                    shard_files.append((int(shard_id), fname))

        shard_files.sort(key=lambda x: x[0])
        return [fname for _, fname in shard_files]

    def _choose_new_train_pool(self):
        """Choose a new active shard pool."""
        shard_files = self._list_train_shards()
        if not shard_files:
            raise FileNotFoundError(
                f"No train.bin or trainX.bin shards found in {self.data_dir}"
            )

        num_shards = min(self.SHARDS_PER_POOL, len(shard_files))
        chosen_idx = torch.randperm(len(shard_files))[:num_shards].tolist()
        self._train_pool_shards = [shard_files[i] for i in chosen_idx]
        self._train_pool_steps_used = 0

    def _sample_from_memmap(self, data, n_samples):
        """Sample n_samples (x, y) pairs from a memmap-backed token array."""
        if len(data) <= self.block_size:
            raise ValueError(
                f"Data file length {len(data)} is too small for block_size={self.block_size}"
            )

        ix = torch.randint(len(data) - self.block_size, (n_samples,))
        x = torch.stack([
            torch.from_numpy((data[i:i + self.block_size]).astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy((data[i + 1:i + 1 + self.block_size]).astype(np.int64))
            for i in ix
        ])
        return x, y

    def _get_local_batch(self, split):
        # Recreate np.memmap every batch to avoid a memory leak, as per:
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

        if split == "train":
            train_path = os.path.join(self.data_dir, "train.bin")

            # Fast path: original behavior when monolithic train.bin exists.
            if os.path.exists(train_path):
                data = np.memmap(train_path, dtype=np.uint16, mode="r")
                x, y = self._sample_from_memmap(data, self.batch_size)

            else:
                # Refresh pool if needed.
                if (
                    self._train_pool_shards is None
                    or self._train_pool_steps_used >= self.STEPS_PER_POOL
                ):
                    self._choose_new_train_pool()

                num_pool_shards = len(self._train_pool_shards)
                per_shard = (self.batch_size + num_pool_shards - 1) // num_pool_shards

                x_parts = []
                y_parts = []

                for shard_name in self._train_pool_shards:
                    shard_path = os.path.join(self.data_dir, shard_name)
                    data = np.memmap(shard_path, dtype=np.uint16, mode="r")

                    if len(data) <= self.block_size:
                        continue

                    x_shard, y_shard = self._sample_from_memmap(data, per_shard)
                    x_parts.append(x_shard)
                    y_parts.append(y_shard)

                if not x_parts:
                    raise ValueError(
                        f"Active shard pool contains no shards large enough for block_size={self.block_size}"
                    )

                x = torch.cat(x_parts, dim=0)[:self.batch_size]
                y = torch.cat(y_parts, dim=0)[:self.batch_size]

                self._train_pool_steps_used += 1

        elif split == "val":
            val_path = os.path.join(self.data_dir, "val.bin")
            if not os.path.exists(val_path):
                raise FileNotFoundError(f"Missing validation file: {val_path}")

            data = np.memmap(val_path, dtype=np.uint16, mode="r")
            x, y = self._sample_from_memmap(data, self.batch_size)

        else:
            raise ValueError(f"Unknown split: {split}")

        return self._move_batch_to_device(x, y)

    # -------------------------------------------------------------------------
    # Hugging Face streaming path
    # -------------------------------------------------------------------------

    def _normalize_hf_dataset_ref(self, data_ref):
        """
        Accepts:
          OWNER/DATASET
          https://huggingface.co/datasets/OWNER/DATASET
          https://huggingface.co/datasets/OWNER/DATASET/tree/main/...

        Returns:
          OWNER/DATASET
        """
        prefixes = (
            "https://huggingface.co/datasets/",
            "http://huggingface.co/datasets/",
        )

        for prefix in prefixes:
            if data_ref.startswith(prefix):
                path = data_ref[len(prefix):]
                parts = path.strip("/").split("/")

                if len(parts) < 2:
                    raise ValueError(
                        "Invalid Hugging Face dataset URL. Expected "
                        "https://huggingface.co/datasets/OWNER/DATASET"
                    )

                return "/".join(parts[:2])

        return data_ref

    def _resolve_streaming_split(self, split):
        """
        In holdout mode, both logical train and val come from the HF train split.
        The train/val separation is done by deterministic hashing of each text.
        """
        if self.streaming_holdout_validation:
            return "train"

        if split == "val":
            return self.streaming_val_split

        return split

    def _is_streaming_val_example(self, text):
        """
        Deterministically route raw text examples into the validation stream.

        This avoids storing a validation file on disk. The same text will always
        route to the same split.
        """
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        threshold = int(self.streaming_val_size * (2**64))
        return value < threshold

    def _logical_split_accepts_text(self, logical_split, text):
        """
        Returns True if this raw text belongs to the requested logical split.

        If streaming_holdout_validation=False, all text from the requested HF
        split is accepted.
        """
        if not self.streaming_holdout_validation:
            return True

        is_val = self._is_streaming_val_example(text)

        if logical_split == "val":
            return is_val

        if logical_split == "train":
            return not is_val

        raise ValueError(
            f"Unknown streaming logical split: {logical_split}. "
            "Expected 'train' or 'val' when streaming_holdout_validation=True."
        )

    def _get_streaming_dataset(self, split):
        """
        Lazily create a streaming Hugging Face dataset for the requested split.
        """
        hf_split = self._resolve_streaming_split(split)

        if hf_split not in self._streaming_datasets:
            try:
                from datasets import load_dataset
            except ImportError as e:
                raise ImportError(
                    "Streaming mode requires the Hugging Face datasets package. "
                    "Install it with: pip install datasets"
                ) from e

            if self.streaming_config_name is None:
                dataset = load_dataset(
                    self.data_dir,
                    split=hf_split,
                    streaming=True,
                )
            else:
                dataset = load_dataset(
                    self.data_dir,
                    self.streaming_config_name,
                    split=hf_split,
                    streaming=True,
                )

            self._streaming_datasets[hf_split] = dataset
            self._streaming_iterators[hf_split] = iter(dataset)
            self._streaming_buffers[hf_split] = torch.empty(0, dtype=torch.long)

        return self._streaming_datasets[hf_split]

    def _next_streaming_tokens(self, split):
        """
        Pull the next accepted raw-text example from the HF stream and tokenize it.
        """
        hf_split = self._resolve_streaming_split(split)
        self._get_streaming_dataset(split)

        while True:
            try:
                example = next(self._streaming_iterators[hf_split])
            except StopIteration:
                self._streaming_iterators[hf_split] = iter(self._streaming_datasets[hf_split])
                example = next(self._streaming_iterators[hf_split])

            if self.streaming_text_column not in example:
                raise KeyError(
                    f"Missing text column {self.streaming_text_column!r}. "
                    f"Available columns: {list(example.keys())}"
                )

            text = example[self.streaming_text_column]

            if text is None:
                continue

            if not isinstance(text, str):
                text = str(text)

            if not self._logical_split_accepts_text(split, text):
                continue

            break

        token_ids = self.streaming_tokenizer.encode_ordinary(text)

        if self.streaming_append_eot:
            token_ids.append(self.streaming_tokenizer.eot_token)

        if len(token_ids) == 0:
            return torch.empty(0, dtype=torch.long)

        return torch.tensor(token_ids, dtype=torch.long)

    def _fill_streaming_buffer(self, split):
        """
        Fill the RAM token buffer for this logical split up to streaming_buffer_tokens.
        """
        hf_split = self._resolve_streaming_split(split)
        buffer_key = split if self.streaming_holdout_validation else hf_split

        buffer = self._streaming_buffers.get(
            buffer_key,
            torch.empty(0, dtype=torch.long),
        )

        chunks = []
        current_len = buffer.numel()

        if current_len > 0:
            chunks.append(buffer)

        while current_len < self.streaming_buffer_tokens:
            tokens = self._next_streaming_tokens(split)

            if tokens.numel() == 0:
                continue

            chunks.append(tokens)
            current_len += tokens.numel()

        self._streaming_buffers[buffer_key] = torch.cat(chunks, dim=0)[
            :self.streaming_buffer_tokens
        ]

    def _advance_streaming_buffer(self, split):
        """
        Keep the back half of the current buffer and refill from the stream.

        This avoids storing data on disk while still giving random token windows
        inside a moving RAM buffer.
        """
        hf_split = self._resolve_streaming_split(split)
        buffer_key = split if self.streaming_holdout_validation else hf_split

        buffer = self._streaming_buffers[buffer_key]

        keep = max(
            self.streaming_buffer_tokens // 2,
            self.block_size + 1,
        )

        self._streaming_buffers[buffer_key] = buffer[-keep:].clone()
        self._fill_streaming_buffer(split)

    def _get_streaming_batch(self, split):
        """
        Streaming replacement for local np.memmap batch sampling.

        Returns:
          x: [batch_size, block_size], torch.long
          y: [batch_size, block_size], torch.long
        """
        hf_split = self._resolve_streaming_split(split)
        buffer_key = split if self.streaming_holdout_validation else hf_split

        self._get_streaming_dataset(split)
        self._fill_streaming_buffer(split)

        data = self._streaming_buffers[buffer_key]

        if data.numel() <= self.block_size + 1:
            raise ValueError(
                f"Streaming buffer has only {data.numel()} tokens, "
                f"too small for block_size={self.block_size}"
            )

        max_start = data.numel() - self.block_size - 1

        ix = torch.randint(
            low=0,
            high=max_start,
            size=(self.batch_size,),
        )

        x = torch.stack([
            data[i:i + self.block_size]
            for i in ix
        ])

        y = torch.stack([
            data[i + 1:i + 1 + self.block_size]
            for i in ix
        ])

        self._advance_streaming_buffer(split)

        return self._move_batch_to_device(x, y)