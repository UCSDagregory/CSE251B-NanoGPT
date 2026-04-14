import importlib.util
import sys
import os
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

        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = device_type

        self.SHARDS_PER_POOL = shards_per_pool
        self.STEPS_PER_POOL = steps_per_pool

        # Persistent train-pool state
        self._train_pool_shards = None
        self._train_pool_steps_used = 0

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

    def get_batch(self, split):
        # We recreate np.memmap every batch to avoid a memory leak, as per:
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

        if split == "train":
            train_path = os.path.join(self.data_dir, "train.bin")

            # Fast path: original behavior when monolithic train.bin exists
            if os.path.exists(train_path):
                data = np.memmap(train_path, dtype=np.uint16, mode="r")
                x, y = self._sample_from_memmap(data, self.batch_size)

            else:
                # Refresh pool if needed
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

        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y