import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import tiktoken

# Example usage:
# python data/data_mixer.py --output data/mixed_fineweb70openweb30 --probs 0.70,0.30 --sample-chunk-bytes 16777216 --seed 1337 --overwrite data/fineweb_edu data/openwebtext
#
# This version randomly samples chunks from every train shard in each dataset.
# It samples without replacement within each run, so repeated output shards do not
# just keep walking the same source shards in order.

TRAIN_SHARD_PATTERN = re.compile(r"^train(\d+)\.bin$")


def choose_dtype(enc):
    max_token = getattr(enc, "max_token_value", None)
    if max_token is None:
        return np.uint32
    if max_token < 2**16:
        return np.uint16
    if max_token < 2**32:
        return np.uint32
    return np.uint64


def find_train_files(directory):
    train_files = []

    plain_train = directory / "train.bin"
    if plain_train.is_file():
        train_files.append(plain_train)

    numbered = []
    for path in directory.iterdir():
        if not path.is_file():
            continue

        match = TRAIN_SHARD_PATTERN.match(path.name)
        if match:
            numbered.append((int(match.group(1)), path))

    numbered.sort(key=lambda x: x[0])
    train_files.extend(path for _, path in numbered)

    return train_files


def append_file(src, dst_handle, chunk_size=16 * 1024 * 1024):
    with open(src, "rb") as fsrc:
        shutil.copyfileobj(fsrc, dst_handle, length=chunk_size)


def align_down(n, itemsize):
    return n - (n % itemsize)


def validate_probs(probs, input_dirs, tolerance=1e-6):
    if len(probs) != len(input_dirs):
        raise ValueError(
            "Number of probabilities must match number of input directories. "
            "Got " + str(len(probs)) + " probabilities for " + str(len(input_dirs)) + " directories."
        )

    for prob in probs:
        if prob < 0:
            raise ValueError("Probabilities must be non-negative.")

    prob_sum = sum(probs)
    if abs(prob_sum - 1.0) > tolerance:
        raise ValueError("Probabilities must sum to 1. Got sum=" + str(prob_sum))


def validate_args(
    input_dirs,
    output_dir,
    probs,
    overwrite,
    shard_size_bytes,
    max_output_bytes,
    sample_chunk_bytes,
    itemsize,
):
    if not input_dirs:
        raise ValueError("No input directories were provided.")

    for directory in input_dirs:
        if not directory.exists():
            raise FileNotFoundError("Input directory does not exist: " + str(directory))
        if not directory.is_dir():
            raise NotADirectoryError("Input path is not a directory: " + str(directory))

        train_files = find_train_files(directory)
        if not train_files:
            raise ValueError("No train.bin or trainN.bin files found in " + str(directory))

        for path in train_files:
            if path.stat().st_size % itemsize != 0:
                raise ValueError(
                    "Training file size is not divisible by inferred dtype size. "
                    "file=" + str(path) +
                    " size=" + str(path.stat().st_size) +
                    " itemsize=" + str(itemsize)
                )

    validate_probs(probs, input_dirs)

    if shard_size_bytes <= 0:
        raise ValueError("--shard-size-bytes must be positive.")

    if shard_size_bytes % itemsize != 0:
        shard_size_bytes = align_down(shard_size_bytes, itemsize)
        if shard_size_bytes <= 0:
            raise ValueError("--shard-size-bytes is smaller than one token.")

    if max_output_bytes <= 0:
        raise ValueError("--max-output-bytes must be positive.")

    if max_output_bytes % itemsize != 0:
        max_output_bytes = align_down(max_output_bytes, itemsize)
        if max_output_bytes <= 0:
            raise ValueError("--max-output-bytes is smaller than one token.")

    if sample_chunk_bytes <= 0:
        raise ValueError("--sample-chunk-bytes must be positive.")

    if sample_chunk_bytes % itemsize != 0:
        sample_chunk_bytes = align_down(sample_chunk_bytes, itemsize)
        if sample_chunk_bytes <= 0:
            raise ValueError("--sample-chunk-bytes is smaller than one token.")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("train*.bin"))

    val_path = output_dir / "val.bin"
    if val_path.exists():
        existing.append(val_path)

    metadata_path = output_dir / "mix_metadata.txt"
    if metadata_path.exists():
        existing.append(metadata_path)

    if existing and not overwrite:
        raise FileExistsError(
            "Output directory already contains train*.bin, val.bin, or mix_metadata.txt. "
            "Use --overwrite to replace them."
        )

    if overwrite:
        for path in existing:
            path.unlink()


def get_total_train_bytes(input_dirs):
    totals = []

    for directory in input_dirs:
        total = 0
        for path in find_train_files(directory):
            total += path.stat().st_size
        totals.append(total)

    return totals


def compute_mixed_total_bytes(dataset_bytes, probs, max_output_bytes, itemsize):
    limits = []

    for available_bytes, prob in zip(dataset_bytes, probs):
        if prob == 0:
            continue
        limits.append(available_bytes / prob)

    if not limits:
        raise ValueError("At least one probability must be greater than zero.")

    total_output_bytes = int(min(limits))
    total_output_bytes = min(total_output_bytes, max_output_bytes)
    total_output_bytes = align_down(total_output_bytes, itemsize)

    if total_output_bytes <= 0:
        raise ValueError("Computed output size is zero. Check input sizes and probabilities.")

    return total_output_bytes


def compute_dataset_targets(total_output_bytes, probs, itemsize):
    targets = []
    assigned = 0
    nonzero_indices = [i for i, p in enumerate(probs) if p > 0]

    for prob in probs:
        if prob == 0:
            targets.append(0)
            continue

        amount = int(total_output_bytes * prob)
        amount = align_down(amount, itemsize)

        targets.append(amount)
        assigned += amount

    remainder = total_output_bytes - assigned

    if remainder > 0:
        last = nonzero_indices[-1]
        targets[last] += remainder

    return targets


def compute_shard_plan(current_shard_size, probs, itemsize):
    plan = []
    assigned = 0
    nonzero_indices = [i for i, p in enumerate(probs) if p > 0]

    for prob in probs:
        if prob == 0:
            plan.append(0)
            continue

        amount = int(current_shard_size * prob)
        amount = align_down(amount, itemsize)

        plan.append(amount)
        assigned += amount

    remainder = current_shard_size - assigned

    if remainder > 0:
        last = nonzero_indices[-1]
        plan[last] += remainder

    return plan


class DatasetChunkSampler:
    """
    Randomly samples byte chunks from all train shards in a dataset.

    Sampling is without replacement within a run:
    - Each source train file is split into virtual chunks.
    - The virtual chunks are shuffled.
    - Reads consume from the shuffled chunk list.
    - If a requested read only uses part of a chunk, the remainder of that chunk
      is kept as the active chunk and consumed next.

    This avoids repeatedly walking the same source shard order across output shards.
    """

    def __init__(self, directory, itemsize, sample_chunk_bytes, rng):
        self.directory = directory
        self.itemsize = itemsize
        self.sample_chunk_bytes = align_down(sample_chunk_bytes, itemsize)
        self.rng = rng

        if self.sample_chunk_bytes <= 0:
            raise ValueError("sample_chunk_bytes must be at least one token.")

        self.files = find_train_files(directory)
        self.chunks = []
        self.chunk_index = 0
        self.active_chunk = None

        self._build_chunks()
        self.rng.shuffle(self.chunks)

        self.total_bytes = sum(size for _, _, size in self.chunks)
        self.remaining_bytes = self.total_bytes

    def _build_chunks(self):
        for path in self.files:
            file_size = path.stat().st_size

            if file_size % self.itemsize != 0:
                raise ValueError(
                    "Training file size is not divisible by itemsize. "
                    "file=" + str(path) +
                    " size=" + str(file_size) +
                    " itemsize=" + str(self.itemsize)
                )

            offset = 0
            while offset < file_size:
                chunk_size = min(self.sample_chunk_bytes, file_size - offset)
                chunk_size = align_down(chunk_size, self.itemsize)

                if chunk_size <= 0:
                    break

                self.chunks.append((path, offset, chunk_size))
                offset += chunk_size

    def _next_chunk(self):
        if self.active_chunk is not None:
            return self.active_chunk

        if self.chunk_index >= len(self.chunks):
            return None

        self.active_chunk = self.chunks[self.chunk_index]
        self.chunk_index += 1

        return self.active_chunk

    def read(self, nbytes):
        nbytes = align_down(nbytes, self.itemsize)

        if nbytes <= 0:
            return []

        if nbytes > self.remaining_bytes:
            raise RuntimeError(
                "Dataset sampler exhausted for "
                + str(self.directory)
                + ". Wanted "
                + str(nbytes)
                + " bytes, but only "
                + str(self.remaining_bytes)
                + " bytes remain."
            )

        pieces = []
        remaining_to_read = nbytes

        while remaining_to_read > 0:
            chunk = self._next_chunk()

            if chunk is None:
                raise RuntimeError(
                    "Dataset sampler unexpectedly exhausted for " + str(self.directory)
                )

            path, offset, chunk_size = chunk
            take = min(chunk_size, remaining_to_read)
            take = align_down(take, self.itemsize)

            if take <= 0:
                raise RuntimeError("Internal alignment error while sampling chunks.")

            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read(take)

            if len(data) != take:
                raise RuntimeError(
                    "Short read from "
                    + str(path)
                    + ". Wanted "
                    + str(take)
                    + " bytes, got "
                    + str(len(data))
                    + " bytes."
                )

            if len(data) % self.itemsize != 0:
                raise RuntimeError(
                    "Internal alignment error: sampled byte count is not divisible by itemsize."
                )

            pieces.append(data)

            remaining_to_read -= take
            self.remaining_bytes -= take

            leftover = chunk_size - take
            if leftover > 0:
                self.active_chunk = (path, offset + take, leftover)
            else:
                self.active_chunk = None

        return pieces


def write_mixed_train_shards(
    input_dirs,
    output_dir,
    probs,
    shard_size_bytes,
    max_output_bytes,
    sample_chunk_bytes,
    seed,
    itemsize,
):
    rng = np.random.default_rng(seed)

    dataset_bytes = get_total_train_bytes(input_dirs)

    for directory, available_bytes, prob in zip(input_dirs, dataset_bytes, probs):
        print(
            "dataset",
            directory,
            "available_train_bytes=",
            available_bytes,
            "prob=",
            prob,
        )

    total_output_bytes = compute_mixed_total_bytes(
        dataset_bytes=dataset_bytes,
        probs=probs,
        max_output_bytes=max_output_bytes,
        itemsize=itemsize,
    )

    dataset_targets = compute_dataset_targets(
        total_output_bytes=total_output_bytes,
        probs=probs,
        itemsize=itemsize,
    )

    print("total_output_bytes=", total_output_bytes)
    print("dataset_target_bytes=", dataset_targets)
    print("sample_chunk_bytes=", sample_chunk_bytes)
    print("seed=", seed)

    samplers = [
        DatasetChunkSampler(
            directory=directory,
            itemsize=itemsize,
            sample_chunk_bytes=sample_chunk_bytes,
            rng=rng,
        )
        for directory in input_dirs
    ]

    remaining_by_dataset = list(dataset_targets)

    shard_index = 0
    total_written = 0

    while total_written < total_output_bytes:
        current_shard_size = min(shard_size_bytes, total_output_bytes - total_written)
        current_shard_size = align_down(current_shard_size, itemsize)

        if current_shard_size <= 0:
            break

        requested_plan = compute_shard_plan(current_shard_size, probs, itemsize)

        shard_plan = []
        assigned = 0

        for i, requested_amount in enumerate(requested_plan):
            amount = min(requested_amount, remaining_by_dataset[i])
            amount = align_down(amount, itemsize)
            shard_plan.append(amount)
            assigned += amount

        deficit = current_shard_size - assigned

        if deficit > 0:
            for i in range(len(shard_plan)):
                if deficit <= 0:
                    break

                available_extra = remaining_by_dataset[i] - shard_plan[i]
                available_extra = align_down(available_extra, itemsize)

                if available_extra <= 0:
                    continue

                extra = min(deficit, available_extra)
                extra = align_down(extra, itemsize)

                shard_plan[i] += extra
                deficit -= extra

        actual_shard_size = sum(shard_plan)

        if actual_shard_size <= 0:
            break

        shard_path = output_dir / ("train" + str(shard_index) + ".bin")
        print("writing", shard_path)
        print("shard_plan_bytes=", shard_plan)

        pieces = []

        for i, amount in enumerate(shard_plan):
            if amount <= 0:
                continue

            dataset_pieces = samplers[i].read(amount)
            pieces.extend(dataset_pieces)
            remaining_by_dataset[i] -= amount

        rng.shuffle(pieces)

        with open(shard_path, "wb") as fout:
            written_this_shard = 0

            for piece in pieces:
                fout.write(piece)
                written_this_shard += len(piece)

        if written_this_shard != actual_shard_size:
            raise RuntimeError(
                "Internal write accounting error for "
                + str(shard_path)
                + ". Expected "
                + str(actual_shard_size)
                + " bytes, wrote "
                + str(written_this_shard)
                + " bytes."
            )

        total_written += actual_shard_size
        shard_index += 1

    return shard_index, total_written, dataset_targets


def combine_validation_files(input_dirs, output_dir, itemsize):
    val_output_path = output_dir / "val.bin"
    val_count = 0

    with open(val_output_path, "wb") as val_output:
        for directory in input_dirs:
            src_path = directory / "val.bin"

            if src_path.is_file():
                if src_path.stat().st_size % itemsize != 0:
                    raise ValueError(
                        "val.bin size is not divisible by inferred dtype size. "
                        "file=" + str(src_path) +
                        " size=" + str(src_path.stat().st_size) +
                        " itemsize=" + str(itemsize)
                    )

                print("appending", src_path, "to", val_output_path)
                append_file(src_path, val_output)
                val_count += 1

    if val_count == 0:
        val_output_path.unlink()
        print("no val.bin files found")

    return val_count


def write_mix_metadata(
    output_dir,
    input_dirs,
    probs,
    dataset_targets,
    total_written,
    max_output_bytes,
    encoding,
    dtype,
    itemsize,
    sample_chunk_bytes,
    seed,
):
    metadata_path = output_dir / "mix_metadata.txt"

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("encoding\t" + str(encoding) + "\n")
        f.write("dtype\t" + str(np.dtype(dtype)) + "\n")
        f.write("itemsize\t" + str(itemsize) + "\n")
        f.write("total_train_bytes\t" + str(total_written) + "\n")
        f.write("max_output_bytes\t" + str(max_output_bytes) + "\n")
        f.write("sample_chunk_bytes\t" + str(sample_chunk_bytes) + "\n")
        f.write("seed\t" + str(seed) + "\n")
        f.write("sampling\twithout_replacement_random_chunks\n")
        f.write("dataset\tprobability\ttarget_train_bytes\n")

        for directory, prob, target in zip(input_dirs, probs, dataset_targets):
            f.write(str(directory) + "\t" + str(prob) + "\t" + str(target) + "\n")

    print("wrote", metadata_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Mix train .bin shards from multiple dataset directories according to probabilities. "
            "Training data is sampled as randomized chunks across all train shards in each dataset."
        )
    )

    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="Input directories containing train.bin, trainN.bin, and optionally val.bin.",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--probs",
        type=str,
        required=True,
        help="Comma-separated mixture probabilities, e.g. 0.85,0.15",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="gpt2",
        help="Same tiktoken encoding used to create the input .bin files. Default: gpt2.",
    )

    parser.add_argument(
        "--shard-size-bytes",
        type=int,
        default=5_000_000_000,
        help="Maximum size of each output trainN.bin shard in bytes. Default: 5,000,000,000.",
    )

    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=22 * 1024 * 1024 * 1024,
        help="Maximum total size of output train shards in bytes. Default: 22 GiB.",
    )

    parser.add_argument(
        "--sample-chunk-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help=(
            "Chunk size used when randomly sampling from input train shards. "
            "Smaller values give better mixing but more file seeks. Default: 16 MiB."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible mixing. Change this between experiments for different samples.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing train*.bin, val.bin, and mix_metadata.txt in output directory.",
    )

    args = parser.parse_args()

    args.probs = [float(x.strip()) for x in args.probs.split(",")]

    input_dirs = [Path(x).resolve() for x in args.input_dirs]
    output_dir = Path(args.output).resolve()

    enc = tiktoken.get_encoding(args.encoding)
    dtype = choose_dtype(enc)
    itemsize = np.dtype(dtype).itemsize

    print("Using encoding:", args.encoding)
    print("Inferred dtype:", np.dtype(dtype))
    print("Inferred itemsize:", itemsize)

    shard_size_bytes = align_down(args.shard_size_bytes, itemsize)
    max_output_bytes = align_down(args.max_output_bytes, itemsize)
    sample_chunk_bytes = align_down(args.sample_chunk_bytes, itemsize)

    validate_args(
        input_dirs=input_dirs,
        output_dir=output_dir,
        probs=args.probs,
        overwrite=args.overwrite,
        shard_size_bytes=shard_size_bytes,
        max_output_bytes=max_output_bytes,
        sample_chunk_bytes=sample_chunk_bytes,
        itemsize=itemsize,
    )

    shard_count, total_written, dataset_targets = write_mixed_train_shards(
        input_dirs=input_dirs,
        output_dir=output_dir,
        probs=args.probs,
        shard_size_bytes=shard_size_bytes,
        max_output_bytes=max_output_bytes,
        sample_chunk_bytes=sample_chunk_bytes,
        seed=args.seed,
        itemsize=itemsize,
    )

    val_count = combine_validation_files(
        input_dirs=input_dirs,
        output_dir=output_dir,
        itemsize=itemsize,
    )

    write_mix_metadata(
        output_dir=output_dir,
        input_dirs=input_dirs,
        probs=args.probs,
        dataset_targets=dataset_targets,
        total_written=total_written,
        max_output_bytes=max_output_bytes,
        encoding=args.encoding,
        dtype=dtype,
        itemsize=itemsize,
        sample_chunk_bytes=sample_chunk_bytes,
        seed=args.seed,
    )

    print("done")
    print("training shards written:", shard_count)
    print("training bytes written:", total_written)
    print("validation files combined:", val_count)
    print("output directory:", output_dir)


if __name__ == "__main__":
    main()