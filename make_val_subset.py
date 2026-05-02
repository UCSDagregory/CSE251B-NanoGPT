import argparse
from pathlib import Path

import numpy as np


DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def resolve_output_path(output_arg):
    output_path = Path(output_arg).resolve()

    if output_path.exists() and output_path.is_dir():
        return output_path / "val.bin"

    if output_path.suffix == "":
        return output_path / "val.bin"

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a validation subset from an already-tokenized val.bin by sampling "
            "random contiguous chunks across the full file."
        )
    )

    parser.add_argument("input_file", help="Input val.bin file.")
    parser.add_argument("--output", "-o", required=True, help="Output file or output directory.")
    parser.add_argument("--tokens", type=int, default=1_000_000, help="Default: 1,000,000.")
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="uint16")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=8192,
        help="Contiguous chunk size in tokens. Default: 8192.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input_file).resolve()
    output_path = resolve_output_path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError("Input file does not exist: " + str(input_path))

    if args.tokens <= 0:
        raise ValueError("--tokens must be positive.")

    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive.")

    dtype = DTYPE_MAP[args.dtype]
    itemsize = np.dtype(dtype).itemsize
    input_bytes = input_path.stat().st_size

    if input_bytes % itemsize != 0:
        raise ValueError(
            "Input file size is not divisible by dtype size. "
            + "file_size="
            + str(input_bytes)
            + " dtype="
            + args.dtype
            + " itemsize="
            + str(itemsize)
        )

    available_tokens = input_bytes // itemsize
    tokens_to_write = min(args.tokens, available_tokens)

    if tokens_to_write < args.tokens:
        print(
            "warning: requested",
            args.tokens,
            "tokens but source only has",
            available_tokens,
            "tokens; writing all available tokens",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError("Output file exists. Use --overwrite: " + str(output_path))

    data = np.memmap(input_path, dtype=dtype, mode="r")
    rng = np.random.default_rng(args.seed)

    chunks = []
    remaining = tokens_to_write

    max_chunk = min(args.chunk_tokens, available_tokens)

    while remaining > 0:
        current_chunk_tokens = min(max_chunk, remaining)

        max_start = available_tokens - current_chunk_tokens
        if max_start < 0:
            raise RuntimeError("Internal error: chunk size exceeds available tokens.")

        start = int(rng.integers(0, max_start + 1))
        end = start + current_chunk_tokens

        chunk = np.asarray(data[start:end], dtype=dtype)
        chunks.append(chunk)

        remaining -= current_chunk_tokens

    subset = np.concatenate(chunks)

    if subset.shape[0] != tokens_to_write:
        raise RuntimeError(
            "Internal token count mismatch. Expected "
            + str(tokens_to_write)
            + " tokens, got "
            + str(subset.shape[0])
            + " tokens."
        )

    subset.tofile(output_path)

    written_bytes = output_path.stat().st_size
    written_tokens = written_bytes // itemsize

    if written_tokens != tokens_to_write:
        raise RuntimeError(
            "Output token count mismatch. Expected "
            + str(tokens_to_write)
            + " tokens, wrote "
            + str(written_tokens)
            + " tokens."
        )

    print("done")
    print("input:", input_path)
    print("output:", output_path)
    print("dtype:", args.dtype)
    print("available tokens:", available_tokens)
    print("tokens written:", written_tokens)
    print("chunk tokens:", args.chunk_tokens)
    print("chunks written:", len(chunks))
    print("seed:", args.seed)


if __name__ == "__main__":
    main()