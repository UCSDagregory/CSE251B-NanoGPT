# fineweb-edu dataset

## overview

Uses the `sample-10BT` subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a high-quality educational web content dataset filtered from FineWeb using a classifier trained to score educational value.

## after running `prepare.py` we get (approximate):

- train.bin is ~19GB, val.bin ~9.5MB
- train has ~10B tokens, val has ~5M tokens
- from ~9.67M documents

tokenized with GPT-2 BPE via tiktoken (vocab size 50,257). documents are separated by `<|endoftext|>` tokens.

## usage

```bash
pip install tiktoken datasets tqdm numpy
python prepare.py
```

## notes

- the `sample-10BT` subset is far more data than a 100M param model needs for a single epoch — you likely won't exhaust it
- for faster iteration during development, you can use a smaller slice:
  ```python
  dataset = dataset.select(range(500_000))  # ~500k docs for quick experiments
  ```
- to use even less data, swap `sample-10BT` for loading the full dataset with streaming and taking the first N examples

## references

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [FineWeb-Edu paper (arXiv:2406.17557)](https://arxiv.org/abs/2406.17557)
- [FineWeb (parent dataset)](https://huggingface.co/datasets/HuggingFaceFW/fineweb)