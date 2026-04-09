"""
Download and tokenize FineWeb-Edu (sample-10BT) for training.
Produces train.bin and val.bin in this directory.
Run once: python data/fineweb/prepare.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

num_proc = 8
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

if __name__ == '__main__':
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           split="train", num_proc=num_proc)

    split = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split['val'] = split.pop('test')

    for name, dset in split.items():
        tokenized = dset.map(
            process,
            remove_columns=dset.column_names,
            desc=f"tokenizing {name}",
            num_proc=num_proc,
        )
        arr_len = np.sum(tokenized['len'], dtype=np.uint64)
        fname = os.path.join(os.path.dirname(__file__), f'{name}.bin')
        arr = np.memmap(fname, dtype=np.uint16, mode='w+', shape=(arr_len,))
        idx = 0
        for batch_idx in tqdm(range(1024), desc=f'writing {name}.bin'):
            batch = tokenized.shard(num_shards=1024, index=batch_idx,
                                    contiguous=True).with_format('numpy')
            chunk = np.concatenate(batch['ids'])
            arr[idx:idx + len(chunk)] = chunk
            idx += len(chunk)
        arr.flush()
        print(f"{name}.bin: {arr_len:,} tokens")
