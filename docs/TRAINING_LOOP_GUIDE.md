# Training a Custom `torch.nn.Module` with `train.py`

## TL;DR

To get started:

1. Copy the `MODEL_EXAMPLE` folder.
2. Rename the copy to your model folder name (for example, `my_model`).
3. Make sure your copied folder includes:
   - `model.py`
   - `training_model_params.json`
   - `training_opt_params.json`
4. Implement your model in `model.py` as a `torch.nn.Module`.
5. Keep the required helper methods that `train.py` depends on.
6. Point `train.py` to your model folder and your dataset folder containing `train.bin` and `val.bin`.
7. Run either:
   - **scratch training** to start fresh, or
   - **resume training** to continue from a checkpoint.

The easiest path is to start from `MODEL_EXAMPLE` and only change the parts that define your model and optimizer behavior.

---

## How to Start

Create a copy of the `MODEL_EXAMPLE` folder. It already contains everything `train.py` expects, in the format it expects.

`train.py` looks for the following files:

### `training_model_params.json`

This file defines the parameters used to create an instance of your model in `model.py`.

Important details:

- The file is parsed as an **ordered list of values**, based on the order the entries appear in the JSON.
- The **key names do not matter** for functionality. They only exist for readability.
- The **order of the entries is what matters**.

Example:

```json
{ "a": 2, "b": 3, "c": 1 }
```

This will be passed to the class constructor as:

```python
{2, 3, 1}
```

In other words, whatever names you use in the JSON are ignored by the loader for construction purposes; only the order is used.

### `training_opt_params.json`

This file uses the same format as `training_model_params.json`.

Important details:

- The naming convention in the file is only for readability.
- The order of the entries is the only thing that matters.
- This file is used to create the optimizer, as defined in `model.py`.

### `model.py`

This file defines your model.

Required:

- A constructor for a class of type `torch.nn.Module`
- All parameters needed to create your model instance

---

## Functions Required by `train.py`

### Functions you do not need to modify, but should include

These functions are required for `train.py` to run correctly.

#### `estimate_mfu(self, fwdbwd_per_iter, dt)`

Keep this function with the same signature.

You can copy it directly from the `model.py` in `MODEL_EXAMPLE`.

This is a useful training statistic that reports how many FLOPs are being used as a percentage of an A100’s theoretical throughput.

#### Helper functions used by `train.py`

These are relied on for checkpoint saving/loading or for implementation simplicity. In most custom implementations, they do not need to be changed.

```python
def getArgs(checkpoint, model_folder_name="N/A", chkpt_folder_name="N/A")
def loadFromCheckpoint(model_folder_name: str, checkpoint_file_path: str) -> tuple[nn.Module, Any, Any, Any]
def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module
```

These generally do not need to be modified for a custom implementation. Depending on how much you change your architecture or checkpoint format, they might need adjustments, but that is unlikely.

---

## Functions You May Need to Modify

These functions are the ones most likely to require changes when implementing a custom model.

### `saveCheckpoint(self, optimizer: nn.Module, val_loss: int)`

Keep the same signature.

This is the only part of checkpoint saving you would typically ever need to change.

### `forward(self, input_ids, targets=None)`

This function should behave as follows:

- If `targets` is **not** provided, return the logits produced by the forward pass.
- If `targets` **is** provided, return a tuple:

```python
(logits, loss)
```

where `loss` is computed from the labelled example.

### `configure_optimizers(self, weight_decay, learning_rate, betas, device_type)`

This function does not need to use this exact signature, in the same way that your model constructor does not need to exactly mirror the JSON keys.

Its purpose is to create an optimizer using values from `training_opt_params.json`, in a manner similar to how the model constructor uses `training_model_params.json`.

---

## How to Run `train.py`

### Train a model from scratch

```bash
python train.py --device cuda|cpu|mps --type scratch --folder my_model --data_fd_name data/shakespeare_char [--chpn custom_checkpoints]
```

Arguments:

- `--device cuda|cpu|mps`  
  Selects the device to train on.
- `--type scratch`  
  Starts training from scratch.
- `--folder my_model`  
  The name of the directory containing the required files.
- `--data_fd_name data/shakespeare_char`  
  The path to the folder containing the `train.bin` and `val.bin` files you want to train on.
- `--chpn custom_checkpoints` *(optional)*  
  The name of the folder where `train.py` should save checkpoints if you want something other than the default, which is `checkpoints`.

### Resume training from a checkpoint

```bash
python train.py --device cuda|cpu|mps --type resume --folder my_model --data_fd_name data/shakespeare_char --chpr checkpoints/my_checkpoint.pt
```

Arguments:

- `--device cuda|cpu|mps`  
  Selects the device to train on.
- `--type resume`  
  Resumes training from a checkpoint.
- `--folder my_model`  
  The name of the model directory.
- `--data_fd_name data/shakespeare_char`  
  Same dataset path format as scratch training.
- `--chpr checkpoints/my_checkpoint.pt`  
  The path, relative to your model directory, to the checkpoint you want to load.

Checkpoint paths must be of the form:

```text
checkpoint_folder_name/checkpoint_name.pt
```

---

## Sample Commands

### Scratch

```bash
python train.py --device cpu --type scratch --folder my_model --data_fd_name data/shakespeare_char
```

### Resume

```bash
python train.py --device cpu --type resume --folder my_model --data_fd_name data/shakespeare_char --chpr checkpoints/checkpoint_name.pt
```

---

## Recommended Workflow

1. Copy `MODEL_EXAMPLE`.
2. Replace the model implementation in `model.py` with your own `torch.nn.Module`.
3. Update `training_model_params.json` so its values match your constructor argument order.
4. Update `training_opt_params.json` so its values match how your optimizer is created.
5. Keep the required helper methods and signatures that `train.py` depends on.
6. Run training from scratch.
7. Use checkpoint resume when continuing training later.
