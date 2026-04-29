#!/bin/bash
set -euo pipefail

cd /dsmlp/home-fs03/05/305/dagregory/CSE251B-NanoGPT

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Started: $(date)"

echo "Python:"
which python
python --version

echo "GPU:"
nvidia-smi

echo "Torch CUDA:"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

if [ -f ~/.hf_token_env ]; then
    source ~/.hf_token_env
else
    echo "ERROR: ~/.hf_token_env not found."
    exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="${TMPDIR:-/tmp}/hf_home_${USER}_${SLURM_JOB_ID:-manual}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! python -c "import datasets, huggingface_hub, tiktoken" >/dev/null 2>&1; then
    echo "Missing streaming dependencies; installing into job-local TMPDIR..."

    export PYTHONUSERBASE="${TMPDIR:-/tmp}/pyuserbase_${USER}_${SLURM_JOB_ID:-manual}"
    python -m pip install --user --no-cache-dir datasets huggingface_hub tiktoken

    export PATH="$PYTHONUSERBASE/bin:$PATH"
    export PYTHONPATH="$(python -c 'import site; print(site.getusersitepackages())'):${PYTHONPATH:-}"
else
    echo "Streaming dependencies already installed."
fi

echo "Dependency check:"
python -c "import datasets, huggingface_hub, tiktoken; print('deps ok')"
python -c "import os; print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))"

echo "Repo contents:"
ls

echo "Model folder contents:"
ls DG_test_initial7M

echo "Starting training smoke test..."

if timeout 3m python train.py \
  --device cuda \
  --type scratch \
  --folder DG_test_initial7M \
  --data_fd_name "https://huggingface.co/datasets/HuggingFaceFW/fineweb" \
  --stream T \
  --stream_config 10BT
then
    echo "Training finished before timeout."
else
    code=$?
    if [ "$code" -eq 124 ]; then
        echo "Smoke test stopped after 3m as expected."
    else
        echo "Training failed with exit code $code."
        exit "$code"
    fi
fi

echo "Finished: $(date)"