#!/bin/bash
#SBATCH --job-name=dg-fineweb-train
#SBATCH --chdir=/dsmlp/home-fs03/05/305/dagregory/CSE251B-NanoGPT
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Started: $(date)"

echo "Python:"
which python
python --version

echo "GPU:"
nvidia-smi || true

echo "Torch CUDA:"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

# Load Hugging Face token/private env vars.
if [ -f ~/.hf_token_env ]; then
    source ~/.hf_token_env
else
    echo "ERROR: ~/.hf_token_env not found."
    echo "Create it with:"
    echo '  echo '\''export HF_TOKEN="hf_your_token_here"'\'' > ~/.hf_token_env'
    echo '  echo '\''export HF_XET_HIGH_PERFORMANCE=1'\'' >> ~/.hf_token_env'
    echo "  chmod 600 ~/.hf_token_env"
    exit 1
fi

# Hugging Face/network settings.
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false

# Keep HF caches out of your DSMLP home quota.
export HF_HOME="${TMPDIR:-/tmp}/hf_home_${USER}_${SLURM_JOB_ID:-manual}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

# Optional PyTorch CUDA allocation setting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install Python deps into job-local scratch if missing.
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

python train.py \
  --device cuda \
  --type scratch \
  --folder DG_test_initial7M \
  --data_fd_name "https://huggingface.co/datasets/HuggingFaceFW/fineweb" \
  --stream T \
  --stream_config 10BT

echo "Finished: $(date)"