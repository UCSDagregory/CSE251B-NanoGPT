#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPT_DIR"

cd "$REPO"

mkdir -p "$REPO/logs"

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

# Prefer a repo-local token file, but allow HF_TOKEN to already be exported.
HF_TOKEN_ENV_FILE="${HF_TOKEN_ENV_FILE:-$REPO/.hf_token_env}"

if [ -f "$HF_TOKEN_ENV_FILE" ]; then
    source "$HF_TOKEN_ENV_FILE"
elif [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN already set in environment."
#else
#    echo "ERROR: Hugging Face token not found."
#    echo "Expected either:"
#    echo "  1. HF_TOKEN already exported in the environment, or"
#    echo "  2. Token file at: $HF_TOKEN_ENV_FILE"
#    echo
#    echo "Create it with:"
#    echo "  echo 'export HF_TOKEN=hf_your_token_here' > \"$REPO/.hf_token_env\""
#    echo "  chmod 600 \"$REPO/.hf_token_env\""
#    exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false

# Keep Hugging Face caches out of home quota.
export HF_HOME="${TMPDIR:-/tmp}/hf_home_${USER}_${SLURM_JOB_ID:-manual}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install Python packages into temporary job-local storage.
# This avoids using your limited DSMLP home quota.
export PYTHONUSERBASE="${TMPDIR:-/tmp}/pyuserbase_${USER}_${SLURM_JOB_ID:-manual}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$(python -c 'import site; print(site.getusersitepackages())'):${PYTHONPATH:-}"

echo "Using PYTHONUSERBASE=$PYTHONUSERBASE"

python -m pip install --user --no-cache-dir --upgrade pip setuptools wheel

# Install project requirements if present.
# Pip will print "Requirement already satisfied" for packages already available.
if [ -f "$REPO/requirements.txt" ]; then
    echo "Installing project requirements from $REPO/requirements.txt..."
    python -m pip install --user --no-cache-dir -r "$REPO/requirements.txt"
else
    echo "No requirements.txt found; installing known runtime dependencies explicitly..."
fi

# Explicit dependencies needed by this training/streaming path.
# Keep torch out of this list because the DSMLP image should already provide
# the CUDA-compatible PyTorch build.
python -m pip install --user --no-cache-dir \
    numpy \
    tqdm \
    requests \
    pyarrow \
    datasets \
    huggingface_hub \
    tiktoken \
    git+https://github.com/KellerJordan/Muon

echo "Dependency check:"
python -c "import numpy, tqdm, requests, pyarrow, datasets, huggingface_hub, tiktoken; from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam; print('core deps ok')"
python -c "import torch; print('torch ok:', torch.__version__, 'cuda:', torch.cuda.is_available())"
#python -c "import os; print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))"

echo "Repo contents:"
ls "$REPO"

echo "Starting training smoke test..."

if timeout 12h python "$REPO/train.py" \
  --device cuda \
  --type resume \
  --folder start_dist \
  --data_fd_name "$REPO/data/mixed-data" \
  --chpr checkpoints/ITER00033000_003.2923val_loss_nanoGPT_DaginGregory.pt
then
    echo "Training finished before timeout."
else
    code=$?
    if [ "$code" -eq 124 ]; then
        echo "Training finished successfully."
    else
        echo "Training failed with exit code $code."
        exit "$code"
    fi
fi

echo "Finished: $(date)"
