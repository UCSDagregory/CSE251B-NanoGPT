#!/usr/bin/env bash
set -euo pipefail

HF_REVISION="${1:-main}"

REPO="$HOME/CSE251B-NanoGPT"
SCRATCH_BASE="/scratch/$USER"
DATASET_NAME="mixed-data"
SAFE_REVISION="${HF_REVISION//\//__}"
DATASET_DIR="$SCRATCH_BASE/$DATASET_NAME-$SAFE_REVISION"
HF_CACHE="$SCRATCH_BASE/hf-cache"
LINK_PATH="$REPO/data/$DATASET_NAME"
HF_REPO="123Ginger321/mixed-data"
REPO_TYPE="dataset"

cd "$REPO"

mkdir -p "$REPO/data"

# If data/mixed-data exists but is not the correct symlink, stop.
# This avoids accidentally deleting real data in home.
if [ -e "$LINK_PATH" ] && [ ! -L "$LINK_PATH" ]; then
  echo "ERROR: $LINK_PATH exists and is not a symlink."
  echo "Move/delete it manually before running this script."
  exit 1
fi

# Create or fix symlink.
if [ ! -L "$LINK_PATH" ] || [ "$(readlink -f "$LINK_PATH" 2>/dev/null || true)" != "$DATASET_DIR" ]; then
  echo "[setup] Creating symlink: $LINK_PATH -> $DATASET_DIR"
  rm -f "$LINK_PATH"
  mkdir -p "$DATASET_DIR"
  ln -s "$DATASET_DIR" "$LINK_PATH"
else
  echo "[setup] Symlink already correct."
fi

# Verify symlink.
if [ ! -L "$LINK_PATH" ]; then
  echo "ERROR: $LINK_PATH was not created as a symlink."
  exit 1
fi

if [ "$(readlink -f "$LINK_PATH" 2>/dev/null || true)" != "$DATASET_DIR" ]; then
  echo "ERROR: $LINK_PATH points to the wrong location."
  echo "Expected: $DATASET_DIR"
  echo "Actual:   $(readlink -f "$LINK_PATH" 2>/dev/null || true)"
  exit 1
fi

echo "[setup] Verified symlink: $LINK_PATH -> $DATASET_DIR"

mkdir -p "$DATASET_DIR" "$HF_CACHE"

export PATH="$HOME/.local/bin:$PATH"

# Keep HF cache on scratch.
export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"

# Do not disable Xet unless your cluster has problems with it.
# Some newer HF repos use Xet-backed storage.
# export HF_HUB_DISABLE_XET=1

export HF_HUB_ENABLE_HF_TRANSFER=0

echo "[setup] Checking HF auth..."
hf auth whoami >/dev/null 2>&1 || hf auth login

echo "[setup] Inspecting remote files in $HF_REPO at revision $HF_REVISION..."

REMOTE_FILES="$(python3 - <<PY
from huggingface_hub import HfApi

repo_id = "$HF_REPO"
repo_type = "$REPO_TYPE"
revision = "$HF_REVISION"

api = HfApi()
files = api.list_repo_files(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
)

for f in files:
    print(f)
PY
)"

echo "$REMOTE_FILES" | sed 's/^/  /'

REMOTE_DATA_FILES="$(
  echo "$REMOTE_FILES" | grep -Ev '(^$|^\.gitattributes$|^README\.md$|^dataset_infos\.json$|^LICENSE$|^\.git)' || true
)"

if [ -z "$REMOTE_DATA_FILES" ]; then
  echo "ERROR: Remote repo appears to contain only metadata files."
  echo "Repo checked: https://huggingface.co/datasets/$HF_REPO/tree/$HF_REVISION"
  echo
  echo "This means there is probably no actual dataset uploaded to this dataset repo,"
  echo "or the data is in a different repo type, branch, or private/gated location."
  exit 1
fi

# If files already appear to be downloaded, exit.
# This accepts common dataset formats instead of only *.bin.
if find "$DATASET_DIR" -type f \
  \( -name "*.bin" \
     -o -name "*.npy" \
     -o -name "*.npz" \
     -o -name "*.pt" \
     -o -name "*.pth" \
     -o -name "*.safetensors" \
     -o -name "*.parquet" \
     -o -name "*.arrow" \
     -o -name "*.jsonl" \
     -o -name "*.json" \
     -o -name "*.txt" \
     -o -name "*.csv" \
     -o -name "*.tsv" \
     -o -name "*.gz" \
     -o -name "*.zst" \
     -o -name "*.zip" \) \
  | grep -q .; then
  if ! find "$DATASET_DIR" -type f \( -name "*.incomplete" -o -name "*.lock" \) | grep -q .; then
    echo "[setup] mixed-data appears already downloaded for revision $HF_REVISION:"
    du -sh "$DATASET_DIR"
    find "$DATASET_DIR" -maxdepth 2 -type f -printf "  %P\n" | sort
    exit 0
  fi
fi

echo "[setup] Download not complete. Starting/resuming download..."

hf download "$HF_REPO" \
  --repo-type "$REPO_TYPE" \
  --revision "$HF_REVISION" \
  --local-dir "$LINK_PATH" \
  --max-workers 6

echo "[setup] Verifying local downloaded files..."

LOCAL_DATA_FILES="$(
  find "$DATASET_DIR" -type f \
    ! -name ".gitattributes" \
    ! -name "README.md" \
    ! -name "dataset_infos.json" \
    ! -name "LICENSE" \
    ! -name "*.lock" \
    ! -name "*.incomplete" \
    -printf "%P\n" | sort
)"

if [ -z "$LOCAL_DATA_FILES" ]; then
  echo "ERROR: Download completed, but no real dataset files were found locally."
  echo "Only metadata may have been downloaded."
  echo
  echo "Remote files seen were:"
  echo "$REMOTE_FILES" | sed 's/^/  /'
  exit 1
fi

echo "[setup] Download complete for revision $HF_REVISION:"
du -sh "$DATASET_DIR"
echo "$LOCAL_DATA_FILES" | sed 's/^/  /'

# Optional: clear HF cache after successful download to save scratch space.
rm -rf "$HF_CACHE"
echo "[setup] Cleared HF cache: $HF_CACHE"
