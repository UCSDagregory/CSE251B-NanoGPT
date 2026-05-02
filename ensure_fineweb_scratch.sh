#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/CSE251B-NanoGPT"
SCRATCH_BASE="/scratch/$USER"
DATASET_DIR="$SCRATCH_BASE/fineweb"
HF_CACHE="$SCRATCH_BASE/hf-cache"
LINK_PATH="$REPO/data/fineweb"
HF_REPO="123Ginger321/fineweb-edu-shards"

cd "$REPO"

mkdir -p "$REPO/data"

# If data/fineweb exists but is not the correct symlink, stop.
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

# If files already appear to be downloaded, exit.
# Adjust this check if your repo has different required files.
if find "$DATASET_DIR" -maxdepth 1 -type f -name "*.bin" | grep -q .; then
  if ! find "$DATASET_DIR" -type f \( -name "*.incomplete" -o -name "*.lock" \) | grep -q .; then
    echo "[setup] FineWeb appears already downloaded:"
    du -sh "$DATASET_DIR"
    find "$DATASET_DIR" -maxdepth 1 -type f -printf "  %f\n" | sort
    exit 0
  fi
fi

echo "[setup] Download not complete. Starting/resuming download..."

mkdir -p "$DATASET_DIR" "$HF_CACHE"

export PATH="$HOME/.local/bin:$PATH"
# Keep small auth token/config persistent in home.
# Keep large downloaded/cache files on scratch.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

hf auth login

hf download "$HF_REPO" \
  --repo-type dataset \
  --local-dir "$LINK_PATH" \
  --max-workers 6

echo "[setup] Download complete:"
du -sh "$DATASET_DIR"
find "$DATASET_DIR" -maxdepth 1 -type f -printf "  %f\n" | sort

# Optional: clear HF cache after successful download to save scratch space.
rm -rf "$HF_CACHE"
echo "[setup] Cleared HF cache: $HF_CACHE"
