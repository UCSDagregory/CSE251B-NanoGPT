#!/bin/bash
set -euo pipefail

echo "Step 1: Fetching dataset to volatile scratch drive..."
python -m pip install --user -U "huggingface_hub[cli]"
source ~/.hf_token_env
bash ensure_mixed_data.sh 45151525

echo "Step 2: Launching training in the background..."
# This is your teammate's exact magic command
nohup bash -c 'PYTHONUNBUFFERED=1 bash slurm_train.sh 2>&1 | PYTHONUNBUFFERED=1 python -u ./circular_log.py' > monitor.log 2>&1 &

# Save the process ID so you can kill it later if needed
echo $! > train.pid

echo "========================================="
echo "Success! Training is running in the background."
echo "You can safely close this terminal or monitor the progress by running:"
echo "tail -f monitor.log"
echo "========================================="
