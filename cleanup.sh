# cleanup.sh
# Source this file, do not execute it with `bash cleanup.sh`.

cleanup_jobs() {
  local pids

  pids="$(jobs -p)"

  if [[ -z "$pids" ]]; then
    echo "No background jobs to clean up."
    return 0
  fi

  pkill -f "train.py"
  pkill -f "slurm_train.sh"

  echo "$pids" | xargs -r kill -9
  echo "Killed background jobs:"
  echo "$pids"
  rm -rf .././.scratch
}

cleanup_jobs
