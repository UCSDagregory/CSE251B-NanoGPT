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
python -c "import os; print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))"