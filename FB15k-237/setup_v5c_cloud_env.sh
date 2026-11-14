#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-intersection_feature_cu128}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10 pip setuptools wheel
fi
conda activate "$ENV_NAME"

python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0
python -m pip install --index-url https://pypi.org/simple \
  numpy==1.24.1 \
  scipy==1.15.3 \
  pandas==2.1.4 \
  scikit-learn==1.3.2 \
  datasketch==1.6.4 \
  tqdm==4.70.0

cd "$PROJECT_DIR"
python -m pip install --no-build-isolation -e .
python - <<'PY'
import torch
import openke

print("openke:", openke.__file__)
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the new environment")
print("GPU:", torch.cuda.get_device_name(0))
PY
