#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${MARINEEVT_CONDA_ENV:-marineevt-label}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$PROJECT_ROOT/deploy/marineevt-label.requirements.txt"
python -m pip install -e "$PROJECT_ROOT/evt_r1/tools/sam3"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA; fix the NVIDIA driver/runtime before continuing.")
PY

echo
echo "Environment ready. Activate it with: conda activate $ENV_NAME"
