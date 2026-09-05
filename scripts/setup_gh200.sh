#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# The GH200 image supplies a CUDA-enabled ARM64 Torch build for Python 3.10.
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip==25.3 setuptools==82.0.1
.venv/bin/python -m pip install -e '.[capture]' -r requirements-gh200.txt
.venv/bin/python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'The environment must supply CUDA-enabled ARM64 PyTorch'
print('GPU:', torch.cuda.get_device_name(), torch.cuda.get_device_capability())
PY
cmake -S . -B build-gh200 -DSTATECUT_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gh200 --parallel 4
