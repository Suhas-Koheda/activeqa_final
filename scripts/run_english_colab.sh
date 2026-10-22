#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p outputs/english_searchqa data/hf_home
export HF_HOME="${HF_HOME:-$ROOT_DIR/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required. Select a Colab GPU runtime and rerun.")
print("GPU:", torch.cuda.get_device_name(0))
PY
python -m pip install -q \
  'transformers>=4.40,<5' 'datasets>=2.19' 'rank-bm25>=0.2.2' \
  'PyYAML>=6' 'psutil>=5.9' 'tqdm>=4.66' 'safetensors>=0.4' \
  'sentencepiece>=0.2.0' 'protobuf>=4.25'
python scripts/train_searchqa_rl.py --config configs/colab_english_activeqa.yaml
