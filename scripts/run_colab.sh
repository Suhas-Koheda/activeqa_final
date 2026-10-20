#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p outputs/colab_logs data/hf_home
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="outputs/colab_logs/colab_${RUN_STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "ActiveQA Telugu Colab GPU run started: $(date -Is)"
echo "Project: $ROOT_DIR"
echo "Log: $LOG_FILE"
export HF_HOME="${HF_HOME:-$ROOT_DIR/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU found. In Colab select Runtime > Change runtime type > T4 GPU, then rerun.")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY

echo "Installing Python dependencies without replacing Colab's CUDA PyTorch..."
python -m pip install -q \
  'transformers>=4.40,<5' 'datasets>=2.19' 'rank-bm25>=0.2.2' 'peft>=0.11' \
  'PyYAML>=6' 'psutil>=5.9' 'tqdm>=4.66' 'safetensors>=0.4' \
  'sentencepiece>=0.2.0' 'protobuf>=4.25'

run_stage() {
  local stage="$1"
  shift
  echo
  echo "===== ${stage} started: $(date -Is) ====="
  "$@"
  echo "===== ${stage} finished: $(date -Is) ====="
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv; fi
}

run_stage "prepare smoke split" python scripts/prepare_data.py --config configs/colab_gpu.yaml
run_stage "REINFORCE smoke training" python scripts/train.py --config configs/colab_gpu.yaml
run_stage "paired smoke evaluation" python scripts/evaluate.py --config configs/colab_gpu.yaml --checkpoint outputs/colab_policy.pt

echo
echo "Run completed: $(date -Is)"
echo "Metrics: outputs/train_metrics.json and outputs/eval_metrics.json"
echo "Predictions: outputs/eval_pretrained.json and outputs/eval_trained.json"
echo "Full log: $LOG_FILE"
