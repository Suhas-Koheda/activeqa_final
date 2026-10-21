#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG_PATH="${1:-configs/colab_external_smoke.yaml}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config not found: $CONFIG_PATH" >&2
  echo "Usage: bash scripts/run_colab.sh [configs/colab_external_smoke.yaml]" >&2
  exit 2
fi
mkdir -p outputs/colab_logs data/hf_home
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="outputs/colab_logs/colab_${RUN_STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "ActiveQA Telugu Colab GPU run started: $(date -Is)"
echo "Project: $ROOT_DIR"
echo "Config: $CONFIG_PATH"
echo "Log: $LOG_FILE"
export HF_HOME="${HF_HOME:-$ROOT_DIR/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: PyTorch is not installed. Run this inside Colab after selecting Runtime > Change runtime type > GPU."
  echo "This script intentionally does not install a PyTorch wheel because that could replace Colab's CUDA build."
  exit 1
fi

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

# Colab may include an old optional torchao (for example 0.10.0). Recent PEFT
# raises while dispatching LoRA if that incompatible optional package is present,
# even though this project does not use torchao quantization.
if python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
if find_spec("torchao") is None:
    raise SystemExit(0)
try:
    from packaging.version import Version
    if Version(version("torchao")) < Version("0.16.0"):
        raise SystemExit(1)
except PackageNotFoundError:
    raise SystemExit(0)
PY
then
  echo "torchao is absent or compatible with PEFT."
else
  echo "Removing incompatible optional torchao so PEFT can attach standard LoRA layers."
  python -m pip uninstall -y torchao
fi

run_stage() {
  local stage="$1"
  shift
  echo
  echo "===== ${stage} started: $(date -Is) ====="
  "$@"
  echo "===== ${stage} finished: $(date -Is) ====="
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv; fi
}

run_stage "prepare data split" python scripts/prepare_data.py --config "$CONFIG_PATH"
CHECKPOINT_PATH="$(python - "$CONFIG_PATH" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    print(yaml.safe_load(stream)["training"]["checkpoint"])
PY
)"
if [[ "${2:-}" == "--eval-only" || "${COLAB_EVAL_ONLY:-0}" == "1" ]]; then
  run_stage "paired evaluation (eval-only)" python scripts/evaluate.py --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT_PATH"
else
  run_stage "policy training" python scripts/train.py --config "$CONFIG_PATH"
  run_stage "paired evaluation" python scripts/evaluate.py --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT_PATH"
fi

echo
echo "Run completed: $(date -Is)"
echo "Metrics: outputs/train_metrics.json and outputs/eval_metrics.json"
echo "Predictions: outputs/eval_pretrained.json and outputs/eval_trained.json"
echo "Full log: $LOG_FILE"
