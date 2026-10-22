#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
echo "=== Training Telugu ActiveQA policy ==="
bash scripts/run_colab.sh configs/colab_user_data_rl.yaml
echo "=== Training English ActiveQA policy on SearchQA ==="
python scripts/train_searchqa_rl.py --config configs/colab_english_activeqa.yaml
echo "Both full-model RL policies are ready:"
echo "  Telugu full RL policy: outputs/user_data_rl_policy.pt"
echo "  English full RL policy: outputs/english_searchqa_rl_policy.pt"
echo "  English HF export: outputs/english_searchqa_rl_model/"
