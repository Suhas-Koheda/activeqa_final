#!/usr/bin/env python3
"""One command for the whole Telugu reformulator pipeline on Kaggle/Colab.

    python scripts/kaggle_run_all.py

Stages (each in its own process, so the SFT run's memory is released before RL):

1. scripts/colab_run_sft.py   — install deps, download model, LoRA SFT,
                                base-vs-SFT held-out paraphrase evaluation.
2. scripts/kaggle_run_rl.py   — load the SFT adapter, run on-policy RFT with the
                                frozen Telugu QA model as reward, evaluate the
                                SFT and RL policies on held-out QA, and decide
                                whether the RL adapter is an improvement.

Rerun only one stage:
    python scripts/kaggle_run_all.py --from rl
    python scripts/kaggle_run_all.py --from sft
    python scripts/kaggle_run_all.py --reuse-sft          # reuse a saved adapter

Forward stage-specific options with quoted strings:
    python scripts/kaggle_run_all.py --rl-args "--questions 500 --epochs 3"
    python scripts/kaggle_run_all.py --sft-args "--epochs 2 --lr 1.5e-4"

Artifacts:
    models/indicbartss-reformulator-lora/      SFT adapter, sft_metrics.json, run.log
    models/indicbartss-reformulator-lora-rl/   RL adapter, rl_metrics.json,
                                               qa_eval.json, run.log
    models/pipeline_summary.json               both stages in one file
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = ROOT / "models" / "indicbartss-reformulator-lora"
RL_DIR = ROOT / "models" / "indicbartss-reformulator-lora-rl"


def adapter_ready(directory: Path) -> bool:
    adapter = directory / "adapter"
    return (adapter / "adapter_config.json").is_file() and (
        (adapter / "adapter_model.safetensors").is_file()
        or (adapter / "adapter_model.bin").is_file())


def run_stage(name: str, script: Path, extra: str) -> int:
    command = [sys.executable, "-u", str(script), *shlex.split(extra)]
    print(f"\n{'=' * 72}\n{name}: {' '.join(command)}\n{'=' * 72}", flush=True)
    started = time.perf_counter()
    code = subprocess.call(command, cwd=str(ROOT))
    print(f"\n{name} finished with exit code {code} in "
          f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return code


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="stage", default="all",
                        choices=("all", "sft", "rl"),
                        help="start at this stage (default: all)")
    parser.add_argument("--reuse-sft", action="store_true",
                        help="skip SFT when models/indicbartss-reformulator-lora/adapter exists")
    parser.add_argument("--sft-args", default="",
                        help="quoted options for scripts/colab_run_sft.py")
    parser.add_argument("--rl-args", default="",
                        help="quoted options for scripts/kaggle_run_rl.py")
    args = parser.parse_args()

    if args.stage in ("all", "sft"):
        if args.reuse_sft and adapter_ready(SFT_DIR):
            print("Reusing the existing SFT adapter (--reuse-sft).")
        else:
            code = run_stage("STAGE 1 — SFT", ROOT / "scripts" / "colab_run_sft.py",
                             args.sft_args)
            if code != 0:
                print("\nSFT failed — stopping before RL. See "
                      f"{SFT_DIR / 'run.log'} for the traceback.")
                return code
            if not adapter_ready(SFT_DIR):
                print("\nSFT exited successfully but produced no adapter. "
                      f"Check {SFT_DIR / 'run_status.json'} and run.log.")
                return 2
    if args.stage in ("all", "rl"):
        if not adapter_ready(SFT_DIR):
            print(f"\nNo SFT adapter at {SFT_DIR / 'adapter'}. "
                  "Run without --from rl (or with --from all) first.")
            return 2
        code = run_stage("STAGE 2 — RL", ROOT / "scripts" / "kaggle_run_rl.py",
                         args.rl_args)
        if code != 0:
            print("\nRL failed. The SFT adapter is still valid; see "
                  f"{RL_DIR / 'run.log'} for the traceback.")
            return code

    sft_metrics = load_json(SFT_DIR / "sft_metrics.json")
    rl_eval = load_json(RL_DIR / "qa_eval.json")
    summary = {
        "sft_metrics": sft_metrics,
        "qa_eval": rl_eval,
        "artifacts": {
            "sft_adapter": str(SFT_DIR / "adapter"),
            "rl_adapter": str(RL_DIR / "adapter"),
        },
    }
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "models" / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    if sft_metrics:
        print(f"SFT paraphrase token-F1: "
              f"{sft_metrics.get('baseline', {}).get('token_f1', float('nan')):.4f} "
              f"→ {sft_metrics.get('token_f1', float('nan')):.4f}   "
              f"(optimizer steps {sft_metrics.get('optimizer_steps', '?')})")
    if rl_eval:
        verdict = rl_eval["verdict"]
        print(f"RL held-out QA selected-F1: "
              f"{rl_eval['sft']['selected_f1']:.4f} → "
              f"{rl_eval['rl']['selected_f1']:.4f} "
              f"({rl_eval['delta']['selected_f1']:+.4f})")
        print(f"Verdict: {verdict['reason']} → {verdict['best_policy'].upper()} adapter")
    print(f"Summary: {ROOT / 'models' / 'pipeline_summary.json'}")
    print("Zip models/indicbartss-reformulator-lora and "
          "models/indicbartss-reformulator-lora-rl to download every artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
