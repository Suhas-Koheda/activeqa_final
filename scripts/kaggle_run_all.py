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


def stage_args(extra: str, add: tuple[str, ...] = ()) -> str:
    """Combine pass-through options without requiring nested shell quotes."""
    parts = shlex.split(extra)
    for option in add:
        if option not in parts:
            parts.append(option)
    return shlex.join(parts)


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
                        help="options for scripts/kaggle_run_rl.py; use "
                             '--rl-args="--questions 700 --epochs 2" (the equals '
                             "form) so values starting with -- are not parsed "
                             "as wrapper options")
    parser.add_argument("--eval-only", action="store_true",
                        help="run only the SFT-adapter selector diagnostic from "
                             "the RL stage (no training); implies --from rl")
    args = parser.parse_args()
    if args.eval_only:
        if args.stage == "sft":
            parser.error("--eval-only applies to the RL stage; use --from rl")
        args.stage = "rl"

    if args.stage in ("all", "sft"):
        if args.reuse_sft and adapter_ready(SFT_DIR):
            print("Reusing the existing SFT adapter (--reuse-sft).")
        else:
            code = run_stage("STAGE 1 — SFT", ROOT / "scripts" / "colab_run_sft.py",
                             stage_args(args.sft_args))
            if code != 0:
                print("\nSFT failed — stopping before RL. See "
                      f"{SFT_DIR / 'run.log'} for the traceback.")
                return code
            if not adapter_ready(SFT_DIR):
                print("\nSFT exited successfully but produced no adapter. "
                      f"Check {SFT_DIR / 'run_status.json'} and run.log.")
                return 2
            metrics = load_json(SFT_DIR / "sft_metrics.json") or {}
            collapsed = (metrics.get("distinct_best_ratio", 1.0) < 0.5
                         or metrics.get("token_f1", 1.0) == 0.0)
            if collapsed:
                print("\nSFT produced a collapsed adapter (token_f1="
                      f"{metrics.get('token_f1')}, distinct_best_ratio="
                      f"{metrics.get('distinct_best_ratio')}). Stopping before "
                      "RL. Most likely transformers 5.x was active instead of "
                      f"4.57.6; see {SFT_DIR / 'run.log'}.")
                return 3
    if args.stage in ("all", "rl"):
        if not adapter_ready(SFT_DIR):
            print(f"\nNo SFT adapter at {SFT_DIR / 'adapter'}. "
                  "Run without --from rl (or with --from all) first.")
            return 2
        code = run_stage("STAGE 2 — RL", ROOT / "scripts" / "kaggle_run_rl.py",
                         stage_args(args.rl_args,
                                    ("--eval-only",) if args.eval_only else ()))
        if code != 0:
            print("\nRL failed. The SFT adapter is still valid; see "
                  f"{RL_DIR / 'run.log'} for the traceback.")
            return code

    sft_metrics = load_json(SFT_DIR / "sft_metrics.json")
    qa_path = RL_DIR / "qa_eval.json"
    sft_only_path = RL_DIR / "qa_eval_sft_only.json"
    # A --eval-only run writes qa_eval_sft_only.json. Prefer it when it is newer
    # than qa_eval.json so the summary never shows stale RL numbers as fresh.
    eval_only = (sft_only_path.exists()
                 and (not qa_path.exists()
                      or sft_only_path.stat().st_mtime
                      > qa_path.stat().st_mtime))
    rl_eval = None if eval_only else load_json(qa_path)
    sft_only = load_json(sft_only_path) if eval_only else None
    summary = {
        "sft_metrics": sft_metrics,
        "qa_eval": rl_eval,
        "sft_selector_diagnostics": sft_only,
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
    if sft_only:
        ev = sft_only["sft"]
        print("Selector-only diagnostic (no RL training this run):")
        print(f"  original {ev['original_f1']:.4f} | "
              f"consistent {ev['selected_f1']:.4f} | "
              f"corrective {ev['corrective_f1']:.4f} | "
              f"consensus {ev['consensus_f1']:.4f}")
        print(f"  best confidence margin on dev: {ev['best_margin']} "
              f"({ev['margin_sweep'][ev['best_margin']]['selected_f1']:.4f})")
    if rl_eval:
        verdict = rl_eval["verdict"]
        print(f"RL held-out QA selected-F1: "
              f"{rl_eval['sft']['selected_f1']:.4f} → "
              f"{rl_eval['rl']['selected_f1']:.4f} "
              f"({rl_eval['delta']['selected_f1']:+.4f})")
        print(f"Verdict: {verdict['reason']} → "
              f"{verdict['recommended_action'].upper()} "
              f"({verdict['best_policy']} adapter, "
              f"{verdict['recommended_selector']} selector)")
    print(f"Summary: {ROOT / 'models' / 'pipeline_summary.json'}")
    print("Zip models/indicbartss-reformulator-lora and "
          "models/indicbartss-reformulator-lora-rl to download every artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
