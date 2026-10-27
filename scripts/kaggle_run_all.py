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

RL without the SFT stage (raw checkpoint + fresh LoRA):
    python scripts/kaggle_run_all.py --from-base

A stronger base checkpoint (any HF seq2seq, prompted — no SFT):
    python scripts/kaggle_run_all.py --model bigscience/mt0-base
    python scripts/kaggle_run_all.py --from-base --model bigscience/mt0-large

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
    parser.add_argument("--from-base", action="store_true",
                        help="skip SFT; train RL directly on the raw checkpoint "
                             "with a fresh LoRA (no SFT adapter needed); implies "
                             "--from rl and forwards --from-base to the RL stage")
    parser.add_argument("--model", default=None,
                        help="HF seq2seq checkpoint for from-base RL "
                             "(e.g. bigscience/mt0-base); implies --from-base")
    parser.add_argument("--prompt", default=None,
                        help="override the paraphrase instruction prompt for "
                             "--model checkpoints ({question} placeholder); "
                             "forwarded to the RL stage")
    args = parser.parse_args()
    if args.model:
        if args.stage == "sft":
            parser.error("--model is a from-base RL option; use --from all or --from rl")
        args.stage = "rl"
    if args.from_base:
        if args.stage == "sft":
            parser.error("--from-base skips SFT; use --from all or --from rl")
        args.stage = "rl"
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
        if not args.from_base and not adapter_ready(SFT_DIR):
            print(f"\nNo SFT adapter at {SFT_DIR / 'adapter'}. "
                  "Run without --from rl (or with --from all) first, or use "
                  "--from-base to train RL straight from the raw checkpoint.")
            return 2
        extra_options: tuple[str, ...] = ()
        if args.eval_only:
            extra_options = (*extra_options, "--eval-only")
        if args.from_base:
            extra_options = (*extra_options, "--from-base")
        if args.model:
            extra_options = (*extra_options, "--model", args.model)
        if args.prompt:
            extra_options = (*extra_options, "--rl-prompt", args.prompt)
        code = run_stage("STAGE 2 — RL", ROOT / "scripts" / "kaggle_run_rl.py",
                         stage_args(args.rl_args, extra_options))
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
        print(f"  ceiling (original + rewrites) {ev['ceiling_f1']:.4f} "
              f"({ev['ceiling_delta_f1']:+.4f}); "
              f"recoverable {ev['recoverable_rate']:.3f}")
        best_margin = ev["margin_sweep"][ev["best_margin"]]["selected_f1"]
        print(f"  best confidence margin on dev: {ev['best_margin']} "
              f"({best_margin:.4f}) — diagnostic, not a deployment claim")
        # Only a rule whose paired CI clears zero is deployable; otherwise the
        # sweep maximum is a number, not a result.
        rules = {
            "consistent": (ev["selected_f1"], ev.get("delta_ci95")),
            "corrective": (ev["corrective_f1"],
                           ev.get("corrective_delta_ci95")),
            "consensus": (ev["consensus_f1"],
                          ev.get("consensus_delta_ci95")),
        }
        for name, row in ev.get("corrective_sweep", {}).items():
            rules[f"corrective@{name}"] = (row["selected_f1"],
                                           row.get("delta_ci95"))
        for name, row in ev.get("selective_sweep", {}).items():
            rules[f"selective:{name}"] = (row["selected_f1"],
                                          row.get("delta_ci95"))
        deployable = {name: value for name, value in rules.items()
                      if value[0] > ev["original_f1"] + 1e-9
                      and value[1] and value[1][0] > 0}
        if deployable:
            best_rule = max(deployable,
                            key=lambda name: deployable[name][0])
            print(f"  deployable rule: {best_rule} "
                  f"({deployable[best_rule][0]:.4f}, "
                  f"CI[{deployable[best_rule][1][0]:+.4f},"
                  f"{deployable[best_rule][1][1]:+.4f}])")
        else:
            print("  deployable rule: none — no rule clears the original with a "
                  "CI above zero")
            print("  DEPLOYMENT: answer the original question; do not rewrite.")
    if rl_eval:
        verdict = rl_eval["verdict"]
        print(f"RL held-out QA selected-F1: "
              f"{rl_eval['sft']['selected_f1']:.4f} → "
              f"{rl_eval['rl']['selected_f1']:.4f} "
              f"({rl_eval['delta']['selected_f1']:+.4f})")
        if verdict.get("recommended_action") == "original":
            print(f"Verdict: {verdict['reason']} → answer the original "
                  f"question; no {verdict['best_policy'].upper()} selector is "
                  f"deployable")
        else:
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
