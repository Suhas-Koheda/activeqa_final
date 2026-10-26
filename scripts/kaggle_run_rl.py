#!/usr/bin/env python3
"""Telugu reformulator RL stage: starting policy → QA-reward RFT → paired held-out QA.

This is stage 2 of the one-command pipeline (stage 1 is scripts/colab_run_sft.py).
It uses the frozen Telugu QA model as the reward, runs on-policy
rejection-sampling fine-tuning, then evaluates the starting and RL policies on
the same held-out questions:

    python scripts/kaggle_run_rl.py                 # RL on top of the SFT adapter
    python scripts/kaggle_run_rl.py --from-base     # RL from the raw checkpoint
                                                    # (fresh LoRA, no SFT needed)

The starting policy is the SFT LoRA adapter when available, or the raw
checkpoint with a fresh LoRA when ``--from-base`` is passed.

Checkpoint selection guarantees the run can never ship a policy worse than the
one it started with, no matter how noisy RL training is: every
``--checkpoint-every`` optimizer updates (default 12) the current adapter is
saved to <out>/probe/ and measured with ``evaluate_qa_pairs`` on a dev subset
(``--checkpoint-eval-questions``, default 80). The trained end-state is probed
too. The probe with the best subset ``selected_f1`` — including the starting
policy itself — is reloaded and shipped.

The number that matters is `selected_f1` in <out>/qa_eval.json: the answer
token-F1 of the deployed confidence-selected reformulation, measured against
the original question on the same passage.

Everything is written to the run directory (models/indicbartss-reformulator-lora-rl):
  adapter/            RL LoRA adapter (shipped policy; the starting policy when
                      no checkpoint beat it)
  run.log             full transcript
  run_status.json     last completed stage (survives a crash)
  rl_metrics.json     reward/selection/validity statistics, checkpoint history
  rl_samples.json     per-question training selections (raw, unfiltered)
  qa_eval.json        start vs RL held-out QA summary + accept/reject verdict
  qa_eval_samples.json per-question predictions for both policies
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "ai4bharat/MultiIndicParaphraseGenerationSS"
PINNED_TRANSFORMERS = "4.57.6"
LOCAL_MODEL_DIR = ROOT / "models" / "MultiIndicParaphraseGenerationSS"
QA_MODEL = "l3cube-pune/telugu-question-answering-squad-bert"
SFT_ADAPTER = ROOT / "models" / "indicbartss-reformulator-lora" / "adapter"
RL_DIR = ROOT / "models" / "indicbartss-reformulator-lora-rl"

LOG_HANDLE: dict[str, Path | None] = {"path": None}


def log(message: str) -> None:
    print(message, flush=True)
    path = LOG_HANDLE["path"]
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(str(message) + "\n")


def record_status(path: Path, stage: str, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update({
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    payload.update(fields)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-adapter", default=str(SFT_ADAPTER),
                    help="LoRA adapter produced by scripts/colab_run_sft.py; "
                         "ignored with --from-base")
    ap.add_argument("--from-base", action="store_true",
                    help="train RL directly from the raw checkpoint with a "
                         "fresh LoRA — no SFT stage required")
    ap.add_argument("--rl-lora-rank", type=int, default=16,
                    help="rank of the fresh LoRA when --from-base")
    ap.add_argument("--rl-lora-alpha", type=int, default=32,
                    help="alpha of the fresh LoRA when --from-base")
    ap.add_argument("--checkpoint-eval-questions", type=int, default=80,
                    help="dev questions evaluated at every RL checkpoint "
                         "(checkpoint selection; 0 disables)")
    ap.add_argument("--checkpoint-every", type=int, default=12,
                    help="run a checkpoint eval every N optimizer updates")
    ap.add_argument("--out", default=str(RL_DIR),
                    help="run directory; the RL adapter is written to <out>/adapter")
    ap.add_argument("--train", default="data/indic_rag_te/train.jsonl")
    ap.add_argument("--dev", default="data/indic_rag_te/dev.jsonl")
    ap.add_argument("--questions", type=int, default=640,
                    help="training questions per epoch (0 = all)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--batch-questions", type=int, default=8,
                    help="selected targets per optimizer update")
    ap.add_argument("--generate-batch", type=int, default=8,
                    help="questions generated together (speed); lower if OOM")
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="required reward gain over the original question")
    ap.add_argument("--conf-slack", type=float, default=0.0,
                    help="allowed confidence deficit vs the original question "
                         "(0.0 matches the deployed selector; positive values "
                         "train on rewrites deployment would not choose)")
    ap.add_argument("--max-input", type=int, default=96)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--eval-questions", type=int, default=200)
    ap.add_argument("--eval-candidates", type=int, default=4)
    ap.add_argument("--conf-override", type=float, default=0.5,
                    help="confidence margin that lets a corrective rewrite replace "
                         "the original's answer (reported as corrective_f1)")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip both held-out QA evaluations (fast training only)")
    ap.add_argument("--eval-only", action="store_true",
                    help="evaluate the SFT adapter and the selector sweep on the "
                         "held-out set, then stop (no RL training)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU-friendly code check: 2 questions, 2 candidates")
    return ap.parse_args()


def installed_transformers() -> str:
    """Version on disk. Must be checked before importing transformers."""
    try:
        from importlib.metadata import version
        return version("transformers")
    except Exception:
        return ""


def ensure_deps() -> None:
    """Install the exact pin, then restart before transformers is imported.

    Kaggle now preinstalls transformers 5.x. A ">= 4.57" check accepts it, but
    5.x loads this checkpoint without tying its shared embedding table, which
    collapses every output to the same string. Only an exact version match is
    safe, and the restart avoids keeping already-imported 5.x modules alive.
    """
    current = installed_transformers()
    if current == PINNED_TRANSFORMERS:
        log(f"[deps] transformers {current} already installed")
        return
    log(f"[deps] installing transformers {PINNED_TRANSFORMERS} "
        f"(found {current or 'none'}) …")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        f"transformers=={PINNED_TRANSFORMERS}", "peft>=0.13,<0.22",
        "sentencepiece>=0.2.0", "protobuf>=4.25", "tqdm>=4.66",
        "huggingface_hub>=0.23,<1.0",
    ], stdout=subprocess.DEVNULL)
    log("[deps] done; restarting so the pinned build is the one imported")
    os.execv(sys.executable,
             [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:])


def resolve_model_path() -> str:
    if LOCAL_MODEL_DIR.is_dir() and (list(LOCAL_MODEL_DIR.glob("*.safetensors"))
                                     or list(LOCAL_MODEL_DIR.glob("*.bin"))):
        return str(LOCAL_MODEL_DIR)
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return snapshot_download(
        repo_id=MODEL_ID, local_dir=str(LOCAL_MODEL_DIR),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"])


def adapter_ready(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file()
        or (path / "adapter_model.bin").is_file())


def load_rows(path: str, limit: int, seed: int) -> list[dict]:
    from activeqa.data import _load_jsonl_qa
    rows = _load_jsonl_qa(path)
    random.Random(seed).shuffle(rows)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def paired_table(sft: dict, rl: dict) -> str:
    keys = ("original_f1", "selected_f1", "corrective_f1", "consensus_f1",
            "generation_oracle_f1", "improved_rate", "harmed_rate",
            "switch_rate", "corrective_switch_rate", "corrective_harmed_rate",
            "consensus_switch_rate", "consensus_harmed_rate",
            "valid_generation_rate", "distinct_selected_ratio")
    lines = [f"  {'metric':28s} {'SFT':>9s} {'RL':>9s} {'delta':>9s}"]
    for key in keys:
        delta = rl[key] - sft[key]
        lines.append(f"  {key:28s} {sft[key]:>9.4f} {rl[key]:>9.4f} {delta:>+9.4f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_HANDLE["path"] = out_dir / "run.log"
    status_path = out_dir / "run_status.json"
    record_status(status_path, "started", args=vars(args))

    if args.smoke:
        args.skip_install = installed_transformers() == PINNED_TRANSFORMERS
        args.questions = 2
        args.epochs = 1
        args.candidates = 2
        args.batch_questions = 1
        args.generate_batch = 1
        args.eval_questions = 2
        args.eval_candidates = 2
        args.checkpoint_eval_questions = 2
        args.checkpoint_every = 1
        args.max_new = 16

    sft_adapter = Path(args.sft_adapter)
    if args.eval_only and args.from_base:
        raise SystemExit("--eval-only evaluates the SFT adapter; it is "
                         "incompatible with --from-base")
    if not args.from_base and not adapter_ready(sft_adapter):
        raise SystemExit(
            f"SFT adapter not found at {sft_adapter}. Run scripts/colab_run_sft.py "
            "first, use scripts/kaggle_run_all.py for both stages, or pass "
            "--from-base to train RL without the SFT adapter.")

    import torch
    if not args.skip_install:
        ensure_deps()
    import transformers
    import peft
    if (transformers.__version__ != PINNED_TRANSFORMERS
            and not args.smoke):
        raise SystemExit(
            f"transformers {transformers.__version__} is loaded but this "
            f"pipeline requires {PINNED_TRANSFORMERS}. Rerun without "
            "--skip-install (or reinstall the pin) in a fresh process.")
    log(f"[1/6] RL stage — transformers {transformers.__version__}  "
        f"peft {peft.__version__}  torch {torch.__version__}")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if not use_cuda and not args.smoke:
        raise SystemExit("No CUDA GPU found. Select a GPU runtime (T4) on Kaggle/Colab.")
    if use_cuda:
        log(f"      GPU: {torch.cuda.get_device_name(0)}  "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.path.insert(0, str(ROOT / "scripts"))
    from activeqa.qa_model import TeluguQA
    from activeqa.qa_eval import evaluate_qa_pairs, paired_delta_ci, deployable_rule
    from activeqa.reformulator import Reformulator
    from activeqa.rft import rft_on_policy

    # ── policy (SFT adapter if present, otherwise the raw checkpoint + LoRA) ─
    model_path = resolve_model_path()
    log(f"[2/6] Reformulator {model_path}")
    started = time.perf_counter()
    if args.from_base:
        log(f"      --from-base: no SFT adapter; fresh LoRA "
            f"(r={args.rl_lora_rank}, alpha={args.rl_lora_alpha})")
        ref = Reformulator(model_path, cache_dir="models", device=device,
                           rank=args.rl_lora_rank, alpha=args.rl_lora_alpha)
    else:
        log(f"      loading SFT adapter {sft_adapter}")
        ref = Reformulator(model_path, cache_dir="models", device=device,
                           adapter_path=str(sft_adapter))
    log(f"      trainable params: {ref.trainable_count:,}  "
        f"({time.perf_counter() - started:.1f}s)")
    record_status(status_path, "policy_loaded",
                  adapter=str(sft_adapter) if not args.from_base else None,
                  from_base=args.from_base, trainable=ref.trainable_count)

    qa_started = time.perf_counter()
    log(f"[3/6] Frozen QA reward model {QA_MODEL}")
    qa = TeluguQA(QA_MODEL, cache_dir="models", device=device)
    log(f"      loaded in {time.perf_counter() - qa_started:.1f}s")

    if args.no_eval:
        sft_eval = rl_eval = None
        sft_records = rl_records = []
    else:
        start_name = "base" if args.from_base else "sft"
        dev_rows = load_rows(args.dev, args.eval_questions, args.seed)
        log(f"[4/6] Held-out QA eval — starting policy "
            f"({'raw checkpoint' if args.from_base else 'SFT adapter'}), "
            f"{len(dev_rows)} questions")
        torch.manual_seed(args.seed)
        sft_eval, sft_records = evaluate_qa_pairs(
            ref, qa, dev_rows, candidates=args.eval_candidates,
            limit=args.eval_questions, max_input=args.max_input,
            max_new=args.max_new, confidence_override=args.conf_override,
            log=log, label=start_name)
        record_status(status_path, "pre_eval_done",
                      selected_f1=sft_eval["selected_f1"])
        log(f"      {start_name} selected_f1={sft_eval['selected_f1']:.4f} "
            f"(original {sft_eval['original_f1']:.4f}, "
            f"generation oracle {sft_eval['generation_oracle_f1']:.4f})")
        if args.eval_only:
            print("\n── SFT selector diagnostics (no training) ─────────────────────")
            print(f"  original question        {sft_eval['original_f1']:.4f}")
            print(f"  consistent (deployed)    {sft_eval['selected_f1']:.4f}  "
                  f"switch={sft_eval['switch_rate']:.3f}  "
                  f"harmed={sft_eval['harmed_rate']:.3f}")
            print(f"  confident correction     {sft_eval['corrective_f1']:.4f}  "
                  f"switch={sft_eval['corrective_switch_rate']:.3f}  "
                  f"harmed={sft_eval['corrective_harmed_rate']:.3f}")
            print(f"  answer consensus         {sft_eval['consensus_f1']:.4f}  "
                  f"switch={sft_eval['consensus_switch_rate']:.3f}  "
                  f"harmed={sft_eval['consensus_harmed_rate']:.3f}")
            print("  confidence-margin sweep (consistent rule):")
            for margin, row in sft_eval["margin_sweep"].items():
                print(f"    +{margin}  f1={row['selected_f1']:.4f}  "
                      f"switch={row['switch_rate']:.3f}  "
                      f"harmed={row['harmed_rate']:.3f}")
            print(f"  best margin on this dev set: {sft_eval['best_margin']} "
                  f"(diagnostic only; confirm on a second split)")
            print(f"  ceiling (best of original+rewrites) "
                  f"{sft_eval['ceiling_f1']:.4f} "
                  f"({sft_eval['ceiling_delta_f1']:+.4f}); "
                  f"recoverable {sft_eval['recoverable_rate']:.3f} "
                  f"(gain {sft_eval['recoverable_mean_gain']:+.4f}, "
                  f"orig conf {sft_eval['recoverable_original_confidence']:+.2f} "
                  f"vs {sft_eval['nonrecoverable_original_confidence']:+.2f})")
            auc = sft_eval["signal_auc"]
            print(f"  signal AUC for recoverability: "
                  f"low_original_conf {auc['low_original_confidence']:.3f}  "
                  f"confidence_gain {auc['confidence_gain']:.3f}  "
                  f"disagreement {auc['rewrite_disagreement']:.3f}")
            print("  corrective sweep (highest-confidence rewrite past a "
                  "confidence margin):")
            for margin, row in sft_eval["corrective_sweep"].items():
                print(f"    +{margin}  f1={row['selected_f1']:.4f}  "
                      f"delta={row['delta_f1']:+.4f}  "
                      f"CI[{row['delta_ci95'][0]:+.4f},{row['delta_ci95'][1]:+.4f}]  "
                      f"switch={row['switch_rate']:.3f}  "
                      f"imp={row['improved_rate']:.3f}  "
                      f"harm={row['harmed_rate']:.3f}")
            print(f"  best corrective margin on this dev set: "
                  f"{sft_eval['best_corrective_margin']} "
                  f"({sft_eval['corrective_sweep'][sft_eval['best_corrective_margin']]['selected_f1']:.4f})")
            print("  low-confidence gate sweep (best valid rewrite when the "
                  "original answer is uncertain):")
            for quantile, row in sft_eval["gate_sweep"].items():
                print(f"    q{quantile} ≤{row['threshold']:+.3f}  "
                      f"f1={row['selected_f1']:.4f}  "
                      f"gated={row['examples']:3d}  "
                      f"orig {row['gated_original_f1']:.4f}  "
                      f"ceil {row['gated_ceiling_f1']:.4f}  "
                      f"recov={row['gated_recoverable_rate']:.3f}")
            best = sft_eval["selective_sweep"][sft_eval["best_selective"]]
            print(f"  best gate+margin combo: {sft_eval['best_selective']}  "
                  f"f1={best['selected_f1']:.4f} ({best['delta_f1']:+.4f})  "
                  f"CI[{best['delta_ci95'][0]:+.4f},{best['delta_ci95'][1]:+.4f}]  "
                  f"switch={best['switch_rate']:.3f}")
            rule, rule_options = deployable_rule(sft_eval)
            if rule:
                print(f"  DEPLOYABLE RULE: {rule}  f1={rule_options[rule][0]:.4f}  "
                      f"CI[{rule_options[rule][1][0]:+.4f},"
                      f"{rule_options[rule][1][1]:+.4f}]  "
                      f"(original {sft_eval['original_f1']:.4f})")
            else:
                print("  DEPLOYABLE RULE: none — every selector's paired 95% CI "
                      "includes zero; do not rewrite.")
            (out_dir / "qa_eval_sft_only.json").write_text(
                json.dumps({"sft": sft_eval}, indent=2, ensure_ascii=False),
                encoding="utf-8")
            (out_dir / "qa_eval_samples_sft_only.json").write_text(
                json.dumps({"sft": sft_records}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
            record_status(status_path, "eval_only_done",
                          selected_f1=sft_eval["selected_f1"],
                          consensus_f1=sft_eval["consensus_f1"])
            return

    # ── RL ───────────────────────────────────────────────────────────────────
    # Checkpoint selection: a probe adapter is saved and measured on a dev
    # subset after every N optimizer updates. Only checkpoints that beat the
    # starting policy's subset selected_f1 may survive; otherwise the starting
    # policy is reused for the final held-out eval. RL can therefore never
    # make the shipped policy worse than what it started from.
    import shutil

    cp_rows = []
    starting_subset_f1 = None
    checkpoint_history: list[dict] = []
    best_checkpoint: dict = {"updates": 0, "subset_selected_f1": None,
                             "path": None}
    if not args.no_eval and args.checkpoint_eval_questions > 0:
        cp_rows = load_rows(args.dev, args.checkpoint_eval_questions, args.seed)
    if cp_rows:
        log(f"      checkpoint selection on {len(cp_rows)} dev questions, "
            f"every {args.checkpoint_every} updates")
        torch.manual_seed(args.seed)
        ref.model.eval()
        subset_base, _ = evaluate_qa_pairs(
            ref, qa, cp_rows, candidates=args.eval_candidates,
            limit=len(cp_rows), max_input=args.max_input,
            max_new=args.max_new, confidence_override=args.conf_override,
            log=log, label="checkpoint-baseline")
        ref.model.train()
        starting_subset_f1 = subset_base["selected_f1"]
        best_checkpoint["subset_selected_f1"] = starting_subset_f1
        checkpoint_history.append(
            {"updates": 0, "state": "starting policy",
             "selected_f1": starting_subset_f1})
        log(f"      starting policy subset selected_f1: {starting_subset_f1:.4f}")

    def checkpoint_hook(n_updates: int) -> None:
        if not cp_rows or n_updates % max(1, args.checkpoint_every):
            return
        probe = out_dir / "probe" / f"ckpt_{n_updates:04d}"
        ref.save_adapter(str(probe))
        ref.model.eval()
        torch.manual_seed(args.seed)
        summary, _ = evaluate_qa_pairs(
            ref, qa, cp_rows, candidates=args.eval_candidates,
            limit=len(cp_rows), max_input=args.max_input,
            max_new=args.max_new, confidence_override=args.conf_override,
            log=log, label=f"checkpoint@{n_updates}")
        ref.model.train()
        f1 = summary["selected_f1"]
        checkpoint_history.append({"updates": n_updates, "state": str(probe),
                                   "selected_f1": f1})
        if f1 > best_checkpoint["subset_selected_f1"]:
            best_checkpoint.update(updates=n_updates, subset_selected_f1=f1,
                                   path=str(probe))
            log(f"      checkpoint {n_updates} is the new best subset "
                f"selected_f1: {f1:.4f}")

    train_rows = load_rows(args.train, args.questions, args.seed)
    log(f"[5/6] RL (RFT) on {len(train_rows)} questions × {args.epochs} epoch(s)"
        + ("  [--from-base]" if args.from_base else ""))
    metrics, train_records = rft_on_policy(
        ref, qa, train_rows,
        candidates=args.candidates,
        batch_questions=args.batch_questions,
        epochs=args.epochs,
        learning_rate=args.lr,
        margin=args.margin,
        confidence_slack=args.conf_slack,
        max_input=args.max_input,
        max_new=args.max_new,
        generate_batch=args.generate_batch,
        seed=args.seed,
        after_update=checkpoint_hook if cp_rows else None,
        skip_solved=True,
        log=log)
    record_status(status_path, "train_done",
                  optimizer_steps=metrics["optimizer_steps"],
                  selected_rate=metrics["selected_rate"])

    if metrics["optimizer_steps"] == 0:
        log("      WARNING: no qualifying rewrite was found; the starting policy "
            "was not changed. Lower --margin / --conf-slack or raise --candidates.")
        record_status(status_path, "no_updates")
        (out_dir / "rl_metrics.json").write_text(
            json.dumps({"training": metrics,
                        "sft_adapter": str(sft_adapter),
                        "from_base": args.from_base},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nRL produced no updates — keeping the starting policy unchanged.")
        return

    adapter_dir = out_dir / "adapter"
    used_checkpoint_selection = bool(cp_rows and not args.no_eval)
    reverted = False
    if used_checkpoint_selection:
        # Measure the trained end-state too (its last partial window of
        # updates was not covered by a periodic probe).
        final_probe = out_dir / "probe" / "ckpt_final"
        ref.save_adapter(str(final_probe))
        ref.model.eval()
        torch.manual_seed(args.seed)
        summary, _ = evaluate_qa_pairs(
            ref, qa, cp_rows, candidates=args.eval_candidates,
            limit=len(cp_rows), max_input=args.max_input,
            max_new=args.max_new, confidence_override=args.conf_override,
            log=log, label="checkpoint@final")
        f1 = summary["selected_f1"]
        checkpoint_history.append({
            "updates": metrics["optimizer_steps"], "state": str(final_probe),
            "selected_f1": f1})
        if f1 > best_checkpoint["subset_selected_f1"]:
            best_checkpoint.update(updates=metrics["optimizer_steps"],
                                   subset_selected_f1=f1,
                                   path=str(final_probe))
        if best_checkpoint["path"] is None:
            # No RL checkpoint beat the starting policy on the subset:
            # the shipped RL policy is the starting policy itself.
            reverted = True
            log(f"      RL never beat the starting policy on the checkpoint "
                f"subset ({starting_subset_f1:.4f}); shipping the starting "
                "policy unchanged")
        # Reload the chosen weights so the final held-out eval measures the
        # policy that will actually be shipped, not the drifted end-state.
        ref_device = str(ref.device)
        del ref
        torch.cuda.empty_cache()
        if reverted:
            if args.from_base:
                ref = Reformulator(model_path, cache_dir="models",
                                   device=ref_device,
                                   rank=args.rl_lora_rank,
                                   alpha=args.rl_lora_alpha)
            else:
                ref = Reformulator(model_path, cache_dir="models",
                                   device=ref_device,
                                   adapter_path=str(sft_adapter))
            ref.model.eval()
        else:
            ref = Reformulator(model_path, cache_dir="models",
                               device=ref_device,
                               adapter_path=str(best_checkpoint["path"]))
            ref.model.eval()
            log(f"      shipping checkpoint at {best_checkpoint['updates']} "
                f"updates (subset selected_f1 "
                f"{best_checkpoint['subset_selected_f1']:.4f})")
        # Clean the probe adapters only after the chosen weights are reloaded
        # elsewhere, so the files are no longer needed.
        shutil.rmtree(out_dir / "probe", ignore_errors=True)
    if args.from_base and reverted:
        rl_adapter_path = None      # the starting (adapter-free) policy ships
    else:
        ref.save_adapter(str(adapter_dir))
        rl_adapter_path = str(adapter_dir)
        log(f"[6/6] RL adapter saved: {adapter_dir}")
    record_status(status_path, "adapter_saved",
                  optimizer_steps=metrics["optimizer_steps"],
                  reverted_to_start=reverted)
    (out_dir / "rl_metrics.json").write_text(json.dumps({
        "training": metrics,
        "sft_adapter": str(sft_adapter),
        "from_base": args.from_base,
        "rl_adapter": rl_adapter_path,
        "checkpoint_selection": {
            "enabled": used_checkpoint_selection,
            "subset_questions": len(cp_rows),
            "checkpoint_every": args.checkpoint_every,
            "starting_subset_selected_f1": starting_subset_f1,
            "best": best_checkpoint,
            "history": checkpoint_history,
            "reverted_to_start": reverted,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "rl_samples.json").write_text(
        json.dumps(train_records, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.no_eval:
        log("Skipping held-out QA eval (--no-eval).")
        record_status(status_path, "finished", accepted=None)
        return

    # ── post-RL eval on the same questions ───────────────────────────────────
    log(f"[6/6] Held-out QA eval — RL policy, {len(dev_rows)} questions")
    torch.manual_seed(args.seed)
    rl_eval, rl_records = evaluate_qa_pairs(
        ref, qa, dev_rows, candidates=args.eval_candidates,
        limit=args.eval_questions, max_input=args.max_input,
        max_new=args.max_new, confidence_override=args.conf_override,
        log=log, label="rl")
    record_status(status_path, "eval_done", selected_f1=rl_eval["selected_f1"])

    delta = rl_eval["selected_f1"] - sft_eval["selected_f1"]
    corrective_delta = rl_eval["corrective_f1"] - sft_eval["corrective_f1"]
    consensus_delta = rl_eval["consensus_f1"] - sft_eval["consensus_f1"]
    # RL − SFT must be bootstrapped on the per-question difference, not on each
    # policy's difference from the original (those are different quantities).
    selected_ci = paired_delta_ci(rl_records, sft_records, "selected_f1")
    corrective_ci = paired_delta_ci(rl_records, sft_records, "corrective_f1")
    consensus_ci = paired_delta_ci(rl_records, sft_records, "consensus_f1")
    stable = (rl_eval["valid_generation_rate"] >= sft_eval["valid_generation_rate"] - 0.05
              and rl_eval["distinct_selected_ratio"] >= 0.50
              and rl_eval["tamil_spill_outputs"] == 0
              and rl_eval["kannada_spill_outputs"] == 0
              and rl_eval["devanagari_spill_outputs"] == 0)
    strict_ok = bool(delta > 0
                     and rl_eval["harmed_rate"] <= sft_eval["harmed_rate"] + 0.05
                     and stable)
    corrective_ok = bool(corrective_delta > 0
                         and rl_eval["corrective_harmed_rate"]
                         <= sft_eval["corrective_harmed_rate"] + 0.05
                         and stable)
    consensus_ok = bool(consensus_delta > 0
                        and rl_eval["consensus_harmed_rate"]
                        <= sft_eval["consensus_harmed_rate"] + 0.05
                        and stable)
    accepted = strict_ok or corrective_ok or consensus_ok
    significant = bool(selected_ci[0] > 0 or corrective_ci[0] > 0
                       or consensus_ci[0] > 0)

    best_policy = "rl" if accepted else start_name
    best_eval = rl_eval if accepted else sft_eval
    selector_scores = {
        "consistent": best_eval["selected_f1"],
        "corrective": best_eval["corrective_f1"],
        "consensus": best_eval["consensus_f1"],
    }
    rule, rule_options = deployable_rule(best_eval)
    best_selector_f1 = (rule_options[rule][0] if rule
                        else max(selector_scores.values()))
    # A swept threshold only counts as deployable when its CI excludes zero.
    recommended_selector = rule or max(selector_scores,
                                       key=selector_scores.get)
    recommended_action = best_policy if rule else "original"
    if accepted:
        reason = "RL improved the held-out QA objective"
    elif rule:
        reason = ("RL did not beat the starting policy; keep the better "
                  f"{start_name} selector")
    else:
        reason = ("No reformulation policy beat the original question on "
                  "held-out QA with a CI above zero")
    verdict = {
        "accepted": accepted,
        "strict_selection_improved": strict_ok,
        "corrective_selection_improved": corrective_ok,
        "consensus_selection_improved": consensus_ok,
        "significant": significant,
        "recommended_selector": recommended_selector,
        "recommended_action": recommended_action,
        "beats_original": bool(rule),
        "best_policy": best_policy,
        "best_selector_f1": round(best_selector_f1, 4),
        "original_f1": rl_eval["original_f1"],
        "delta": {"selected_f1": round(delta, 4),
                  "corrective_f1": round(corrective_delta, 4),
                  "consensus_f1": round(consensus_delta, 4)},
        "paired_ci95": {"selected_f1": selected_ci,
                        "corrective_f1": corrective_ci,
                        "consensus_f1": consensus_ci},
        "reason": reason,
        "from_base": args.from_base,
        "sft_adapter": str(sft_adapter),
        "rl_adapter": str(adapter_dir),
    }

    print("\n── Held-out QA: original vs selected reformulation ──────────────")
    print(paired_table(sft_eval, rl_eval))
    print(f"\n  Consistent selector   RL − SFT selected_f1: {delta:+.4f}   "
          f"paired 95% CI [{selected_ci[0]:+.4f}, {selected_ci[1]:+.4f}]")
    print(f"  Confident correction  RL − SFT corrective_f1: {corrective_delta:+.4f}   "
          f"paired 95% CI [{corrective_ci[0]:+.4f}, {corrective_ci[1]:+.4f}] "
          f"(override ≥ {rl_eval['confidence_override']})")
    print(f"  Answer consensus      RL − SFT consensus_f1: {consensus_delta:+.4f}   "
          f"paired 95% CI [{consensus_ci[0]:+.4f}, {consensus_ci[1]:+.4f}]")
    print(f"  Policy quality (generation oracle): "
          f"{rl_eval['generation_oracle_f1'] - sft_eval['generation_oracle_f1']:+.4f}")
    print("  SFT confidence-margin sweep: " + "  ".join(
        f"{margin}→{row['selected_f1']:.4f}"
        for margin, row in sft_eval["margin_sweep"].items()))
    print(f"  SFT confidence-gate sweep: best q={sft_eval['best_gate']} "
          f"→ {sft_eval['gate_sweep'][sft_eval['best_gate']]['selected_f1']:.4f} "
          f"(original {sft_eval['original_f1']:.4f}); "
          f"ceiling {sft_eval['ceiling_f1']:.4f}")
    print(f"  SFT corrective sweep: best margin "
          f"{sft_eval['best_corrective_margin']} → "
          f"{sft_eval['corrective_sweep'][sft_eval['best_corrective_margin']]['selected_f1']:.4f}  "
          f"| best gate+margin {sft_eval['best_selective']} → "
          f"{sft_eval['selective_sweep'][sft_eval['best_selective']]['selected_f1']:.4f}")
    print(f"  Deployable rule (CI above zero): {rule or 'none'}")
    print(f"  Original question F1: {rl_eval['original_f1']:.4f}   "
          f"best policy+selector: {best_selector_f1:.4f} "
          f"({best_policy}/{recommended_selector})")
    print(f"  Verdict: {verdict['reason']} → use the {best_policy.upper()} "
          f"adapter with the {recommended_selector} selector")
    if recommended_action == "original":
        print("  DEPLOYMENT: do not rewrite. No selector clears the original "
              "question with a CI above zero on this held-out set.")
    if accepted and not significant:
        print("  NOTE: the change is positive but its paired 95% CI includes 0; "
              "rerun with more --questions / --epochs before claiming a gain.")
    if not accepted:
        print("  The RL adapter is kept for inspection; downstream runs should keep "
              "using the starting policy unless you decide otherwise.")

    print("\n── Raw RL examples (no decode-time filtering beyond the policy's "
          "own validity gate) ──")
    for record in rl_records[:3]:
        print(f"\nQ: {record['question']}")
        print(f"   gold: {record['references']}")
        print(f"   original ({record['original_f1']:.2f}): {record['original_answer']}")
        if record["switched"]:
            print(f"   selected ({record['selected_f1']:.2f}): "
                  f"{record['selected_question']}")
        else:
            print("   selected: original question kept")
        for index, (text, f1, valid) in enumerate(zip(
                record["candidate_texts"], record["candidate_f1s"],
                record["candidate_validity"]), 1):
            print(f"     cand {index} ({f1:.2f}, valid={int(valid)}): {text}")

    (out_dir / "qa_eval.json").write_text(json.dumps({
        "sft": sft_eval,
        "rl": rl_eval,
        "delta": {"selected_f1": round(delta, 4),
                  "corrective_f1": round(corrective_delta, 4),
                  "consensus_f1": round(consensus_delta, 4),
                  "generation_oracle_f1": round(
                      rl_eval["generation_oracle_f1"]
                      - sft_eval["generation_oracle_f1"], 4)},
        "paired_ci95": verdict["paired_ci95"],
        "verdict": verdict,
        "training": metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "qa_eval_samples.json").write_text(json.dumps({
        "sft": sft_records, "rl": rl_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    record_status(status_path, "finished", accepted=accepted,
                  best_policy=verdict["best_policy"],
                  recommended_action=verdict["recommended_action"],
                  recommended_selector=verdict["recommended_selector"],
                  delta_selected_f1=round(delta, 4))
    log(f"\nDone. Metrics: {out_dir / 'qa_eval.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        log("FAILED with traceback above")
        path = LOG_HANDLE["path"]
        if path:
            try:
                record_status(path.parent / "run_status.json", "failed",
                              error=traceback.format_exc().strip().splitlines()[-1][:300])
            except Exception:
                pass
        raise
