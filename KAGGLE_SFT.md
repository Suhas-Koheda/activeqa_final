# Kaggle runbook: SFT → RL in one command

The whole Telugu reformulator pipeline now runs as a single command:

1. **SFT** — LoRA fine-tune `ai4bharat/MultiIndicParaphraseGenerationSS` on
   52,877 Telugu question/paraphrase pairs, with a base-vs-SFT held-out
   evaluation on identical questions.
2. **RL** — load that adapter and run on-policy rejection-sampling fine-tuning
   (RFT) with the frozen `l3cube-pune/telugu-question-answering-squad-bert` QA
   model as the reward.
3. **Paired held-out QA** — answer 200 `data/indic_rag_te/dev.jsonl` questions
   with the original question and with the confidence-selected rewrite, for the
   SFT policy and the RL policy. This is the number that says whether RL helped.

## Notebook settings

- **Accelerator:** GPU T4 (the script uses the first GPU; fp16 autocast is
  probed and automatically falls back to fp32 because this checkpoint produces
  NaN under fp16 on T4 + torch 2.11)
- **Internet:** on (model download from Hugging Face)
- **Persistence:** none needed — everything lives under `/kaggle/working`

## Cell 1 — get the code

```python
!rm -rf /kaggle/working/activeqa_final
!git clone --depth 1 https://github.com/Suhas-Koheda/activeqa_final.git /kaggle/working/activeqa_final
%cd /kaggle/working/activeqa_final
```

## Cell 2 — run everything

```python
!cd /kaggle/working/activeqa_final && python scripts/kaggle_run_all.py
```

Expected wall time on a T4: ~8–12 min for SFT (including the two held-out
evaluations), ~6–10 min for RL (pre-eval, 2 × 320 training questions, post-eval),
plus ~2 min of downloads on a cold session.

Watch for these lines:

```
GPU: Tesla T4  14.6 GiB  amp=auto
embeddings tied to checkpoint shared table
AMP probe (float16): loss=nan finite=False
non-finite under autocast -> training in fp32
first loss 2.6
done: 1102 steps (1102 optimizer updates) ... fp16_amp=off
token_f1 0.2698 → 0.2907   delta_token_f1 0.0209
...
[1/6] RL stage — transformers 4.57.6  peft ...
      trainable params: 2,359,296
[5/6] RL (RFT) on 320 questions × 2 epoch(s)
      RFT ... selected=0.4x  lift=+0.0xx  loss=...
      RFT done: 640 questions, NN updates, selected=0.4x, lift=+0.0xx
── Held-out QA: original vs selected reformulation ──────────────
  metric                              SFT        RL     delta
  original_f1                      ...
  selected_f1                      ...
  generation_oracle_f1             ...
  ...
  RL − SFT selected_f1: +0.0xx   95% CI [...]
  Verdict: ... → use the SFT/RL adapter
```

Reading the RL verdict:

- **`selected_f1`** — deployable held-out QA token-F1 after confidence selection.
  This is the objective.
- **`generation_oracle_f1`** — best rewrite the policy produced, ignoring
  selection. If this rises but `selected_f1` does not, the selector is the
  bottleneck; if it does not rise, the policy itself did not improve.
- **`delta_ci95`** — paired bootstrap interval over dev questions. A positive
  delta whose CI includes 0 needs a longer run before claiming a gain.
- **`harmed_rate` / `valid_generation_rate` / `distinct_selected_ratio`** —
  the stability guards behind `accepted`.

## Cell 3 — inspect the metrics

```python
import json
from pathlib import Path
root = Path("/kaggle/working/activeqa_final/models")
sft = json.loads((root / "indicbartss-reformulator-lora/sft_metrics.json").read_text())
print("SFT token-F1:", sft["baseline"]["token_f1"], "->", sft["token_f1"])
qa_path = root / "indicbartss-reformulator-lora-rl/qa_eval.json"
if qa_path.exists():
    qa = json.loads(qa_path.read_text())
    print(json.dumps({"sft": qa["sft"], "rl": qa["rl"],
                      "delta": qa["delta"], "verdict": qa["verdict"]},
                     indent=2, ensure_ascii=False))
else:
    print((root / "indicbartss-reformulator-lora-rl/run.log").read_text()[-4000:])
```

Per-question predictions (questions, selected rewrites, answers, gold answers,
per-candidate F1s) are in `models/indicbartss-reformulator-lora-rl/qa_eval_samples.json`.

## Cell 4 — download the artifacts

```python
import os, shutil
models = "/kaggle/working/activeqa_final/models"
keep = [d for d in ("indicbartss-reformulator-lora",
                    "indicbartss-reformulator-lora-rl")
        if os.path.isdir(os.path.join(models, d))]
if keep:
    stage = "/kaggle/working/activeqa_stage"
    os.makedirs(stage, exist_ok=True)
    for name in keep:
        shutil.copytree(os.path.join(models, name), os.path.join(stage, name),
                        dirs_exist_ok=True)
    shutil.copy2(os.path.join(models, "pipeline_summary.json"), stage)
    shutil.make_archive("/kaggle/working/activeqa_artifacts", "zip", stage)
    print(round(os.path.getsize("/kaggle/working/activeqa_artifacts.zip") / 1e6, 2), "MB")
    print("Download from the Output panel → activeqa_artifacts.zip")
else:
    print("No artifacts produced — inspect the run logs:")
    for name in ("indicbartss-reformulator-lora", "indicbartss-reformulator-lora-rl"):
        log = os.path.join(models, name, "run.log")
        print(f"\n--- {name}/run.log tail ---")
        print("".join(open(log).readlines()[-60:]) if os.path.exists(log) else "(missing)")
```

## Rerun only one stage

```python
# reuse the saved SFT adapter and rerun RL with more data/steps
!cd /kaggle/working/activeqa_final && python scripts/kaggle_run_all.py --from rl \
  --rl-args "--questions 700 --epochs 3 --eval-questions 200"

# reuse everything and just redo RL from the SFT adapter in a fresh session
!cd /kaggle/working/activeqa_final && python scripts/kaggle_run_all.py --from rl

# rerun SFT only (no RL)
!cd /kaggle/working/activeqa_final && python scripts/kaggle_run_all.py --from sft

# the individual stages still work on their own
!cd /kaggle/working/activeqa_final && python scripts/colab_run_sft.py
!cd /kaggle/working/activeqa_final && python scripts/kaggle_run_rl.py --skip-install
```

Useful overrides:

```python
# faster RL pass (fewer questions, still 2 epochs)
!... python scripts/kaggle_run_all.py --from rl --rl-args "--questions 200 --epochs 2"

# stricter / looser RFT sample selection
!... python scripts/kaggle_run_all.py --from rl --rl-args "--margin 0.05 --conf-slack 0.0"

# SFT overrides are forwarded the same way
!... python scripts/kaggle_run_all.py --sft-args "--epochs 2 --lr 1.5e-4"
```

## What changed in the RL stage

- The SFT adapter is loaded with PEFT (`Reformulator(adapter_path=...)`), and
  RL only updates the ~2.36M LoRA weights.
- Sampling and teacher-forced scoring are batched across 8 questions per
  `generate()` / forward pass, which is what keeps RL to minutes on a T4.
- RFT updates only on rewrites that (a) pass the validity gate, (b) score at
  least as well as the original question under the QA model, and (c) are within
  `--conf-slack` of the original span confidence. Batches with no qualifying
  rewrite are skipped instead of pushing every sample down (the failure mode of
  the earlier all-negative REINFORCE runs).
- The validity gate was relaxed to what it is for — stability, not semantic
  judging. It now rejects only unfinished/empty/copy outputs, foreign-script
  spill, Latin/tokenizer noise, changed *known* answer types, and rewrites that
  share no content anchor with the question. The old 55%-anchor + strict
  answer-type rule rejected 4 of 5 genuine SFT paraphrases, which silently made
  RL a no-op in earlier attempts.

## SFT diagnostics (still valid)

1. `first loss ≈1.2–2.7` — the checkpoint's real weights (including the tied
   `model.shared.weight` embedding) are loaded. A first loss around 9–11 with
   the decoder outputting the same string for every question means the
   embedding table was re-initialized; the script pins `transformers==4.57.6`
   and explicitly re-ties the shared embedding as a guard.
2. `baseline` and `delta_*` in `sft_metrics.json` measure improvement on
   identical held-out questions.
3. Degenerate output is detected via `empty_rate`, `exact_copy_rate`, and
   `distinct_best_ratio` (≈0 means every question collapsed to the same text).
4. `optimizer_steps` must equal `train_steps`; the script hard-fails if zero
   optimizer updates happened, so a no-op adapter cannot masquerade as success.

Script purity is reported, not enforced: raw model text is printed with
`te/ta/kn/deva` character counts so Tamil/Kannada/Devanagari spill is visible
in the numbers rather than hidden by a filter.

## If something fails

Every stage writes `run.log` and `run_status.json` into its output directory
**at startup**, so a crash always leaves the last completed stage and the
traceback. For RL, look at
`models/indicbartss-reformulator-lora-rl/run.log`. Common causes:

- `No qualifying rewrite was found` — too strict selection or too few
  candidates; lower `--margin`, raise `--conf-slack`, or raise `--candidates`.
- CUDA OOM during RL — lower `--generate-batch` (generation batch) or
  `--candidates`.
- `SFT adapter not found` — the SFT stage did not finish; run without
  `--from rl`, or check the SFT `run_status.json`.
