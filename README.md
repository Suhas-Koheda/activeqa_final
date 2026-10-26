# ActiveQA Telugu

## Current pipeline: LoRA SFT → QA-reward RL (one command)

The production path is now:

```bash
python scripts/kaggle_run_all.py
```

It runs two stages in sequence on a GPU host:

1. **SFT** (`scripts/colab_run_sft.py`) — LoRA fine-tune of
   `ai4bharat/MultiIndicParaphraseGenerationSS` on the 24,883 pairs in
   `data/indic_rag_te/sft_pairs_v2.jsonl`, built with **only** the
   [`nlpctx/telugu-qa-codemixed`](https://huggingface.co/datasets/nlpctx/telugu-qa-codemixed)
   dataset, used as-is — code-mixed rewrites paired within each doc and, when
   complete, into the native-Telugu level-5 rewrite. The dataset content is
   never modified (only truncated/junk cells are skipped), and no rule- or
   regex-generated targets exist anymore,
   with a base-vs-SFT held-out paraphrase evaluation.
   The last verified run reached token-F1 **0.2698 → 0.2907** with 100% of
   outputs ending in a question mark, zero Tamil/Kannada/Devanagari spill, and
   no copies or empties.
2. **RL** (`scripts/kaggle_run_rl.py`) — on-policy rejection-sampling
   fine-tuning (RFT) of that adapter with the frozen
   `l3cube-pune/telugu-question-answering-squad-bert` model as the reward. It
   then answers 200 held-out `dev.jsonl` questions with the original question
   and with the confidence-selected rewrite, for the SFT and RL policies, and
   reports the paired RL − SFT delta with a per-question bootstrap interval.

The evaluation reports the do-nothing baseline (`original_f1`), the deployed
answer-consistent selector, a single-confident-answer corrective selector, an
answer-consensus selector (at least two rewrites must agree on a different
answer), a corrective-margin sweep, a confidence-gate sweep, and signal AUCs
for recoverability. A gain is only claimed when the paired RL − SFT interval
excludes zero **and** the chosen rule beats the original question with a
paired CI above zero. Otherwise `verdict.recommended_action` is `original`:
reformulation is not worth deploying for that policy.

### Measured result (held-out `dev.jsonl`, 200 questions)

| rule | F1 |
| --- | --- |
| original question (do nothing) | **0.7720** |
| SFT + consistent selector (deployed default) | 0.7666 |
| SFT + corrective, best margin (+2.0) | 0.7697 |
| SFT + consensus | 0.7446 |
| oracle over {original, rewrites} | 0.8372 |

The oracle ceiling is real (+0.0652) and concentrated in 27 questions, but no
deployable rule reaches it: at every corrective margin that actually switches,
zero questions improve, and the best achievable rule is abstention. The
question-level signal is weak (AUC 0.699 for low original confidence, 0.628 for
confidence gain), and high confidence gain mostly indicates a *drifted* rewrite
that the QA model answers confidently. RL fine-tuning was also negative
(`selected_f1` 0.7666 → 0.7658 and 0.7657 in two runs). The honest deployment
is therefore to answer the original question; the SFT adapter is kept because
it improves paraphrase quality (token-F1 0.2698 → 0.2907) and is the base for
future work.

Artifacts: `models/indicbartss-reformulator-lora/` (SFT adapter) and
`models/indicbartss-reformulator-lora-rl/` (RL adapter, `qa_eval.json`,
per-question predictions). See `KAGGLE_SFT.md` for notebook cells, expected log
lines, overrides, and failure diagnostics.

The sections below document the earlier full-parameter REINFORCE experiments
and the research/leakage notes; they are kept for context, not as the current
recommended path.

## Telugu reformulation-only REINFORCE

For training the Telugu question rewriter itself, use `scripts/train_telugu_reformulation_rl.py`. This is full-parameter REINFORCE only: no supervised warm-up, no LoRA, no hand-written rewrite templates, and no QA-answer reward. Each sampled output receives a reward from token F1 against the user's reformulation references, while invalid, off-topic, answer-type-changing, foreign-script, and duplicate outputs are penalized. The script filters obvious bad pairs from the 107-question user dataset, splits by source question, writes the filtered JSONL and a rejection report, and keeps the best held-out checkpoint so a later epoch cannot silently replace a better policy.

Run on a Colab T4 after cloning/pulling the repository and enabling GPU:

```bash
python scripts/train_telugu_reformulation_rl.py \
  --config configs/colab_telugu_reformulation_rl.yaml
```

It writes `outputs/telugu_reference_reinforce.pt`, metrics, held-out generations, and filtered data under `data/indic_rag_te/rl_reference_pairs/`. To demo the resulting full model, pass that checkpoint to `scripts/demo_reformulators.py --language te --telugu-base models/hf_fresh/telugu --telugu-checkpoint outputs/telugu_reference_reinforce.pt --question "..."`. Its generated text is model output; the demo does not fill missing candidates with templates. The labels are still user-provided and not independently human verified, so the held-out generations should be reviewed before treating this policy as reliable.

A research implementation of Telugu question reformulation with REINFORCE and a frozen extractive QA model. It runs on CPU for smoke tests and on a Colab GPU for the full experiment. The runtime loads models lazily when scripts execute; no model or dataset download is initiated by installation. This repository does **not** contain completed experiment results.

## Research and data limits

### Leakage-safe Telugu RL data

`data/indic_rag_te/` now contains an 804-row training split and a 200-row development split sampled from the Telugu `train` config of [AI4Bharat Indic-Rag-Suite](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite). Sampling uses seed 42, drops answers not found in their source paragraph, requires contexts of at least 100 characters, deduplicates records, and keeps every Wikipedia article wholly in one split (zero article overlap). The source is synthetic Telugu Wikipedia QA attributed in its records to Meta-Llama-3.3-70B-Instruct; automatic filters do not establish factual correctness, so manually audit before treating it as gold data. Its MIT dataset-card license and sampling details are in `data/indic_rag_te/metadata.json`.

The separate `configs/colab_external_rl.yaml` config trains on this external source and uses the IndicQA Telugu test set only for final held-out evaluation. This avoids the old exploratory practice of training on part of IndicQA's test split. The Colab configs load the generated training rows from the Hub repo [`nlpctx/activeqa-telugu-rl-data`](https://huggingface.co/datasets/nlpctx/activeqa-telugu-rl-data) by default, falling back to the committed `data/indic_rag_te/train.jsonl` when the Hub is unavailable. Run it in Colab with `bash scripts/run_colab.sh configs/colab_external_rl.yaml`. The development split is available for manual tuning and is not mixed into the training retrieval corpus.

To reproduce or extend the subset, run `python scripts/build_rl_dataset.py --train-size 800 --dev-size 200 --seed 42`. Successful Hugging Face API pages are cached under `data/cache/indic_rag_api_pages/` so interrupted runs can reuse them. To regenerate the 100 draft paraphrases with the free OpenCode provider, run `python scripts/generate_reformulations_opencode.py --count 100 --batch-size 10`; it writes `data/indic_rag_te/reformulations_review.jsonl`. Those MiniMax-generated candidates are explicitly drafts requiring human semantic review; they are not used as verified labels or to select answers at inference time.

Publish the prepared dataset under your Hugging Face account by first signing in with a write token using `~/projects/Python/myenv/bin/hf auth login`, then run `~/projects/Python/myenv/bin/python scripts/push_dataset_hf.py`. The upload script creates a public dataset repository named `activeqa-telugu-rl-data` under the authenticated username. No token is stored in the repository.

IndicQA (`ai4bharat/IndicQA`) publishes Telugu in `data/indicqa.te.json` using a SQuAD-style `data/articles/paragraphs/qas` structure; the loader downloads just that language JSON via `huggingface_hub` and flattens it. This bypasses the legacy `IndicQA.py` custom loader, which Datasets 5.x no longer supports. The downloaded JSON is 3,119,401 bytes and has 1,734 flattened QA rows. After dropping rows with empty question/context/answer and exact duplicates, 1,398 remain. There is no official training split in this release. `data.py` deterministically shuffles and splits the remaining test rows into 1,119 exploratory policy-training and 279 held-out rows (seed 42). This is exploratory reuse of benchmark test data, not a valid official benchmark score. Do not publish those evaluation numbers as a clean test result.

The requested QA model is a BERT extractive QA checkpoint. Its Hugging Face repository currently lists a 948 MB safetensors weight file; it takes `(question, context)` tokenizer inputs and returns start/end span logits. The QA model is frozen and evaluated in inference mode. The reformulator repository identifies this as a 0.2B parameter IndicBARTSS/mBART encoder-decoder and documents AlbertTokenizer with `use_fast=False`, `do_lower_case=False`, and Telugu language token `<2te>`. The implementation follows its source suffix, target language decoder start, and slow tokenizer requirements (SentencePiece and protobuf). A LoRA adapter is attached only if compatible attention projections are found. The code fails loudly otherwise. Model compatibility, actual reformulator size, memory and useful Telugu semantic preservation remain to be measured on the target host. Numeric and Latin capitalized entities are screened conservatively, but the resulting signature metric is only a heuristic and does not measure Telugu semantic preservation; manually rate sampled reformulations before claiming semantic validity.

Training reward is `token F1 + 0.1*EM + 0.1*entity signature preservation - invalid penalty`, computed on the **canonical supporting passage only**. BM25 contexts are no longer mixed into the training reward; they are an optional evaluation diagnostic. Each sampled rewrite is centred against the reward of the original question on the same passage, so the advantage measures whether the rewrite beat not rewriting rather than whether the question was easy. Primary evaluation uses the same canonical passage for every method, samples rewrites with the same stochastic policy used in training (rather than beam search), and reports BM25-only retrieval results separately when `evaluation.retrieval_metrics` is enabled. QA candidate and context ranking uses extractive span log-confidence without references; the QA selector can choose the original question as a fallback and rejects rewrites whose extracted answer differs substantially from the original. This is a conservative answer-consistency safeguard, not proof of semantic equivalence. Oracle selection is explicitly offline and uses references. Evaluation predictions include references for manual audit, but those references do not affect deployable candidate or context selection. Metrics are paired over fixed evaluation questions and predictions are saved as JSON.

Reformulation generation explicitly supplies the model card's `<s>` and `</s>` token IDs and defaults to 64 output tokens. Any sampled sequence that reaches the generation limit without `</s>` is treated as incomplete and falls back to the original question for downstream scoring; its sampled token sequence still receives the REINFORCE invalid-action penalty. Validity also requires the rewrite to retain the original question's Telugu interrogative type (which/who/when/where/why/how/how-much/how-many) and its numeric/Latin entity signature, because a paraphrase that changes the answer type is not a valid reformulation.

## Fedora setup

Use the requested environment at `~/projects/Python/myenv` (this assumes it already exists):

```bash
source ~/projects/Python/myenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If it does not exist, create it with `python -m venv ~/projects/Python/myenv` before activation. This project was checked in Python 3.14.3 with CPU PyTorch, Transformers, PEFT, and rank-bm25 installed. Verify that the pinned dependency set installs in your own environment before a long run. Fedora system packages beyond Python build tooling are not required when wheels are available. All scripts force CPU and configure OMP/MKL/PyTorch thread counts.

```bash
python scripts/benchmark_cpu.py --config configs/cpu.yaml
python scripts/prepare_data.py --config configs/smoke_test.yaml
python scripts/train.py --config configs/smoke_test.yaml
python scripts/evaluate.py --config configs/smoke_test.yaml --checkpoint outputs/smoke_policy.pt
python scripts/run_pipeline.py --question "తెలుగులో ఒక ప్రశ్న"
```

## Google Colab GPU run

Clone this repository in a Colab GPU runtime, then run the logged smoke workflow:

```bash
git clone https://github.com/Suhas-Koheda/activeqa_final.git
cd activeqa_final
bash scripts/run_colab.sh
```

The script checks that Colab has an enabled CUDA GPU, keeps Colab's installed CUDA PyTorch, installs the remaining Python dependencies, and runs data preparation, training, and paired evaluation. It defaults to the 16/8 smoke config. It writes a timestamped complete log to `outputs/colab_logs/` and saves the policy and machine-readable metrics under `outputs/`. It also removes an incompatible preinstalled `torchao` version when present; this experiment uses standard LoRA and does not need torchao. The script defaults to regular HTTP Hub downloads (`HF_HUB_DISABLE_XET=1`) and keeps model/data caches in the project directory. Colab storage is temporary; download or copy `outputs/` and cache directories to Drive before ending the session if you need to retain them. If you run the script outside Colab without PyTorch installed, it exits with a clear runtime message rather than installing a potentially incompatible PyTorch build.

The default `bash scripts/run_colab.sh` uses `configs/colab_external_smoke.yaml`, which trains on 16 rows from the separate Indic-Rag-Suite sample and evaluates on 8 held-out IndicQA test questions. Both configs load the generated training split from the Hub repo and fall back to the committed `data/indic_rag_te/train.jsonl` when needed. The smoke run keeps retrieval diagnostics enabled so the full logging path is exercised.

To run the full external training sample and the full IndicQA Telugu test set after the smoke run, use:

```bash
bash scripts/run_colab.sh configs/colab_external_rl.yaml
```

This config uses 804 policy-training examples, four sampled actions, three epochs, LoRA rank 16/alpha 32, learning rate 1e-4, and eight questions per optimizer update (9,648 sampled actions, 303 updates). It evaluates on all 1,398 held-out IndicQA Telugu test questions. Policy log-probabilities and QA scoring are batched across candidates, and the training loop and evaluation both show tqdm progress bars. Use the smoke log to estimate throughput before launching the full run. The earlier 1,119/279 exploratory run trained and evaluated on disjoint portions of IndicQA test data and must not be reported as a clean benchmark.

**Recommended: use the RFT config instead after this REINFORCE run collapses or shows no gain:**

```bash
bash scripts/run_colab.sh configs/colab_external_rft.yaml
```

It samples 4 candidates per question, keeps the best valid rewrite that ties or beats the original question's QA reward, and fine-tunes on that sequence only (rejection-sampling fine-tuning). Two epochs, LoRA rank 8, lr 5e-5, checkpoints every 25 updates. There is also a smoke version, `configs/colab_external_rft_smoke.yaml` (16 train / 8 eval).

The existing `configs/cpu.yaml` and `configs/smoke_test.yaml` remain CPU configurations. CUDA execution is opt-in through the Colab config; requesting CUDA on a machine without an available GPU exits with an explicit error.

Training/evaluation need internet access on first use to cache IndicQA, the generated training split, and both model repositories. HF metadata is cached under `data/hf_home` by default (set `HF_HOME` to override); dataset and model files use the configured `data/cache` and `models` directories. Set `HF_HUB_OFFLINE=1` after caching. The smoke configuration uses 16 policy-training and 8 held-out examples, 4 candidates and 1 epoch. The Hugging Face Xet transfer stalled in one observed run; `HF_HUB_DISABLE_XET=1` let the regular HTTP transfer proceed. Model files require multiple GB of download and cache space. Training saves only the LoRA adapter plus optimizer/RNG state and supports `--resume`; checkpoints are written every 50 optimizer updates in the full config and resume at question granularity.

## REINFORCE

A sampled sequence is drawn with `generate(do_sample=True)`. Its generated token IDs (including the end token) are retained and teacher-forced, in one batched forward over all candidates for a question, to obtain the sum of sampled-token log probabilities. The model is switched to eval mode for that forward so the scored policy matches the dropout-free policy that generated the actions. If output validation fails, its answer score uses the original-question fallback and receives an invalid-action penalty, while its policy gradient still uses the actual generated action. The original question is scored on the same canonical passage and each rewrite's advantage is `reward(rewrite) - reward(original)`, which removes per-question difficulty from the baseline. Advantages are std-scaled within each optimizer batch. Loss is `-(sum logprob * advantage) / batch_size`; gradients are checked, clipped and applied to LoRA parameters. QA scoring for the original question and all candidates is batched into one `answer_batch` call under `torch.inference_mode()`.

### Rejection-sampling fine-tuning (recommended)

The first full REINFORCE run with `lr=1e-4` and rank 16 collapsed during epoch 3 (`valid_generation_rate` fell from 0.58 to 0.06): when every sampled rewrite lost to the original question, the whole batch had negative advantages and the update pushed the policy off its pretrained distribution. `configs/colab_external_rft.yaml` replaces that update with rejection-sampling fine-tuning (RFT): sample `candidates_per_question` rewrites, score them on the canonical passage, pick the best *valid* rewrite that is at least as good as the original question, and teacher-force only that sequence. Questions with no good candidate are skipped, so the policy is never reinforced away from all of its samples. It sets `lr=5e-5`, LoRA rank 8, two epochs, `keep_step_checkpoints: true`, and checkpoints every 25 updates, so any intermediate checkpoint can be evaluated separately with `python scripts/evaluate.py --config configs/colab_external_rft.yaml --checkpoint outputs/external_rft_policy_step50.pt`. `training.min_reward_margin` (default 0) requires the selected rewrite to tie or beat the original reward.

## Artifacts

- `outputs/split_manifest.json`: split IDs and sizes.
- `outputs/train_metrics.json`: per-action reward/advantage/validity, per-update loss, timings and trainable count.
- `outputs/eval_metrics.json`: paired metrics for the pretrained and REINFORCE-trained policies.
- `outputs/eval_pretrained.json` / `outputs/eval_trained.json`: per-question predictions for manual audit.
- `outputs/*.pt`: resumable LoRA checkpoints.

No artifacts are checked in as successful experimental results. `entity_signature_preservation_rate` is only a heuristic proxy, not human-judged semantic preservation. Manually inspect a sample and report harmful reformulations before drawing conclusions.

### Earlier CPU smoke result (seed 42, pre-4-candidate config)

The smoke run was executed in `~/projects/Python/myenv` using CPU PyTorch, 16 training examples, 8 held-out examples, 2 sampled actions per question, and 1 epoch. It predates the current 4-candidate, per-question-baseline objective and is kept only as a reproducibility record; do not compare it directly with the new Colab run. It completed 32 optimizer updates in 196.9 seconds, used 294,912 trainable adapter parameters, peaked at 2,052 MB RSS, and had a 68.8% valid-action rate. The paired evaluation took 181.1 seconds for pretrained reformulation and 182.6 seconds for the trained adapter, with approximately 1,909 MB peak RSS. On this tiny exploratory split, original-question QA had EM 0.0 and F1 0.0417; pretrained and REINFORCE reformulation, QA-score selection, and oracle selection each had EM 0.0 and F1 0.0. The policy did not improve QA. Entity-signature preservation was 100% by the heuristic, while 12.5% of pretrained reformulations were harmful by the answer-F1 comparison. These estimates are not benchmark results because the source only provides a test split. Manually judged Telugu semantic preservation was not measured.
