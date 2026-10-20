# ActiveQA Telugu (CPU research scaffold)

A CPU-only research implementation of Telugu question reformulation with REINFORCE and a frozen extractive QA model. The runtime loads models lazily when scripts execute; no model or dataset download is initiated by installation. This repository does **not** contain completed experiment results.

## Research and data limits

IndicQA (`ai4bharat/IndicQA`) publishes Telugu in `data/indicqa.te.json` using a SQuAD-style `data/articles/paragraphs/qas` structure; the loader downloads just that language JSON via `huggingface_hub` and flattens it. This bypasses the legacy `IndicQA.py` custom loader, which Datasets 5.x no longer supports. The downloaded JSON is 3,119,401 bytes and has 1,734 flattened QA rows. After dropping rows with empty question/context/answer and exact duplicates, 1,398 remain. There is no official training split in this release. `data.py` deterministically shuffles and splits the remaining test rows into 1,119 exploratory policy-training and 279 held-out rows (seed 42). This is exploratory reuse of benchmark test data, not a valid official benchmark score. Do not publish those evaluation numbers as a clean test result.

The requested QA model is a BERT extractive QA checkpoint. Its Hugging Face repository currently lists a 948 MB safetensors weight file; it takes `(question, context)` tokenizer inputs and returns start/end span logits. The QA model is frozen and evaluated in inference mode. The reformulator repository identifies this as a 0.2B parameter IndicBARTSS/mBART encoder-decoder and documents AlbertTokenizer with `use_fast=False`, `do_lower_case=False`, and Telugu language token `<2te>`. The implementation follows its source suffix, target language decoder start, and slow tokenizer requirements (SentencePiece and protobuf). A LoRA adapter is attached only if compatible attention projections are found. The code fails loudly otherwise. Model compatibility, actual reformulator size, memory and useful Telugu semantic preservation remain to be measured on the target host. Numeric and Latin capitalized entities are screened conservatively, but the resulting signature metric is only a heuristic and does not measure Telugu semantic preservation; manually rate sampled reformulations before claiming semantic validity.

Training reward is `token F1 + 0.1*EM + 0.1*entity signature preservation - invalid penalty`. The canonical supporting passage is used in the **training reward** along with training-only BM25 retrieval results. Primary evaluation uses each question's canonical supporting passage for every method, while BM25-only retrieval results are reported separately. QA candidate and context ranking uses extractive span log-confidence without references; the QA selector can choose the original question as a fallback and rejects rewrites whose extracted answer differs substantially from the original. This is a conservative answer-consistency safeguard, not proof of semantic equivalence. Oracle selection is explicitly offline and uses references. Evaluation predictions include references for manual audit, but those references do not affect deployable candidate or context selection. Metrics are paired over fixed evaluation questions and predictions are saved as JSON. Repeated sampling is seeded, but generation is stochastic during training.

Reformulation generation explicitly supplies the model card's `<s>` and `</s>` token IDs and defaults to 64 output tokens. Any sampled sequence that reaches the generation limit without `</s>` is treated as incomplete and falls back to the original question for downstream scoring; its sampled token sequence still receives the REINFORCE invalid-action penalty.

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

The script checks that Colab has an enabled CUDA GPU, keeps Colab's installed CUDA PyTorch, installs the remaining Python dependencies, and runs data preparation, 16/8 smoke training, and paired evaluation using `configs/colab_gpu.yaml`. It writes a timestamped complete log to `outputs/colab_logs/` and saves the policy and machine-readable metrics under `outputs/`. It also removes an incompatible preinstalled `torchao` version when present; this experiment uses standard LoRA and does not need torchao. The script defaults to regular HTTP Hub downloads (`HF_HUB_DISABLE_XET=1`) and keeps model/data caches in the project directory. Colab storage is temporary; download or copy `outputs/` and cache directories to Drive before ending the session if you need to retain them. If you run the script outside Colab without PyTorch installed, it exits with a clear runtime message rather than installing a potentially incompatible PyTorch build.

The existing `configs/cpu.yaml` and `configs/smoke_test.yaml` remain CPU configurations. CUDA execution is opt-in through the Colab config; requesting CUDA on a machine without an available GPU exits with an explicit error.

Training/evaluation need internet access on first use to cache IndicQA and both model repositories. HF metadata is cached under `data/hf_home` by default (set `HF_HOME` to override); dataset and model files use the configured `data/cache` and `models` directories. Set `HF_HUB_OFFLINE=1` after caching. Initial smoke configuration uses 16 policy-training and 8 held-out examples, 2 candidates and 1 epoch. The Hugging Face Xet transfer stalled in one observed run; `HF_HUB_DISABLE_XET=1` let the regular HTTP transfer proceed. Model files require multiple GB of download and cache space. Training saves only the LoRA adapter plus optimizer/baseline state and supports `--resume`.

## REINFORCE

A sampled sequence is drawn with `generate(do_sample=True)`. Its generated token IDs (including the end token) are retained and teacher-forced to obtain the sum of sampled-token log probabilities. If output validation fails, its answer score uses the original-question fallback and receives an invalid-action penalty, while its policy gradient still uses the actual generated action. Detached scalar downstream rewards form an advantage against a moving reward baseline. Loss is `-logprob * advantage`; gradients are checked, clipped and applied to LoRA parameters. QA reward inference is under `torch.inference_mode()`. Results remain exploratory until the smoke run, checkpoint roundtrip, and manual Telugu semantic review complete.

## Artifacts

- `outputs/split_manifest.json`: split IDs and sizes.
- `outputs/train_metrics.json`: per-step reward/loss/time and trainable count.
- `outputs/eval_predictions.json`: paired predictions and metrics.
- `outputs/*.pt`: resumable checkpoints.

No artifacts are checked in as successful experimental results. `entity_signature_preservation_rate` is only a heuristic proxy, not human-judged semantic preservation. Manually inspect a sample and report harmful reformulations before drawing conclusions.

### CPU smoke result (seed 42)

The smoke run was executed in `~/projects/Python/myenv` using CPU PyTorch, 16 training examples, 8 held-out examples, 2 sampled actions per question, and 1 epoch. It completed 32 optimizer updates in 196.9 seconds, used 294,912 trainable adapter parameters, peaked at 2,052 MB RSS, and had a 68.8% valid-action rate. The paired evaluation took 181.1 seconds for pretrained reformulation and 182.6 seconds for the trained adapter, with approximately 1,909 MB peak RSS. On this tiny exploratory split, original-question QA had EM 0.0 and F1 0.0417; pretrained and REINFORCE reformulation, QA-score selection, and oracle selection each had EM 0.0 and F1 0.0. The policy did not improve QA. Entity-signature preservation was 100% by the heuristic, while 12.5% of pretrained reformulations were harmful by the answer-F1 comparison. These estimates are not benchmark results because the source only provides a test split. Manually judged Telugu semantic preservation was not measured.
