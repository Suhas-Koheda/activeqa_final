# Running the SFT pipeline on Kaggle

One command does everything: dependency install, model download, tokenization
(cached), LoRA training, held-out evaluation, and an unfiltered raw-output demo.

## Notebook settings

- **Accelerator:** GPU T4 ×2 or P100 (the script uses the first GPU)
- **Internet:** on (model download from Hugging Face)
- **Persistence:** none needed — everything lives under `/kaggle/working`

## Cell 1 — get the code

```python
!rm -rf /kaggle/working/activeqa_final
!git clone --depth 1 https://github.com/Suhas-Koheda/activeqa_final.git /kaggle/working/activeqa_final
%cd /kaggle/working/activeqa_final
```

## Cell 2 — run the whole pipeline

```python
!cd /kaggle/working/activeqa_final && python scripts/colab_run_sft.py
```

Expected wall time on a T4: ~5–10 min training (1 epoch, 52k pairs, batch 48,
fp16, length-bucketed) + ~2–4 min held-out generation + under a minute for data
tokenization on a cold cache.

Outputs:

- `models/indicbartss-reformulator-lora/adapter/` — LoRA adapter + tokenizer
- `models/indicbartss-reformulator-lora/sft_metrics.json` — F1, copy rate,
  empty rate, script-spill counts, loss, throughput
- `models/indicbartss-reformulator-lora/sft_samples.json` — every held-out
  question with all sampled candidates and per-sample script counts

## Cell 3 — inspect the raw demo

The script already prints the demo at the end (raw text, no filtering). To see
the saved metrics without scrolling:

```python
import json
m = json.load(open("/kaggle/working/activeqa_final/models/indicbartss-reformulator-lora/sft_metrics.json"))
print(json.dumps(m, indent=2, ensure_ascii=False))
```

## Cell 4 — download the artifacts

```python
import shutil, os
base = "/kaggle/working/activeqa_final/models/indicbartss-reformulator-lora"
shutil.make_archive("/kaggle/working/activeqa_sft_artifacts", "zip", base)
print(os.path.getsize("/kaggle/working/activeqa_sft_artifacts.zip") / 1e6, "MB")
print("Download from the Output panel → activeqa_sft_artifacts.zip")
```

## Useful overrides

```python
# 2 epochs if the first run still copies the question too often
!cd /kaggle/working/activeqa_final && python scripts/colab_run_sft.py --epochs 2 --lr 1.5e-4

# bigger batch if VRAM allows (T4 16 GB handles 64 fine)
!cd /kaggle/working/activeqa_final && python scripts/colab_run_sft.py --batch-size 64

# skip dependency reinstall on a warm session
!cd /kaggle/working/activeqa_final && python scripts/colab_run_sft.py --skip-install
```

## What "stable output" is checked against

`sft_metrics.json` fails loudly (`WARNING: degenerate generations`) if more
than 10% of held-out outputs are empty or more than 70% are verbatim copies of
the input question. Script purity is reported, not enforced: raw model text is
printed with `te/ta/kn/deva` character counts so Tamil/Kannada/Devanagari spill
is visible in the numbers rather than hidden by a filter.
