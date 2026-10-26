#!/usr/bin/env python3
"""Standalone Colab SFT script.

Model  : ai4bharat/indic-bert  (271 M params, Telugu-native, trains fast on T4)
Task   : seq2seq question reformulation via encoder-decoder LoRA
Output : models/indic-bert-reformulator-lora/adapter/

Run in Colab:
    !cd /content/activeqa_final && git pull && python scripts/colab_run_sft.py
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# ── 0. GPU check ──────────────────────────────────────────────────────────────
import torch

if not torch.cuda.is_available():
    sys.exit("ERROR: No GPU. Runtime > Change runtime type > T4 GPU, then rerun.")

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
print(f"Working : {ROOT}\n")

# ── 1. deps ───────────────────────────────────────────────────────────────────
print("[1/5] Installing dependencies …", flush=True)
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40,<5",
    "peft>=0.13",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25",
    "tqdm>=4.66",
    "huggingface_hub>=0.23",
], stdout=subprocess.DEVNULL)
print("[1/5] Done.\n", flush=True)

from tqdm.auto import tqdm  # noqa: E402  (needs pip install first)

# ── 2. model choice ───────────────────────────────────────────────────────────
# ai4bharat/IndicBART  — 244 M encoder-decoder, Telugu token <2te>,
# trains an epoch on 50k pairs in ~25 min on T4.
# Much faster than Sarvam-1 (2.5 B) while still being Telugu-native.
MODEL_ID  = "ai4bharat/IndicBART"
MODEL_DIR = ROOT / "models" / "indicbart"
ADAPTER_DIR = ROOT / "models" / "indic-bert-reformulator-lora" / "adapter"
DATA_FILE   = ROOT / "data" / "indic_rag_te" / "indic_paraphrase_question_pairs.jsonl"

SEED       = 42
MAX_SRC    = 64
MAX_TGT    = 64
EPOCHS     = 1
BATCH      = 16          # fits easily in 14 GB with a 244 M model
LR         = 3e-4
WARMUP     = 0.05
LORA_R     = 16
LORA_ALPHA = 32

random.seed(SEED)
torch.manual_seed(SEED)

# ── 3. download ───────────────────────────────────────────────────────────────
def _weights_present(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.bin")) or any(d.glob("*.safetensors")) if d.is_dir() else False

if MODEL_DIR.is_dir() and (list(MODEL_DIR.glob("*.bin")) or list(MODEL_DIR.glob("*.safetensors"))):
    print(f"[2/5] Model already at {MODEL_DIR}, skipping download.\n", flush=True)
    MODEL_PATH = str(MODEL_DIR)
else:
    print(f"[2/5] Downloading {MODEL_ID} (~1 GB) …", flush=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    MODEL_PATH = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[2/5] Downloaded to {MODEL_PATH}\n", flush=True)

# ── 4. load data ──────────────────────────────────────────────────────────────
print(f"[3/5] Loading training pairs …", flush=True)

def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = (rec.get("original_question") or rec.get("question") or "").strip()
            refs = rec.get("reformulations") or []
            if not refs:
                p = rec.get("paraphrase", "")
                if p:
                    refs = [p]
            for r in refs[:2]:
                r = r.strip()
                if r and r != q:
                    pairs.append((q, r))
    return pairs

pairs = load_pairs(DATA_FILE)
random.shuffle(pairs)
print(f"[3/5] {len(pairs):,} pairs loaded.\n", flush=True)

# ── 5. model + LoRA ───────────────────────────────────────────────────────────
print(f"[4/5] Loading {MODEL_ID} …", flush=True)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# IndicBART needs src/tgt language tokens
SRC_LANG = "<2te>"   # Telugu source
TGT_LANG = "<2te>"   # Telugu target

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,   # IndicBART fp32 is fine at 244 M
    low_cpu_mem_usage=True,
)

# LoRA on encoder + decoder attention projections
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model.to("cuda")
print(f"[4/5] Model on GPU.\n", flush=True)

# ── 6. tokenize ───────────────────────────────────────────────────────────────
print("[5/5] Tokenizing …", flush=True)

def encode(q: str, p: str):
    src = tok(
        q, max_length=MAX_SRC, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    with tok.as_target_tokenizer():
        tgt = tok(
            p, max_length=MAX_TGT, truncation=True,
            padding="max_length", return_tensors="pt",
        )
    labels = tgt["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    return {
        "input_ids":      src["input_ids"].squeeze(0),
        "attention_mask": src["attention_mask"].squeeze(0),
        "labels":         labels.squeeze(0),
    }

# set language tokens
tok.src_lang = SRC_LANG
if hasattr(tok, "tgt_lang"):
    tok.tgt_lang = TGT_LANG

examples = []
for q, p in tqdm(pairs, desc="Tokenizing", unit="pair"):
    try:
        examples.append(encode(q, p))
    except Exception:
        pass

print(f"[5/5] {len(examples):,} examples ready.\n", flush=True)

# ── 7. training ───────────────────────────────────────────────────────────────
steps_per_epoch = math.ceil(len(examples) / BATCH)
total_steps     = steps_per_epoch * EPOCHS
warmup_steps    = max(1, int(total_steps * WARMUP))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

def lr_lambda(s: int) -> float:
    if s < warmup_steps:
        return s / warmup_steps
    progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

started   = time.perf_counter()
step      = 0
last_loss = 0.0

print(f"Training: {EPOCHS} epoch(s) × {steps_per_epoch} steps = {total_steps} total", flush=True)

for epoch in range(EPOCHS):
    random.shuffle(examples)
    model.train()

    pbar = tqdm(
        range(0, len(examples), BATCH),
        total=steps_per_epoch,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        unit="step",
    )

    for batch_start in pbar:
        chunk = examples[batch_start: batch_start + BATCH]

        input_ids = torch.stack([e["input_ids"]      for e in chunk]).to("cuda")
        attn_mask = torch.stack([e["attention_mask"] for e in chunk]).to("cuda")
        labels    = torch.stack([e["labels"]         for e in chunk]).to("cuda")

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
        step += 1
        last_loss = float(out.loss.item())

        peak = torch.cuda.max_memory_allocated() / 1e9
        pbar.set_postfix(loss=f"{last_loss:.4f}", peak_gb=f"{peak:.1f}")

elapsed = time.perf_counter() - started
print(f"\nTraining done: {step} steps  {elapsed/60:.1f} min  final loss={last_loss:.4f}", flush=True)

# ── 8. save ───────────────────────────────────────────────────────────────────
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_DIR))
tok.save_pretrained(str(ADAPTER_DIR))
metrics = {
    "model": MODEL_ID,
    "steps": step, "examples": len(examples),
    "seconds": round(elapsed, 1), "final_loss": last_loss,
}
(ADAPTER_DIR.parent / "sft_lora_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8")
print(f"Adapter  : {ADAPTER_DIR}", flush=True)
print(f"Metrics  : {ADAPTER_DIR.parent / 'sft_lora_metrics.json'}", flush=True)

# ── 9. demo ───────────────────────────────────────────────────────────────────
print("\n── Demo ──────────────────────────────────────────────────────────────")
model.eval()

TEST_QUESTIONS = [
    "భారతదేశ రాజధాని ఏది?",
    "సౌర వ్యవస్థలో సూర్యునికి అతి దగ్గరగా ఉన్న గ్రహం ఏది?",
    "భారతదేశ మొదటి మహిళా ప్రధానమంత్రి ఎవరు?",
]

forced_bos = tok.lang_code_to_id.get(TGT_LANG) if hasattr(tok, "lang_code_to_id") else None

for q in TEST_QUESTIONS:
    enc = tok(q, return_tensors="pt", max_length=MAX_SRC, truncation=True).to("cuda")
    # drop keys that seq2seq generate() doesn't accept
    enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    gen_kwargs = dict(
        **enc,
        max_new_tokens=MAX_TGT,
        num_beams=4,
        num_return_sequences=3,
        early_stopping=True,
    )
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos
    with torch.no_grad():
        out_ids = model.generate(**gen_kwargs)
    print(f"\nQ: {q}")
    for i, seq in enumerate(out_ids, 1):
        print(f"  {i}. {tok.decode(seq, skip_special_tokens=True).strip()}")

print("\nDone.")
