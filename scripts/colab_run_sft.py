#!/usr/bin/env python3
"""Standalone Colab script: SFT-trains Sarvam-1 on Telugu paraphrase pairs,
saves the LoRA adapter, and runs a 3-question demo.

Run the entire cell below in Colab (T4 GPU runtime):

    !cd /content/activeqa_final && python scripts/colab_run_sft.py

No bash wrapper needed. All steps are inline.
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

# ── 0. basic sanity ──────────────────────────────────────────────────────────
import torch

if not torch.cuda.is_available():
    sys.exit("ERROR: No GPU found. Runtime > Change runtime type > T4 GPU, then rerun.")

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"GPU      : {torch.cuda.get_device_name(0)}")
print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
print(f"Working  : {ROOT}")

# ── 1. install deps ───────────────────────────────────────────────────────────
print("\n[1/5] Installing dependencies …", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40,<5",
    "datasets>=2.19",
    "peft>=0.13",
    "rank-bm25>=0.2.2",
    "PyYAML>=6",
    "psutil>=5.9",
    "tqdm>=4.66",
    "safetensors>=0.4",
    "scikit-learn>=1.3",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25",
    "huggingface_hub>=0.23",
])
print("[1/5] Done.", flush=True)

# ── 2. download model (skip if already on disk) ───────────────────────────────
MODEL_ID   = "sarvamai/sarvam-1"
MODEL_DIR  = ROOT / "models" / "sarvam-1"
ADAPTER_DIR = ROOT / "models" / "sarvam-reformulator-lora" / "adapter"
DATA_FILE  = ROOT / "data" / "indic_rag_te" / "indic_paraphrase_question_pairs.jsonl"

def _weights_present(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.safetensors"))

if _weights_present(MODEL_DIR):
    print(f"\n[2/5] Model weights already at {MODEL_DIR}, skipping download.", flush=True)
    MODEL_PATH = str(MODEL_DIR)
else:
    print(f"\n[2/5] Downloading {MODEL_ID} …", flush=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    MODEL_PATH = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[2/5] Downloaded to {MODEL_PATH}", flush=True)

# ── 3. load data ──────────────────────────────────────────────────────────────
print(f"\n[3/5] Loading training pairs from {DATA_FILE} …", flush=True)

MAX_PER_QUESTION = 2
MAX_TOKENS       = 192
SEED             = 42
random.seed(SEED)
torch.manual_seed(SEED)

def load_pairs(path: Path, max_per_q: int) -> list[tuple[str, str]]:
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
            if not q or not refs:
                continue
            for r in refs[:max_per_q]:
                r = r.strip()
                if r and r != q:
                    pairs.append((q, r))
    return pairs

pairs = load_pairs(DATA_FILE, MAX_PER_QUESTION)
random.shuffle(pairs)
print(f"[3/5] Loaded {len(pairs):,} pairs.", flush=True)

# ── 4. SFT training ───────────────────────────────────────────────────────────
print(f"\n[4/5] Loading model and tokenizer from {MODEL_PATH} …", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model.to("cuda")

# tokenize examples — loss only on the paraphrase tokens
def make_example(q: str, p: str) -> dict | None:
    prompt_ids = tok(f"ప్రశ్న: {q}\nపునర్రచన:", add_special_tokens=False)["input_ids"]
    target_ids = tok(f" {p.strip()}\n",          add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(target_ids) > MAX_TOKENS:
        return None
    return {
        "input_ids": prompt_ids + target_ids,
        "labels":    [-100] * len(prompt_ids) + target_ids,
    }

examples = [e for q, p in pairs if (e := make_example(q, p)) is not None]
skipped = len(pairs) - len(examples)
if skipped:
    print(f"[4/5] Skipped {skipped:,} pairs exceeding {MAX_TOKENS} tokens.", flush=True)
print(f"[4/5] Training on {len(examples):,} examples.", flush=True)

# hyper-params
EPOCHS     = 1
MICRO      = 4
ACCUM      = 4
LR         = 1e-4
WARMUP     = 0.03

steps_per_epoch = max(1, math.ceil(len(examples) / (MICRO * ACCUM)))
total_steps     = steps_per_epoch * EPOCHS
warmup_steps    = max(1, int(total_steps * WARMUP))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

def lr_lambda(s: int) -> float:
    if s < warmup_steps:
        return s / warmup_steps
    return max(0.1, 0.5 * (1 + math.cos(
        math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
pad_id    = tok.pad_token_id
started   = time.perf_counter()
step      = 0
last_loss = 0.0

print(f"[4/5] Training: {total_steps} optimizer steps …", flush=True)

for epoch in range(EPOCHS):
    random.shuffle(examples)
    model.train()
    for batch_start in range(0, len(examples), MICRO * ACCUM):
        chunk     = examples[batch_start: batch_start + MICRO * ACCUM]
        max_blen  = max(len(e["input_ids"]) for e in chunk)
        input_ids = torch.full((len(chunk), max_blen), pad_id, dtype=torch.long)
        attention  = torch.zeros((len(chunk), max_blen), dtype=torch.long)
        labels_t   = torch.full((len(chunk), max_blen), -100, dtype=torch.long)
        for i, e in enumerate(chunk):
            n = len(e["input_ids"])
            input_ids[i, :n] = torch.tensor(e["input_ids"])
            attention[i,  :n] = 1
            labels_t[i,   :n] = torch.tensor(e["labels"])
        input_ids = input_ids.to("cuda")
        attention  = attention.to("cuda")
        labels_t   = labels_t.to("cuda")
        out = model(input_ids=input_ids, attention_mask=attention, labels=labels_t)
        (out.loss / ACCUM).backward()
        if (batch_start // MICRO) % ACCUM == ACCUM - 1 or batch_start + MICRO >= len(examples):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
            step += 1
            last_loss = float(out.loss.item())
            if step % 50 == 0 or step == 1:
                peak = torch.cuda.max_memory_allocated() / 1e9
                rate = step / max(1e-9, time.perf_counter() - started)
                print(f"  step {step:4d}/{total_steps}  loss={last_loss:.4f}"
                      f"  {rate:.2f} it/s  peak={peak:.2f} GiB", flush=True)

elapsed = time.perf_counter() - started
print(f"[4/5] Training done: {step} steps, {elapsed/60:.1f} min, final loss={last_loss:.4f}", flush=True)

# save adapter
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_DIR))
tok.save_pretrained(str(ADAPTER_DIR))
metrics = {
    "steps": step, "examples": len(examples),
    "seconds": round(elapsed, 1), "final_loss": last_loss,
    "base_model": MODEL_ID, "prompt_style": "completion",
}
(ADAPTER_DIR.parent / "sft_lora_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8")
print(f"[4/5] Adapter saved to {ADAPTER_DIR}", flush=True)

# ── 5. demo ───────────────────────────────────────────────────────────────────
print("\n[5/5] Demo — loading adapter for inference …", flush=True)

model.eval()

TEST_QUESTIONS = [
    "భారతదేశ రాజధాని ఏది?",
    "సౌర వ్యవస్థలో సూర్యునికి అతి దగ్గరగా ఉన్న గ్రహం ఏది?",
    "భారతదేశ మొదటి మహిళా ప్రధానమంత్రి ఎవరు?",
]

for q in TEST_QUESTIONS:
    prompt = f"ప్రశ్న: {q}\nపునర్రచన:"
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            do_sample=True, temperature=0.7, top_p=0.9,
            max_new_tokens=64, num_return_sequences=3,
            pad_token_id=tok.eos_token_id,
        )
    print(f"\nQ: {q}")
    for i, seq in enumerate(out_ids, 1):
        gen = tok.decode(seq[inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        print(f"  {i}. {gen}")

print("\n[5/5] Done.")
print(f"Adapter  : {ADAPTER_DIR}")
print(f"Metrics  : {ADAPTER_DIR.parent / 'sft_lora_metrics.json'}")
