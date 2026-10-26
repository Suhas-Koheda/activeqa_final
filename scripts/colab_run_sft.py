#!/usr/bin/env python3
"""Standalone Colab SFT script — LoRA fine-tune IndicBARTSS on Telugu question
reformulation pairs.

Model : ai4bharat/MultiIndicParaphraseGenerationSS
  - 244 M encoder-decoder (same family as IndicBART)
  - Uses AlbertTokenizer with keep_accents=True
  - Handles NATIVE Telugu script directly (no Devanagari conversion needed)
  - Input format : "<s> Question </s> <2te>"  (add_special_tokens=False)
  - Target format: "<2te> Reformulation </s>" (add_special_tokens=False)
  - decoder_start_token_id = lang_token_id(<2te>)

Run in Colab (T4 GPU):
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

from tqdm.auto import tqdm  # noqa: E402

# ── 2. config ─────────────────────────────────────────────────────────────────
MODEL_ID    = "ai4bharat/MultiIndicParaphraseGenerationSS"
MODEL_DIR   = ROOT / "models" / "MultiIndicParaphraseGenerationSS"
ADAPTER_DIR = ROOT / "models" / "indicbartss-reformulator-lora" / "adapter"
DATA_FILE   = ROOT / "data" / "indic_rag_te" / "indic_paraphrase_question_pairs.jsonl"

LANG_TOKEN = "<2te>"
SEED       = 42
MAX_LEN    = 128    # covers most Telugu questions + reformulations
EPOCHS     = 1
BATCH      = 16
LR         = 3e-4
WARMUP     = 0.05
LORA_R     = 16
LORA_ALPHA = 32

random.seed(SEED)
torch.manual_seed(SEED)

# ── 3. download ───────────────────────────────────────────────────────────────
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
print("[3/5] Loading training pairs …", flush=True)

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

from transformers import AutoModelForSeq2SeqLM, AlbertTokenizer
from peft import LoraConfig, TaskType, get_peft_model

# IndicBARTSS requires AlbertTokenizer with keep_accents=True
# This preserves Telugu Unicode codepoints correctly
tok = AlbertTokenizer.from_pretrained(
    MODEL_PATH, do_lower_case=False, use_fast=False, keep_accents=True)

# get special token ids the official way
bos_id = tok._convert_token_to_id_with_added_voc("<s>")
eos_id = tok._convert_token_to_id_with_added_voc("</s>")
pad_id = tok._convert_token_to_id_with_added_voc("<pad>")
lang_id = tok._convert_token_to_id_with_added_voc(LANG_TOKEN)
print(f"  bos={bos_id} eos={eos_id} pad={pad_id} lang(<2te>)={lang_id}", flush=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False

# LoRA — target the attention projections
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["query", "key", "value", "dense"],   # Albert-style names
)
try:
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
except Exception as e:
    # fallback: try MBart-style names
    print(f"  LoRA with Albert names failed ({e}), trying q_proj/k_proj …", flush=True)
    lora_cfg2 = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(model, lora_cfg2)
    model.print_trainable_parameters()

model.to("cuda")
print("[4/5] Model on GPU.\n", flush=True)

# ── 6. tokenize ───────────────────────────────────────────────────────────────
# Official IndicBARTSS format:
#   src input_ids : <s> Q_tokens </s> <2te>   (no auto special tokens)
#   tgt labels    : <2te> P_tokens </s>        (decoder sees <2te> as start)
print("[5/5] Tokenizing …", flush=True)

def encode(q: str, p: str) -> dict | None:
    src_text = f"{q} </s> {LANG_TOKEN}"
    tgt_text = f"{LANG_TOKEN} {p} </s>"
    src_ids = tok(src_text, add_special_tokens=False,
                  max_length=MAX_LEN, truncation=True)["input_ids"]
    tgt_ids = tok(tgt_text, add_special_tokens=False,
                  max_length=MAX_LEN, truncation=True)["input_ids"]
    if not src_ids or not tgt_ids:
        return None
    labels = list(tgt_ids)
    # mask the leading lang token from loss (it's the decoder start, not a target)
    labels[0] = -100
    return {"src": src_ids, "tgt": tgt_ids[:-1], "labels": labels[1:]}
    # decoder_input_ids = tgt[:-1], labels = tgt[1:]  (standard teacher forcing)

def encode_v2(q: str, p: str) -> dict | None:
    """Standard seq2seq: decoder_input_ids=labels shifted right."""
    src_text = f"{q} </s> {LANG_TOKEN}"
    tgt_text = f"{LANG_TOKEN} {p} </s>"
    src_ids = tok(src_text, add_special_tokens=False,
                  max_length=MAX_LEN, truncation=True)["input_ids"]
    tgt_ids = tok(tgt_text, add_special_tokens=False,
                  max_length=MAX_LEN, truncation=True)["input_ids"]
    if len(src_ids) < 2 or len(tgt_ids) < 2:
        return None
    labels = tgt_ids[:]
    # pad positions in labels will be masked during collation
    return {"src": src_ids, "tgt": tgt_ids}

examples = []
for q, p in tqdm(pairs, desc="Tokenizing", unit="pair"):
    e = encode_v2(q, p)
    if e:
        examples.append(e)

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

def collate(batch: list[dict]) -> dict:
    max_src = max(len(e["src"]) for e in batch)
    max_tgt = max(len(e["tgt"]) for e in batch)
    src_ids  = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    src_mask = torch.zeros((len(batch), max_src), dtype=torch.long)
    tgt_ids  = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    labels   = torch.full((len(batch), max_tgt), -100,   dtype=torch.long)
    for i, e in enumerate(batch):
        s, t = len(e["src"]), len(e["tgt"])
        src_ids[i, :s]  = torch.tensor(e["src"])
        src_mask[i, :s] = 1
        tgt_ids[i, :t]  = torch.tensor(e["tgt"])
        # labels = tgt shifted: decoder input is tgt[:-1], label is tgt[1:]
        # here we pass full tgt as decoder_input_ids; HF model handles shift
        lbl = e["tgt"][:]
        # mask padding in labels
        for j in range(t):
            if lbl[j] != pad_id:
                labels[i, j] = lbl[j]
    return {"src_ids": src_ids, "src_mask": src_mask,
            "tgt_ids": tgt_ids, "labels": labels}

started   = time.perf_counter()
step      = 0
last_loss = 0.0

print(f"Training: {EPOCHS} epoch × {steps_per_epoch} steps = {total_steps} total\n", flush=True)

for epoch in range(EPOCHS):
    random.shuffle(examples)
    model.train()
    pbar = tqdm(range(0, len(examples), BATCH),
                total=steps_per_epoch,
                desc=f"Epoch {epoch+1}/{EPOCHS}",
                unit="step")
    for batch_start in pbar:
        chunk = examples[batch_start: batch_start + BATCH]
        b = collate(chunk)
        out = model(
            input_ids=b["src_ids"].to("cuda"),
            attention_mask=b["src_mask"].to("cuda"),
            decoder_input_ids=b["tgt_ids"].to("cuda"),
            labels=b["labels"].to("cuda"),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
        step += 1
        last_loss = float(out.loss.item())
        peak = torch.cuda.max_memory_allocated() / 1e9
        pbar.set_postfix(loss=f"{last_loss:.4f}", peak_gb=f"{peak:.1f}")

elapsed = time.perf_counter() - started
print(f"\nTraining done: {step} steps  {elapsed/60:.1f} min  final loss={last_loss:.4f}\n", flush=True)

# ── 8. save ───────────────────────────────────────────────────────────────────
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_DIR))
tok.save_pretrained(str(ADAPTER_DIR))
(ADAPTER_DIR.parent / "sft_lora_metrics.json").write_text(
    json.dumps({"model": MODEL_ID, "steps": step, "examples": len(examples),
                "seconds": round(elapsed, 1), "final_loss": last_loss}, indent=2),
    encoding="utf-8")
print(f"Adapter : {ADAPTER_DIR}", flush=True)

# ── 9. demo ───────────────────────────────────────────────────────────────────
print("\n── Demo ──────────────────────────────────────────────────────────────")
model.eval()

TEST_QUESTIONS = [
    "భారతదేశ రాజధాని ఏది?",
    "సౌర వ్యవస్థలో సూర్యునికి అతి దగ్గరగా ఉన్న గ్రహం ఏది?",
    "భారతదేశ మొదటి మహిళా ప్రధానమంత్రి ఎవరు?",
]

for q in TEST_QUESTIONS:
    src_text = f"{q} </s> {LANG_TOKEN}"
    inp = tok(src_text, add_special_tokens=False,
              return_tensors="pt", max_length=MAX_LEN, truncation=True)
    inp = {k: v.to("cuda") for k, v in inp.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        out_ids = model.generate(
            **inp,
            use_cache=True,
            num_beams=4,
            num_return_sequences=3,
            max_length=MAX_LEN,
            min_length=5,
            early_stopping=True,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            decoder_start_token_id=lang_id,   # <2te> starts the decoder
        )
    print(f"\nQ: {q}")
    for i, seq in enumerate(out_ids, 1):
        decoded = tok.decode(seq, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False).strip()
        print(f"  {i}. {decoded}")

print("\nDone.")
