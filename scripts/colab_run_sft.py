#!/usr/bin/env python3
"""One-command Telugu reformulator SFT: install → data → train → eval → demo.

Model : ai4bharat/MultiIndicParaphraseGenerationSS (mBART, 244M)
  - AlbertTokenizer(keep_accents=True) handles native Telugu script
  - source: "<s-free> Question </s> <2te>"   (add_special_tokens=False)
  - target: "Reformulation </s>"             (no <2te>; it is the decoder start)
  - decoder_start_token_id = id("<2te>")

Teacher forcing (the bug that broke the previous run):
  HF shifts `labels` right internally ONLY when `decoder_input_ids` is None.
  The old script passed decoder_input_ids == labels, so every position was
  asked to predict the token it had just been fed — an identity copy task that
  collapsed loss to ~1e-4 and produced empty/repeating generations. Here we
  pass `labels` only, with config.decoder_start_token_id = id("<2te>").

Speed: fp16 autocast + GradScaler (T4), length-bucketed dynamic padding,
AdamW over LoRA parameters only, tokenized dataset cached on disk.
Everything (deps, download, train, held-out eval, raw demo) is one command:

    python scripts/colab_run_sft.py                 # full run on Colab T4
    python scripts/colab_run_sft.py --smoke         # tiny CPU sanity run locally
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

MODEL_ID = "ai4bharat/MultiIndicParaphraseGenerationSS"
MODEL_DIR = ROOT / "models" / "MultiIndicParaphraseGenerationSS"
ADAPTER_DIR = ROOT / "models" / "indicbartss-reformulator-lora" / "adapter"
DATA_FILE = ROOT / "data" / "indic_rag_te" / "indic_paraphrase_question_pairs.jsonl"
CACHE_DIR = ROOT / "models" / ".sft_cache"

LANG_TOKEN = "<2te>"
TAMIL = re.compile(r"[\u0B80-\u0BFF]")
KANNADA = re.compile(r"[\u0C80-\u0CFF]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
TELUGU = re.compile(r"[\u0C00-\u0C7F]")


def log(message: str) -> None:
    print(message, flush=True)
    path = LOG_HANDLE["path"]
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(str(message) + "\n")


LOG_HANDLE: dict[str, Path | None] = {"path": None}


def record_status(path: Path, stage: str, **fields) -> None:
    """Persist progress so a crashed run still leaves a diagnosable trail."""
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA_FILE))
    ap.add_argument("--out", default=str(ADAPTER_DIR))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-src", type=int, default=96)
    ap.add_argument("--max-tgt", type=int, default=96)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eval-questions", type=int, default=60)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the pre-SFT held-out eval (saves a few minutes)")
    ap.add_argument("--lora-mlp", action="store_true",
                    help="also adapt fc1/fc2 (more capacity, more VRAM)")
    ap.add_argument("--amp", default="auto", choices=("auto", "fp16", "fp32"),
                    help="auto probes fp16 and falls back to fp32 on non-finite loss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU-friendly sanity run: 256 pairs, 12 steps, no install")
    return ap.parse_args()


# ── install ──────────────────────────────────────────────────────────────────
def install_deps() -> None:
    log("[1/7] Installing dependencies (torchao is pinned too old by Colab for peft) …")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        # Pin: transformers 5.x pre-releases satisfy "<5" on Kaggle and load this
        # checkpoint without aliasing model.shared.weight to the encoder/decoder/
        # lm_head embeddings (they end up random, loss plateaus ~9, output is
        # the same frequent-token garbage for every question).
        "transformers==4.57.6", "peft>=0.13,<0.22", "sentencepiece>=0.2.0",
        "protobuf>=4.25", "tqdm>=4.66", "huggingface_hub>=0.23,<1.0",
    ], stdout=subprocess.DEVNULL)
    log("[1/7] Done.")


# ── data ─────────────────────────────────────────────────────────────────────
def load_pairs(path: Path, max_per_question: int = 2) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            question = (rec.get("original_question") or rec.get("question") or "").strip()
            refs = rec.get("reformulations") or []
            if not refs and rec.get("paraphrase"):
                refs = [rec["paraphrase"]]
            refs = [r.strip() for r in refs[:max_per_question]
                    if r and r.strip() and r.strip() != question]
            if question and refs:
                records.append({"question": question, "refs": refs})
    return records


def split_questions(records: list[dict], eval_questions: int, seed: int):
    rng = random.Random(seed)
    order = list(records)
    rng.shuffle(order)
    eval_set = order[:eval_questions]
    train_set = order[eval_questions:]
    return train_set, eval_set


# ── tokenization (cached) ────────────────────────────────────────────────────
def cache_key(args: argparse.Namespace) -> str:
    stat = Path(args.data).stat()
    raw = f"{args.data}|{stat.st_size}|{int(stat.st_mtime)}|{args.max_src}|{args.max_tgt}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def tokenize_split(tok, split: list[dict], args: argparse.Namespace,
                   eos_vocab_id: int) -> list[dict]:
    lang_id = tok._convert_token_to_id_with_added_voc(LANG_TOKEN)
    examples = []
    for rec in split:
        src_ids = tok(f"{rec['question']} </s> {LANG_TOKEN}", add_special_tokens=False,
                      max_length=args.max_src, truncation=True)["input_ids"]
        if len(src_ids) < 3 or lang_id not in src_ids:
            continue
        targets = []
        for ref in rec["refs"]:
            # No <2te> here: it is supplied as decoder_start_token_id.
            ids = tok(f"{ref} </s>", add_special_tokens=False,
                      max_length=args.max_tgt, truncation=True)["input_ids"]
            if len(ids) >= 3 and ids[-1] == eos_vocab_id:
                targets.append(ids)
        if targets:
            examples.append({"src": src_ids, "targets": targets})
    return examples


# ── eval helpers ─────────────────────────────────────────────────────────────
def script_stats(text: str) -> dict:
    return {"te": len(TELUGU.findall(text)), "ta": len(TAMIL.findall(text)),
            "kn": len(KANNADA.findall(text)), "hi_deva": len(DEVANAGARI.findall(text))}


def token_f1(candidate: str, reference: str) -> float:
    cand = candidate.split()
    ref = reference.split()
    if not cand or not ref:
        return 0.0
    common = 0
    remaining = list(ref)
    for word in cand:
        if word in remaining:
            remaining.remove(word)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(cand)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def flatten_examples(examples: list[dict]) -> list[dict]:
    flat = []
    for example in examples:
        for target in example["targets"]:
            flat.append({"src": example["src"], "tgt": target})
    return flat


def make_batches(flat: list[dict], batch_size: int, seed: int, shuffle: bool) -> list[list[dict]]:
    rng = random.Random(seed)
    order = sorted(range(len(flat)), key=lambda i: (len(flat[i]["src"]),
                                                    len(flat[i]["tgt"])))
    if shuffle:
        # length-bucketed shuffle: permute inside windows, then permute buckets
        window = max(1, batch_size * 8)
        chunks = [order[i:i + window] for i in range(0, len(order), window)]
        for chunk in chunks:
            rng.shuffle(chunk)
        rng.shuffle(chunks)
        order = [i for chunk in chunks for i in chunk]
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def collate(batch, pad_id, device):
    import torch
    max_src = max(len(e["src"]) for e in batch)
    max_tgt = max(len(e["tgt"]) for e in batch)
    src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    src_mask = torch.zeros((len(batch), max_src), dtype=torch.long)
    labels = torch.full((len(batch), max_tgt), -100, dtype=torch.long)
    for row, example in enumerate(batch):
        s, t = len(example["src"]), len(example["tgt"])
        src[row, :s] = torch.tensor(example["src"], dtype=torch.long)
        src_mask[row, :s] = 1
        labels[row, :t] = torch.tensor(example["tgt"], dtype=torch.long)
    return src.to(device), src_mask.to(device), labels.to(device)


def restore_shared_embeddings(model, model_path: str) -> None:
    """Re-tie the single checkpoint embedding into encoder/decoder/lm_head.

    The checkpoint stores only ``model.shared.weight`` (64015 × 1024). Some
    transformers builds (notably 5.x pre-releases on Kaggle, which satisfy a
    ``<5`` spec) instantiate ``encoder.embed_tokens``, ``decoder.embed_tokens``
    and ``lm_head`` as fresh random tensors instead of aliasing the shared
    table, and LoRA never trains them — loss then plateaus around 9 and every
    question decodes to the same frequent-token string.
    """
    import torch
    from pathlib import Path

    shared = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = Path(model_path) / name
        if not path.exists():
            continue
        if name.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(str(path), framework="pt") as handle:
                if "model.shared.weight" in handle.keys():
                    shared = handle.get_tensor("model.shared.weight")
        else:
            state = torch.load(str(path), map_location="cpu")
            shared = state.get("model.shared.weight")
        if shared is not None:
            break
    if shared is None:
        log("      WARNING: could not find model.shared.weight in the checkpoint; "
            "keeping the loader's embeddings")
        return

    mismatched = []
    with torch.no_grad():
        model.model.shared.weight.copy_(shared)
        for attribute in ("encoder", "decoder"):
            holder = getattr(model.model, attribute)
            if not torch.equal(holder.embed_tokens.weight, shared):
                mismatched.append(attribute)
            holder.embed_tokens.weight.copy_(shared)
        if getattr(model, "lm_head", None) is not None:
            if not torch.equal(model.lm_head.weight, shared):
                mismatched.append("lm_head")
            model.lm_head.weight.copy_(shared)

    # Keep them the same Parameter objects so any later load/save stays tied.
    model.model.encoder.embed_tokens = model.model.shared
    model.model.decoder.embed_tokens = model.model.shared
    if getattr(model, "lm_head", None) is not None:
        model.lm_head.weight = model.model.shared.weight
    model.config.tie_word_embeddings = True
    log(f"      embeddings tied to checkpoint shared table"
        + (f" (repaired random: {', '.join(mismatched)})" if mismatched else ""))


def decode_candidate(tok, ids, lang_id: int, eos_id: int) -> str:
    """Decode one generated sequence without the decoder-start or trailing EOS."""
    tokens = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    if tokens and tokens[0] == lang_id:
        tokens = tokens[1:]
    if eos_id in tokens:
        tokens = tokens[: tokens.index(eos_id)]
    return tok.decode(tokens, skip_special_tokens=True,
                      clean_up_tokenization_spaces=False).strip()


def generate_one(model, tok, question, lang_id, bos_id, eos_id, max_len, device):
    import torch
    src = tok(f"{question} </s> {LANG_TOKEN}", add_special_tokens=False,
              return_tensors="pt", max_length=max_len, truncation=True)
    src = {k: v.to(device) for k, v in src.items() if k in ("input_ids", "attention_mask")}
    generate_kwargs = dict(
        use_cache=True, num_beams=5, num_return_sequences=3,
        max_length=max(16, max_len), min_length=4, early_stopping=True,
        no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3,
        length_penalty=1.0,
        pad_token_id=tok.pad_token_id, bos_token_id=bos_id, eos_token_id=eos_id,
        decoder_start_token_id=lang_id,
    )
    with torch.inference_mode():
        out = model.generate(**src, **generate_kwargs)
    return [decode_candidate(tok, ids, lang_id, eos_id) for ids in out]


def evaluate(model, tok, eval_records, args, lang_id, bos_id, eos_id,
             device) -> tuple[dict, list[dict]]:
    """Generate 3 beam candidates per held-out question and summarize them."""
    import torch
    model.eval()
    rows = []
    for rec in eval_records[: args.eval_questions]:
        candidates = generate_one(model, tok, rec["question"], lang_id,
                                  bos_id, eos_id, args.max_tgt, device)
        best = max(candidates, key=lambda c: max(token_f1(c, r) for r in rec["refs"]))
        rows.append({
            "question": rec["question"],
            "references": rec["refs"],
            "candidates": candidates,
            "best": best,
            "f1": max(token_f1(best, r) for r in rec["refs"]),
            "copy": best.strip() == rec["question"].strip(),
            "empty": not best.strip(),
            "ends_question": best.strip().endswith("?"),
            "scripts": script_stats(best),
        })

    n = max(1, len(rows))
    summary = {
        "eval_questions": len(rows),
        "token_f1": round(sum(r["f1"] for r in rows) / n, 4),
        "exact_copy_rate": round(sum(r["copy"] for r in rows) / n, 4),
        "empty_rate": round(sum(r["empty"] for r in rows) / n, 4),
        "ends_with_question_mark": round(sum(r["ends_question"] for r in rows) / n, 4),
        # Collapse detector: if every question produced the same "best" output,
        # this ratio is ~0 regardless of what the text looks like.
        "distinct_best_ratio": round(len({r["best"] for r in rows}) / n, 4),
        "mean_unique_candidates": round(
            sum(len(set(r["candidates"])) for r in rows) / n, 2),
        "tamil_spill_outputs": sum(1 for r in rows if r["scripts"]["ta"]),
        "kannada_spill_outputs": sum(1 for r in rows if r["scripts"]["kn"]),
        "devanagari_spill_outputs": sum(1 for r in rows if r["scripts"]["hi_deva"]),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_HANDLE["path"] = run_dir / "run.log"
    status_path = run_dir / "run_status.json"
    record_status(status_path, "started", args=vars(args))
    if args.smoke:
        args.skip_install = True
        args.batch_size = min(args.batch_size, 8)
        args.eval_questions = min(args.eval_questions, 6)
        args.max_src = min(args.max_src, 64)
        args.max_tgt = min(args.max_tgt, 64)
        if "SMOKE_PAIRS" not in os.environ:
            os.environ["SMOKE_PAIRS"] = "256"
        if "SMOKE_STEPS" not in os.environ:
            os.environ["SMOKE_STEPS"] = "12"

    import torch
    from transformers import AutoModelForSeq2SeqLM, AlbertTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.smoke
    device = "cuda" if use_cuda else "cpu"
    use_amp = use_cuda and args.amp != "fp32"
    amp_dtype = None
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        capability = torch.cuda.get_device_capability(0)[0]
        amp_dtype = torch.bfloat16 if capability >= 8 else torch.float16
        log(f"GPU: {torch.cuda.get_device_name(0)}  "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB  "
            f"amp={args.amp}")
    else:
        log("No GPU -> CPU smoke mode (correctness check only, not a real run).")

    if not args.skip_install:
        install_deps()
    else:
        log("[1/7] Skipping dependency install (--skip-install).")

    # ── model ────────────────────────────────────────────────────────────────
    log(f"[2/7] Model {MODEL_ID}")
    import transformers as _transformers
    import peft as _peft
    log(f"      transformers {_transformers.__version__}  peft {_peft.__version__}  "
        f"torch {torch.__version__}")
    if MODEL_DIR.is_dir() and (list(MODEL_DIR.glob("*.bin"))
                               or list(MODEL_DIR.glob("*.safetensors"))):
        model_path = str(MODEL_DIR)
        log(f"      using local copy {model_path}")
    else:
        from huggingface_hub import snapshot_download
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        model_path = snapshot_download(
            repo_id=MODEL_ID, local_dir=str(MODEL_DIR),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"])

    tok = AlbertTokenizer.from_pretrained(model_path, do_lower_case=False,
                                         use_fast=False, keep_accents=True)
    bos_id = tok._convert_token_to_id_with_added_voc("<s>")
    eos_id = tok._convert_token_to_id_with_added_voc("</s>")
    pad_id = tok._convert_token_to_id_with_added_voc("<pad>")
    lang_id = tok._convert_token_to_id_with_added_voc(LANG_TOKEN)
    # This checkpoint's tokenizer reports [SEP]/[CLS] as eos/bos (ids 3/2), but
    # the model was trained with the added-vocab tokens </s>=64001 and <s>=64000.
    if lang_id == tok.unk_token_id:
        raise SystemExit(f"Tokenizer is missing {LANG_TOKEN}")
    log(f"      bos={bos_id} eos={eos_id} pad={pad_id} lang={lang_id}")

    def build_model():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, low_cpu_mem_usage=True)
        # The checkpoint leaves decoder_start_token_id unset; generation and
        # HF's shift_tokens_right both need <2te> so the decoder starts in Telugu.
        model.config.decoder_start_token_id = lang_id
        model.generation_config.decoder_start_token_id = lang_id
        model.config.bos_token_id = bos_id
        model.config.eos_token_id = eos_id
        model.config.pad_token_id = pad_id
        model.generation_config.bos_token_id = bos_id
        model.generation_config.eos_token_id = eos_id
        model.generation_config.pad_token_id = pad_id
        model.config.use_cache = False
        return model

    model = build_model()
    restore_shared_embeddings(model, model_path)
    record_status(status_path, "model_loaded", model_path=str(model_path))
    model.to(device)

    if use_amp and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # ── data ─────────────────────────────────────────────────────────────────
    log(f"[3/7] Data {args.data}")
    records = load_pairs(Path(args.data))
    if args.smoke:
        records = records[: int(os.environ["SMOKE_PAIRS"])]
    train_records, eval_records = split_questions(records, args.eval_questions, args.seed)
    log(f"      {len(records):,} questions -> {len(train_records):,} train / "
        f"{len(eval_records)} held out")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(args) + ("-smoke" if args.smoke else "")
    cache_file = CACHE_DIR / f"tokens_{key}.pt"
    if cache_file.exists() and not args.no_cache:
        blob = torch.load(cache_file, weights_only=False)
        train_examples, eval_examples = blob["train"], blob["eval"]
        log(f"[4/7] Loaded tokenized cache {cache_file.name} "
            f"({len(train_examples):,} train / {len(eval_examples)} eval)")
    else:
        log("[4/7] Tokenizing …")
        train_examples = tokenize_split(tok, train_records, args, eos_id)
        eval_examples = tokenize_split(tok, eval_records, args, eos_id)
        torch.save({"train": train_examples, "eval": eval_examples}, cache_file)
        log(f"      cached {cache_file.name}")

    train_flat = flatten_examples(train_examples)
    eval_flat = flatten_examples(eval_examples)
    log(f"      {len(train_flat):,} train pairs, {len(eval_flat)} eval pairs")

    # ── baseline eval (before any LoRA weights exist) ────────────────────────
    baseline_summary, baseline_rows = {}, []
    if not args.no_baseline:
        log(f"[4/7] Held-out eval — BASE model (pre-SFT), "
            f"{len(eval_records[: args.eval_questions])} questions …")
        baseline_summary, baseline_rows = evaluate(
            model, tok, eval_records, args, lang_id, bos_id, eos_id, device)

    # ── LoRA ─────────────────────────────────────────────────────────────────
    targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
    if args.lora_mlp:
        targets += ["fc1", "fc2"]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=targets)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    record_status(status_path, "lora_ready")

    # ── AMP probe (Kaggle T4/torch 2.11 produced NaN under fp16 autocast) ────
    if use_amp:
        probe_src, probe_mask, probe_labels = collate(train_flat[:4], pad_id, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            probe_loss = model(input_ids=probe_src, attention_mask=probe_mask,
                               labels=probe_labels).loss
        probe_value = float(probe_loss)
        log(f"      AMP probe ({str(amp_dtype).replace('torch.', '')}): "
            f"loss={probe_value:.4f} finite={math.isfinite(probe_value)}")
        if not math.isfinite(probe_value):
            log("      non-finite under autocast -> training in fp32")
            use_amp = False
            scaler = None

    # ── train ────────────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_flat) / args.batch_size)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.smoke:
        total_steps = min(total_steps, int(os.environ["SMOKE_STEPS"]))
    warmup_steps = max(1, int(total_steps * args.warmup))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    log(f"[5/7] Training {total_steps} steps × batch {args.batch_size} "
        f"(1 epoch = {steps_per_epoch} steps) …")
    model.train()
    started = time.perf_counter()
    step = 0
    optimizer_steps = 0
    skipped_batches = 0
    losses = []
    epoch = 0
    while step < total_steps:
        batches = make_batches(train_flat, args.batch_size, args.seed + epoch, shuffle=True)
        for indices in batches:
            if step >= total_steps:
                break
            batch = [train_flat[i] for i in indices]
            src, src_mask, labels = collate(batch, pad_id, device)
            amp_context = (torch.autocast("cuda", dtype=amp_dtype)
                           if use_amp else contextlib.nullcontext())
            with amp_context:
                out = model(input_ids=src, attention_mask=src_mask, labels=labels)
                loss = out.loss
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                skipped_batches += 1
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    log(f"      non-finite loss at step {step + 1}; "
                        f"disabling autocast and continuing in fp32")
                    use_amp = False
                    scaler = None
                    continue
                if skipped_batches > 5:
                    raise SystemExit(
                        "Loss is non-finite even in fp32 — aborting instead of "
                        "saving a no-op adapter. Check the batch and labels.")
                log(f"      skipped non-finite batch ({skipped_batches})")
                continue
            if scaler is not None:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                stepped = True
            if stepped:
                scheduler.step()
                optimizer_steps += 1
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss_value)
            if step == 1:
                log(f"      first loss {losses[0]:.4f} "
                    f"(≈1.2 for this checkpoint; >6 means the token embeddings "
                    f"did not load — see the 'embeddings tied' line above)")
                if losses[0] > 6.0:
                    log("      WARNING: first loss is too high for a pretrained "
                        "paraphraser; training would learn nothing useful.")
            if step % 50 == 0 or step == total_steps:
                avg = sum(losses[-50:]) / len(losses[-50:])
                rate = step / max(1e-9, time.perf_counter() - started)
                peak = (torch.cuda.max_memory_allocated() / 1e9) if use_cuda else 0.0
                log(f"      step {step}/{total_steps} loss={avg:.4f} "
                    f"{rate:.2f} it/s" + (f" peak={peak:.1f}GB" if use_cuda else ""))
        epoch += 1

    if optimizer_steps == 0:
        raise SystemExit(
            "No optimizer steps were taken (all losses non-finite or all steps "
            "skipped) — refusing to save a no-op adapter.")
    elapsed = time.perf_counter() - started
    log(f"      done: {step} steps ({optimizer_steps} optimizer updates) in "
        f"{elapsed / 60:.1f} min, mean loss {sum(losses) / len(losses):.4f}, "
        f"fp16_amp={'on' if use_amp else 'off'}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    log(f"[6/7] Adapter saved: {out_dir}")
    record_status(status_path, "adapter_saved", steps=step,
                  optimizer_steps=optimizer_steps, seconds=round(elapsed, 1))

    # ── held-out eval (post-SFT) ─────────────────────────────────────────────
    log(f"[7/7] Held-out eval — SFT adapter, "
        f"{len(eval_records[: args.eval_questions])} questions "
        f"({len(eval_flat)} references) …")
    summary, rows = evaluate(model, tok, eval_records, args, lang_id,
                             bos_id, eos_id, device)
    record_status(status_path, "eval_done", token_f1=summary["token_f1"],
                  distinct_best_ratio=summary["distinct_best_ratio"])
    summary.update({
        "train_steps": step,
        "optimizer_steps": optimizer_steps,
        "amp_mode": "fp16" if use_amp else "fp32",
        "train_seconds": round(elapsed, 1),
        "final_loss": round(sum(losses[-200:]) / len(losses[-200:]), 4),
        "base_model": MODEL_ID,
        "lora_targets": targets,
        "trainable_params": sum(p.numel() for p in model.parameters()
                                if p.requires_grad),
        "smoke": args.smoke,
    })
    if baseline_summary:
        summary["baseline"] = baseline_summary
        for metric in ("token_f1", "distinct_best_ratio", "exact_copy_rate",
                       "empty_rate", "ends_with_question_mark"):
            summary[f"delta_{metric}"] = round(
                summary[metric] - baseline_summary[metric], 4)

    (out_dir.parent / "sft_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir.parent / "sft_samples.json").write_text(
        json.dumps({"baseline": baseline_rows, "sft": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    log("\n── Held-out summary ─────────────────────────────────────────────")
    for key_name, value in summary.items():
        if key_name == "baseline":
            continue
        log(f"  {key_name:28s} {value}")
    if baseline_summary:
        log("\n── Base vs SFT (same held-out questions) ────────────────────────")
        log(f"  {'metric':28s} {'base':>8s} {'sft':>8s} {'delta':>8s}")
        for metric in ("token_f1", "distinct_best_ratio", "exact_copy_rate",
                       "empty_rate", "ends_with_question_mark"):
            log(f"  {metric:28s} {baseline_summary[metric]:>8} {summary[metric]:>8} "
                f"{summary.get('delta_' + metric, 0):>8}")

    degenerate = (summary["empty_rate"] > 0.10
                  or summary["exact_copy_rate"] > 0.70
                  or summary["distinct_best_ratio"] < 0.50)
    if degenerate:
        log("\n  WARNING: degenerate generations (empty/copy/collapse). "
            "Check the loss curve, embedding repair line, and label shift.")

    log("\n── Raw samples (no filtering, no decode-time constraints) ───────")
    for index, row in enumerate(rows[:5]):
        base_row = baseline_rows[index] if index < len(baseline_rows) else None
        log(f"\nQ: {row['question']}")
        for ref in row["references"][:2]:
            log(f"   ref: {ref}")
        if base_row:
            cust = base_row["scripts"]
            log(f"   base 1. {base_row['candidates'][0]}   [te={cust['te']} "
                f"ta={cust['ta']} kn={cust['kn']} deva={cust['hi_deva']}]")
        for candidate_index, candidate in enumerate(row["candidates"], 1):
            stats = row["scripts"]
            log(f"   sft  {candidate_index}. {candidate}   [te={stats['te']} "
                f"ta={stats['ta']} kn={stats['kn']} deva={stats['hi_deva']}]")

    log(f"\nDone. Metrics: {out_dir.parent / 'sft_metrics.json'}")


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
