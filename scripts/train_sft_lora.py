"""LoRA SFT of a Telugu-capable base model on question→reformulation pairs.

Trains the reformulator itself (not the old MultiIndic model) on the
IndicParaphrase question pairs, so zero-shot script spill and meaning drift are
fixed *inside* the model — no decode-time constraints, per project policy.

Default prompt style is the completion format that the Sarvam-1 base respects
(``ప్రశ్న: ... పునర్రచన:``); gemma_chat emits <start_of_turn> pairs for
Navarasa-style instruct checkpoints. Loss is computed on response tokens only.

Runs on a Colab GPU (default) or a CPU box with --smoke for a tiny sanity pass.
Data format: JSONL with {"original_question": str, "reformulations": [str]}.

Usage (Colab):
    python scripts/train_sft_lora.py --config configs/colab_sft_lora.yaml
Local smoke:
    OMP_NUM_THREADS=8 python scripts/train_sft_lora.py \
        --config configs/colab_sft_lora.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import yaml


def load_pairs(path: Path, max_per_question: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = (record.get("original_question")
                        or record.get("question") or "").strip()
            paraphrases = record.get("reformulations")
            if paraphrases is None:
                paraphrases = [p for p in [record.get("paraphrase")] if p]
            if not question or not paraphrases:
                continue
            for p in paraphrases[:max_per_question]:
                p = p.strip()
                if not p or p == question:
                    continue
                pairs.append((question, p))
    return pairs


def build_completion(question: str, paraphrase: str) -> tuple[str, str]:
    prompt = f"ప్రశ్న: {question}\nపునర్రచన:"
    target = f" {paraphrase.strip()}\n"
    return prompt, target


def build_gemma_chat(question: str, paraphrase: str, instruction: str) -> tuple[str, str]:
    prompt = (
        "<start_of_turn>user\n" + question + "\n" + instruction
        + "<end_of_turn>\n<start_of_turn>model\n"
    )
    target = paraphrase.strip() + "<end_of_turn>\n"
    return prompt, target


TRAIN_INSTRUCTION = (
    "ఈ ప్రశ్నను తెలుగులోనే భిన్నంగా రాయండి. అర్థం మారకూడదు; పదాలు/నిర్మాణం "
    "మాత్రమే మారాలి. వచనము ప్రశ్నగా అంతమవ్వాలి."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset, one step — sanity check only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dtype", default=None, choices=("bf16", "fp32"),
                    help="override config dtype (e.g. keep bf16 on CPU)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    data_cfg = cfg["data"]
    training = cfg["training"]
    out_dir = Path(training["output_dir"])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_cfg = str(cfg.get("dtype", "bf16"))
    if args.dtype:
        dtype_cfg = args.dtype
    elif dtype_cfg == "bf16" and device != "cuda" and bool(cfg.get("cpu_dtype_downgrade", True)):
        dtype_cfg = "fp32"
    dtype = torch.bfloat16 if dtype_cfg == "bf16" else torch.float32
    grad_fp32 = device == "cuda" and torch.cuda.get_device_capability(0)[0] < 8

    model_name = cfg["models"].get("reformulator_path") or cfg["models"]["reformulator"]
    print(f"[sft-lora] base model: {model_name} device={device} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True)
    if getattr(model.config, "use_cache", False) and bool(training.get("gradient_checkpointing", False)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads = getattr(model, "enable_input_require_grads", None)

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=int(cfg["models"].get("lora_rank", 16)),
        lora_alpha=int(cfg["models"].get("lora_alpha", 32)),
        lora_dropout=float(cfg["models"].get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["models"].get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    pairs = load_pairs(Path(data_cfg["train_pairs_file"]),
                       int(data_cfg.get("max_pairs_per_question", 2)))
    random.shuffle(pairs)
    if args.smoke:
        pairs = pairs[: int(os.environ.get("SMOKE_PAIRS", "32"))]
    print(f"[sft-lora] training pairs: {len(pairs)}", flush=True)

    style = str(cfg.get("training", {}).get("prompt_style", "completion"))
    max_len = int(cfg.get("training", {}).get("max_input_tokens", 192))
    micro = int(training.get("micro_batch_pairs", 4))
    accum = int(training.get("grad_accum", 4))
    epochs = 1 if args.smoke else int(training.get("epochs", 1))
    lr = float(training.get("learning_rate", 1e-4))
    warmup_ratio = float(training.get("warmup_ratio", 0.03))

    # Pre-tokenize: prompt tokens get -100 labels, target tokens get their ids.
    examples = []
    skipped_long = 0
    for question, paraphrase in pairs:
        if style == "gemma_chat":
            prompt_text, target_text = build_gemma_chat(
                question, paraphrase, TRAIN_INSTRUCTION)
        else:
            prompt_text, target_text = build_completion(question, paraphrase)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(target_ids) > max_len:
            skipped_long += 1
            continue
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        examples.append({"input_ids": input_ids, "labels": labels})
    if skipped_long:
        print(f"[sft-lora] skipped {skipped_long} pairs exceeding {max_len} tokens",
              flush=True)
    print(f"[sft-lora] tokenized examples: {len(examples)}", flush=True)

    steps_per_epoch = max(1, math.ceil(len(examples) / (micro * accum)))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 0.5 * (1 + math.cos(math.pi * (step - warmup_steps)
                                            / max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    pad_id = tok.pad_token_id
    ambient_stack = os.environ.get("SFT_STACK", "")  # for local smoke reporting
    started = time.perf_counter()
    step = 0
    metrics_path = out_dir / "sft_lora_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        random.shuffle(examples)
        model.train()
        optimizer_steps = math.ceil(len(examples) / (micro * accum))
        for batch_start in range(0, len(examples), micro * accum):
            chunk = examples[batch_start: batch_start + micro * accum]
            max_len_batch = max(len(e["input_ids"]) for e in chunk)
            input_ids = torch.full((len(chunk), max_len_batch), pad_id,
                                   dtype=torch.long)
            attention = torch.zeros((len(chunk), max_len_batch), dtype=torch.long)
            labels = torch.full((len(chunk), max_len_batch), -100, dtype=torch.long)
            for row, example in enumerate(chunk):
                length = len(example["input_ids"])
                input_ids[row, :length] = torch.tensor(example["input_ids"])
                attention[row, :length] = 1
                labels[row, :length] = torch.tensor(example["labels"])
            input_ids, attention, labels = (
                input_ids.to(device), attention.to(device), labels.to(device))
            out = model(input_ids=input_ids, attention_mask=attention,
                        labels=labels)
            (out.loss / accum).backward()
            if (batch_start // micro) % accum == accum - 1 or batch_start + micro >= len(examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or step == 1:
                    rate = step / max(1e-9, time.perf_counter() - started)
                    peak = (torch.cuda.max_memory_allocated() / 1e9
                            if device == "cuda" else 0.0)
                    print(f"[sft-lora] step {step}/{total_steps} "
                          f"loss={out.loss.item():.4f} {rate:.2f} it/s "
                          f"peak_gpu={peak:.2f}GiB", flush=True)

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"[sft-lora] adapter saved to {adapter_dir}", flush=True)

    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump({
            "steps": step,
            "examples": len(examples),
            "seconds": round(time.perf_counter() - started, 1),
            "final_loss": float(out.loss.item()),
            "base_model": str(model_name),
            "prompt_style": style,
            "smoke": bool(args.smoke),
        }, stream, indent=2)
    print(f"[sft-lora] metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
