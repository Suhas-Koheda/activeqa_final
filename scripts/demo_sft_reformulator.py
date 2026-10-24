"""Raw-output demo for the LoRA-SFT Telugu reformulator.

Loads base + adapter, samples without any post-hoc filtering, prints exactly
what the model generates plus script-column statistics (Telugu vs Tamil /
Kannada / Latin / Arabic-macrolanguage ranges) so purity is observable in the
raw stream rather than enforced at decode time.

Usage:
    OMP_NUM_THREADS=8 python scripts/demo_sft_reformulator.py \
        --adapter models/sarvam-reformulator-lora/adapter \
        --question "భారతదేశ రాజధాని ఏది?"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

TELUGU = re.compile(r"[\u0C00-\u0C7F]")
TAMIL = re.compile(r"[\u0B80-\u0BFF]")
KANNADA = re.compile(r"[\u0C80-\u0CFF]")
LATIN = re.compile(r"[A-Za-z]")


def script_stats(text: str) -> str:
    counts = {
        "te": len(TELUGU.findall(text)),
        "ta": len(TAMIL.findall(text)),
        "kn": len(KANNADA.findall(text)),
        "la": len(LATIN.findall(text)),
    }
    return ",".join(f"{key}={value}" for key, value in counts.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None,
                    help="base model dir/HF id; defaults to the hub Sarvam-1")
    ap.add_argument("--question", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    base = args.base or Path(args.adapter).parent  # config.json is copied? default hub
    tok = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        args.base or "sarvamai/sarvam-1", dtype=dtype, low_cpu_mem_usage=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompt = f"ప్రశ్న: {args.question}\nపునర్రచన:"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    print(f"script_stats_per_sample (te=pure Telugu target, ta/kn/la=spill):", flush=True)
    with torch.inference_mode():
        for index in range(args.samples):
            out = model.generate(
                **ids, do_sample=True, temperature=args.temperature,
                top_p=0.9, max_new_tokens=args.max_new,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            print(f"{index + 1}. {text}    [{script_stats(text)}]", flush=True)
    peak_mb = __import__("resource").getrusage(
        __import__("resource").RUSAGE_SELF).ru_maxrss / 1024
    print(f"memory_peak_mb={peak_mb:.0f}", flush=True)


if __name__ == "__main__":
    main()
