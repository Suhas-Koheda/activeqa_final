"""Zero-shot pilot for the candidate Sarvam-1 Telugu reformulator.

Loads the local snapshot (bf16, CPU), prompts the completion model with an
explicit Telugu paraphrase instruction, and prints the raw sampled outputs —
no filtering, no rewriting, exactly as the model generates them. This is the
gate check before any training work switches the base model.

Usage:
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
        python scripts/demo_sarvam_pilot.py --question "భారతదేశ రాజధాని ఏది?"
"""
from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("HF_HUB_OFFLINE", "0")

PROMPT = (
    "కింది తెలుగు ప్రశ్నను తెలుగులోనే భిన్నంగా రాయండి. అర్థం మారకూడదు; మాత్రమే పదాల క్రమం/పదజాలం మారాలి.\n"
    "ప్రశ్న: {question}\n"
    "సమానార్థ ప్రశ్న:"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="models/sarvam-1-pilot")
    ap.add_argument("--question", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    print(f"loading bf16 weights from {args.model_dir} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="cpu")
    model.eval()

    prompt = PROMPT.format(question=args.question)
    ids = tok(prompt, return_tensors="pt")
    print(f"prompt_tokens={ids['input_ids'].shape[1]}", flush=True)
    retries = len(args.question) // 3
    with torch.inference_mode():
        for index in range(args.samples):
            out = model.generate(
                **ids,
                do_sample=True, temperature=args.temperature, top_p=0.9,
                max_new_tokens=args.max_new,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            print(f"{index + 1}. {text.strip()}", flush=True)
    gc.collect()
    print(f"memory_peak_mb={__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 1024:.0f}")


if __name__ == "__main__":
    main()
