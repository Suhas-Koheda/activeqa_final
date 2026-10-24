"""Zero-shot pilot for instruct-tuned Indic models (Navarasa Gemma-2b / Sarvam).

Loads a local snapshot, builds a Telugu question-reformulation prompt in the
model's chat format, and prints raw sampled outputs — no filtering, no
rewriting. Run it against several models and keep the one that reliably writes
clean Telugu before investing in training.

Usage:
    OMP_NUM_THREADS=8 python scripts/demo_instruct_reformulator.py \
        --model-dir models/navarasa-gemma2b \
        --question "చార్మినార్‌ను ఏ సంవత్సరంలో నిర్మించారు?"
"""
from __future__ import annotations

import argparse
import os

INSTRUCTION_TAIL = (
    "ఈ ప్రశ్నను తెలుగులోనే భిన్నంగా రాయండి. అర్థం మారకూడదు; పదాలు/నిర్మాణం మాత్రమే మారాలి. "
    "మరో ఏ భాష పదాలు వాడకూడదు. వచనము ప్రశ్నగా అంతమవ్వాలి."
)

INSTRUCTION = "{question}\n" + INSTRUCTION_TAIL


def build_prompt(tok, question: str, style: str,
                 fewshot: list[dict] | None = None) -> dict:
    instruction = INSTRUCTION.format(question=question)
    # Navarasa ships without tokenizer.chat_template; Gemma's native turn
    # format is fully documented, so build it by hand.
    if style == "chat":
        if fewshot:
            turns = [
                "<start_of_turn>user\n" + fewshot[0]["question"]
                + "\n" + INSTRUCTION_TAIL + "<end_of_turn>\n"
                "<start_of_turn>model\n" + fewshot[0]["paraphrase"]
                + "<end_of_turn>\n",
                "<start_of_turn>user\n" + question
                + "\n" + INSTRUCTION_TAIL + "<end_of_turn>\n"
                "<start_of_turn>model\n",
            ]
            return tok("".join(turns), return_tensors="pt")
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
            )
        prompt = (
            "<start_of_turn>user\n" + instruction + "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        return tok(prompt, return_tensors="pt")
    if style == "alpaca":
        prompt = (
            "కింది విషయం మీకు జరిగిన సమాధానం తెలుగులో ఇవ్వండి.\n\n"
            "### Instruction:\n" + instruction + "\n\n### Response:\n"
        )
        return tok(prompt, return_tensors="pt")
    # "raw": plain instruction (sarvam-1 base model has no chat format)
    if fewshot:
        prompt = (
            "ప్రశ్న: " + fewshot[0]["question"] + "\n"
            "పునర్రచన: " + fewshot[0]["paraphrase"] + "\n"
            "ప్రశ్న: " + question + "\n"
            "పునర్రచన:"
        )
        return tok(prompt + "\n", return_tensors="pt")
    return tok(instruction + "\n", return_tensors="pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prompt-style", default="chat",
                    choices=("chat", "alpaca", "raw"))
    ap.add_argument("--fewshot-file", default=None,
                    help="JSONL with question/paraphrase pairs used as an in-context example")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    print(f"loading {args.model_dir} dtype={args.dtype} threads={args.threads} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map="cpu")
    model.eval()

    fewshot = None
    if args.fewshot_file:
        import json
        with open(args.fewshot_file) as stream:
            record = json.loads(stream.readline())
        fewshot = [{"question": record["question"],
                    "paraphrase": record["paraphrase"]}]

    inputs = build_prompt(tok, args.question, args.prompt_style, fewshot)
    ids = inputs["input_ids"]
    print(f"prompt_tokens={ids.shape[1]}", flush=True)
    with torch.inference_mode():
        for index in range(args.samples):
            out = model.generate(
                **inputs, do_sample=True, temperature=args.temperature,
                top_p=0.9, max_new_tokens=args.max_new,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            print(f"{index + 1}. {text.strip()}", flush=True)
    peak_mb = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024
    print(f"memory_peak_mb={peak_mb:.0f}", flush=True)


if __name__ == "__main__":
    main()
