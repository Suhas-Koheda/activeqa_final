#!/usr/bin/env python3
"""Interactive demo: generate three reformulations with the English/Telugu models."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_TELUGU_QUESTION = "భారతదేశానికి స్వాతంత్ర్యం ఏ సంవత్సరంలో వచ్చింది?"


def device_for(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this Python environment cannot see a CUDA GPU.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_english(model_path: str, tokenizer_path: str | None, device: torch.device,
                 allow_download: bool):
    local_only = not allow_download
    tok_path = tokenizer_path or model_path
    try:
        tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=local_only)
    except OSError:
        if tokenizer_path:
            raise
        base_tokenizer = "Vamsi/T5_Paraphrase_Paws"
        print(f"Tokenizer not included with this model; loading {base_tokenizer}.", flush=True)
        tok = AutoTokenizer.from_pretrained(base_tokenizer, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_only)
    model.to(device).eval()
    return tok, model


def load_telugu(base_path: str, checkpoint_path: str, device: torch.device,
                rank: int, alpha: int, allow_download: bool):
    local_only = not allow_download
    tok = AutoTokenizer.from_pretrained(
        base_path, use_fast=False, do_lower_case=False, keep_accents=True,
        local_files_only=local_only,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(base_path, local_files_only=local_only)
    if not checkpoint_path:
        target_id = tok._convert_token_to_id_with_added_voc("<2te>")
        bos_id = tok._convert_token_to_id_with_added_voc("<s>")
        eos_id = tok._convert_token_to_id_with_added_voc("</s>")
        model.config.decoder_start_token_id = target_id
        model.generation_config.decoder_start_token_id = target_id
        model.config.bos_token_id = bos_id
        model.generation_config.bos_token_id = bos_id
        model.config.eos_token_id = eos_id
        model.generation_config.eos_token_id = eos_id
        model.to(device).eval()
        return tok, model, {"target": target_id, "bos": bos_id, "eos": eos_id}, "pretrained base model (no RL adapter)"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        kind = "full model checkpoint"
    elif "adapter" in checkpoint:
        adapter = checkpoint["adapter"]
        inferred_rank = next(
            (int(value.shape[0]) for key, value in adapter.items()
             if "lora_A" in key and hasattr(value, "shape")), rank
        )
        targets = sorted({module for key in adapter for module in ("q_proj", "v_proj")
                          if f".{module}.lora_A." in key})
        if not targets:
            raise ValueError("Could not identify LoRA layers in the Telugu checkpoint.")
        model = get_peft_model(model, LoraConfig(
            r=inferred_rank, lora_alpha=alpha, target_modules=targets,
            lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM",
        ))
        set_peft_model_state_dict(model, adapter)
        kind = f"RL adapter (rank {inferred_rank}, alpha {alpha})"
    else:
        raise ValueError(f"Unsupported Telugu checkpoint format: {checkpoint_path}")

    target_id = tok._convert_token_to_id_with_added_voc("<2te>")
    bos_id = tok._convert_token_to_id_with_added_voc("<s>")
    eos_id = tok._convert_token_to_id_with_added_voc("</s>")
    if target_id == tok.unk_token_id:
        raise ValueError("The Telugu tokenizer does not contain the <2te> language token.")
    model.config.decoder_start_token_id = target_id
    model.generation_config.decoder_start_token_id = target_id
    model.config.bos_token_id = bos_id
    model.generation_config.bos_token_id = bos_id
    model.config.eos_token_id = eos_id
    model.generation_config.eos_token_id = eos_id
    model.to(device).eval()
    return tok, model, {"target": target_id, "bos": bos_id, "eos": eos_id}, kind


def generate_english(tok, model, question: str, count: int, max_tokens: int,
                     temperature: float, top_p: float) -> list[str]:
    encoded = tok("paraphrase: " + question, return_tensors="pt", truncation=True,
                  max_length=128).to(model.device)
    with torch.inference_mode():
        sequences = model.generate(
            **encoded, max_new_tokens=max_tokens, do_sample=True,
            num_return_sequences=count, top_p=top_p, temperature=temperature,
            no_repeat_ngram_size=3, pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return [tok.decode(sequence, skip_special_tokens=True).strip() for sequence in sequences]


def generate_telugu(tok, model, special_ids: dict, question: str, count: int,
                    max_tokens: int, temperature: float, top_p: float) -> list[str]:
    encoded = tok(question + " </s> <2te>", add_special_tokens=False,
                  return_tensors="pt", truncation=True, max_length=96)
    encoded = {key: value.to(model.device) for key, value in encoded.items()
               if key in {"input_ids", "attention_mask"}}
    with torch.inference_mode():
        sequences = model.generate(
            **encoded, max_length=max_tokens, do_sample=False,
            num_beams=max(4, count), num_return_sequences=count, early_stopping=True,
            no_repeat_ngram_size=3, bos_token_id=special_ids["bos"],
            encoder_no_repeat_ngram_size=3,
            eos_token_id=special_ids["eos"], pad_token_id=tok.pad_token_id,
        )
    return [tok.decode(sequence, skip_special_tokens=True).strip() for sequence in sequences]


def _telugu_script_only(text: str) -> bool:
    # Reject multilingual-tokenizer spillover (for example Bengali or Devanagari).
    return not re.search(r"[\u0900-\u0bff\u0d00-\u0d7f]", text)


def choose_telugu_demo_candidates(question: str, generated: list[str], count: int) -> list[str]:
    """Filter garbled, incomplete, and duplicate outputs from the actual model."""
    result: list[str] = []
    seen = {" ".join(question.split()).rstrip(" ?").casefold()}
    for candidate in generated:
        candidate = " ".join(candidate.split()).strip(" \"'“”‘’")
        normalized = candidate.rstrip(" ?").casefold()
        if not candidate or normalized in seen or not _telugu_script_only(candidate):
            continue
        # Discard clipped fragments and outputs that stopped being questions.
        if not candidate.endswith(("?", "؟")):
            continue
        seen.add(normalized)
        result.append(candidate)
        if len(result) >= count:
            break
    return result


def ask_if_missing(value: str | None, prompt: str) -> str:
    if value and value.strip():
        return value.strip()
    answer = input(prompt).strip()
    if not answer:
        raise SystemExit("Please enter a question.")
    return answer


def print_candidates(language: str, question: str, candidates: list[str],
                     expected_answer: str | None = None) -> None:
    print(f"\n{language} original: {question}")
    if expected_answer:
        print(f"Expected answer (reference): {expected_answer}")
    for index, candidate in enumerate(candidates, 1):
        print(f"{index}. {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "te", "both"], default="both")
    parser.add_argument("--question", help="Question for --language en or --language te")
    parser.add_argument("--english-question", help="English question when using --language both")
    parser.add_argument("--telugu-question", default=DEFAULT_TELUGU_QUESTION,
                        help="Telugu question when using --language both")
    parser.add_argument("--answer", help="Optional known answer, printed for a single-language demo")
    parser.add_argument("--english-answer", help="Optional known English answer for --language both")
    parser.add_argument("--telugu-answer", help="Optional known Telugu answer for --language both")
    parser.add_argument("--english-model", default="models/hf_fresh/english",
                        help="Downloaded English full-model directory or Hub model ID")
    parser.add_argument("--english-tokenizer", default=None,
                        help="Optional tokenizer directory/ID; defaults to the model directory")
    parser.add_argument("--telugu-base", default="models/hf_fresh/telugu",
                        help="Downloaded Telugu base-model directory or Hub model ID")
    parser.add_argument("--telugu-checkpoint", default=None,
                        help="Optional Telugu RL checkpoint; by default use the pretrained base model")
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow the tokenizer/model loader to fetch files missing locally")
    args = parser.parse_args()
    if args.candidates < 1:
        parser.error("--candidates must be positive")
    if args.language != "both" and args.question is None:
        # Missing --question intentionally starts a one-question interactive prompt.
        pass
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    print(f"Using {device}; generating {args.candidates} candidate(s) per question.", flush=True)

    if args.language in {"en", "both"}:
        english_question = args.question if args.language == "en" else args.english_question
        english_question = ask_if_missing(english_question, "English question: ")
        print("Loading English model...", flush=True)
        en_tok, en_model = load_english(
            args.english_model, args.english_tokenizer, device, args.allow_download)
        print_candidates("ENGLISH", english_question,
                         generate_english(en_tok, en_model, english_question, args.candidates,
                                          args.max_new_tokens, args.temperature, args.top_p),
                         args.english_answer if args.language == "both" else args.answer)
        del en_model, en_tok
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.language in {"te", "both"}:
        telugu_question = args.question if args.language == "te" else args.telugu_question
        telugu_question = ask_if_missing(telugu_question, "తెలుగు ప్రశ్న: ")
        print("Loading Telugu model...", flush=True)
        te_tok, te_model, special_ids, kind = load_telugu(
            args.telugu_base, args.telugu_checkpoint, device,
            args.lora_rank, args.lora_alpha, args.allow_download)
        print(f"Loaded Telugu {kind}.", flush=True)
        # Seed after loading so the requested seed controls generation directly.
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model_candidates = generate_telugu(te_tok, te_model, special_ids, telugu_question,
                                           args.candidates, args.max_new_tokens,
                                           args.temperature, args.top_p)
        candidates = choose_telugu_demo_candidates(telugu_question, model_candidates,
                                                   args.candidates)
        print_candidates("TELUGU", telugu_question, candidates,
                         args.telugu_answer if args.language == "both" else args.answer)


if __name__ == "__main__":
    main()
