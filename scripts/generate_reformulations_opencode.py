"""Generate a small, review-required Telugu reformulation set with OpenCode free."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import time
import unicodedata
from pathlib import Path


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/indic_rag_te/train.jsonl")
    ap.add_argument("--output", default="data/indic_rag_te/reformulations_review.jsonl")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--model", default="opencode/minimax-m3.1-flash-free")
    args = ap.parse_args()

    all_records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = all_records[:args.count]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    accepted = []
    completed = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            accepted.append(item)
            completed.add(item.get("id"))
    remaining = [r for r in records if r["id"] not in completed]
    for offset in range(0, len(remaining), args.batch_size):
        batch = remaining[offset:offset + args.batch_size]
        prompt_rows = [{"id": r["id"], "question": r["question"], "context": r["context"][:450]}
                       for r in batch]
        prompt = (
            "For every input, write exactly one natural Telugu paraphrase of its question. "
            "Preserve its complete meaning, answer type, entities, dates, numbers, and polarity. "
            "Do not answer the question, add facts, or change what it asks. Use the context only "
            "to resolve meaning. Return only a JSON array of objects with exactly the keys id and "
            "paraphrase, preserving every id exactly.\nINPUT:\n" +
            json.dumps(prompt_rows, ensure_ascii=False)
        )
        result = None
        for attempt in range(3):
            result = subprocess.run(["opencode", "run", "--model", args.model, prompt],
                                    capture_output=True, text=True, timeout=120, check=False)
            if result.returncode == 0:
                break
            print(f"OpenCode batch failed (attempt {attempt + 1}/3): {result.stderr[-800:]}",
                  file=sys.stderr, flush=True)
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
        if result is None or result.returncode:
            raise RuntimeError(f"OpenCode failed on batch at row {offset} after retries")
        generated = extract_json(result.stdout)
        by_id = {str(x.get("id")): str(x.get("paraphrase", "")).strip()
                 for x in generated if isinstance(x, dict)}
        for row in batch:
            candidate = by_id.get(row["id"], "")
            if not candidate:
                continue
            original_nums = re.findall(r"\d+(?:[./:-]\d+)*", row["question"])
            candidate_nums = re.findall(r"\d+(?:[./:-]\d+)*", candidate)
            accepted.append({
                "id": row["id"],
                "question": row["question"],
                "candidate": unicodedata.normalize("NFC", candidate),
                "context": row["context"],
                "answers": row["answers"],
                "source": row["source"],
                "source_url": row["url"],
                "generator": args.model,
                "generated_unix": int(time.time()),
                "needs_manual_review": True,
                "automatic_checks": {
                    "nonempty": True,
                    "preserved_ascii_numbers": not original_nums or original_nums == candidate_nums,
                    "semantic_equivalence_verified": False,
                },
            })
        out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in accepted),
                       encoding="utf-8")
        print(f"generated {len(accepted)}/{len(records)} reformulations", flush=True)
    print(f"wrote {len(accepted)} review-required pairs to {out}")


if __name__ == "__main__":
    main()
