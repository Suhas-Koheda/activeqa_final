"""Expand the deduplicated user-provided triples into leakage-safe SFT pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path


def key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/indic_rag_te/user_provided_reformulations.jsonl")
    parser.add_argument("--out-dir", default="data/indic_rag_te/user_reformulations")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        raise ValueError("dev-fraction must be between 0 and 1")

    examples: dict[str, dict] = {}
    with open(args.input, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("original_question", "")).strip()
            answer = str(row.get("gold_answer", "")).strip()
            rewrites = row.get("reformulations", [])
            if not question or not answer or not isinstance(rewrites, list):
                raise ValueError(f"Incomplete example on line {line_number}")
            qkey = key(question)
            if qkey in examples:
                # Keep one source-question row; all records are grouped before splitting.
                continue
            unique_rewrites = []
            seen_rewrites = {qkey}
            for rewrite in rewrites:
                rewrite = str(rewrite).strip()
                rkey = key(rewrite)
                if rewrite and rkey not in seen_rewrites:
                    unique_rewrites.append(rewrite)
                    seen_rewrites.add(rkey)
            examples[qkey] = {"question": question, "answer": answer,
                              "rewrites": unique_rewrites}

    keys = sorted(examples)
    random.Random(args.seed).shuffle(keys)
    n_dev = max(1, round(len(keys) * args.dev_fraction))
    dev_keys = set(keys[:n_dev])
    splits = {"train": [], "dev": []}
    for qkey, example in examples.items():
        split = "dev" if qkey in dev_keys else "train"
        source_id = hashlib.sha256(qkey.encode("utf-8")).hexdigest()[:16]
        for i, rewrite in enumerate(example["rewrites"], 1):
            splits[split].append({
                "id": f"user-rewrite-{source_id}-{i}",
                "source_id": source_id,
                "question": example["question"],
                "reformulation": rewrite,
                "context": "",
                "answers": [example["answer"]],
                "wiki_id": f"user-{source_id}",
                "source": "user-provided data.txt",
                "label_status": "silver_user_provided_not_human_verified",
                "semantic_equivalence_verified": False,
            })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}_questions={len({r['source_id'] for r in rows})} "
              f"pairs={len(rows)} path={path}")
    print(f"unique_source_questions={len(examples)}")


if __name__ == "__main__":
    main()
