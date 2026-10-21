"""Build a small Telugu rewrite seed set using conservative phrase substitutions.

No language model or external service is called. All generated records are marked
for Telugu-speaker review; these are silver labels, not verified paraphrases.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


# Ordered longest/specific forms first. Each replacement preserves the question
# type and changes only a common interrogative phrase.
RULES = [
    ("where_from", re.compile(r"ఎక్కడి\s*నుంచి"), "ఏ ప్రదేశం నుంచి"),
    ("where_from", re.compile(r"ఎక్కడ\s*నుంచి"), "ఏ ప్రదేశం నుంచి"),
    ("where_to", re.compile(r"ఎక్కడికి"), "ఏ ప్రదేశానికి"),
    ("year", re.compile(r"ఏ\s+సంవత్సరంలో"), "ఏ ఏడాదిలో"),
    ("where", re.compile(r"ఎక్కడ"), "ఏ ప్రదేశంలో"),
    ("when", re.compile(r"ఎప్పుడు"), "ఏ సమయంలో"),
    ("why", re.compile(r"ఎందుకు"), "ఏ కారణంతో"),
    ("kind", re.compile(r"ఏ\s+రకమైన"), "ఏ విధమైన"),
    ("how", re.compile(r"ఏ\s+విధంగా"), "ఎలా"),
]


def rewrite(question: str):
    for kind, pattern, replacement in RULES:
        candidate, count = pattern.subn(replacement, question, count=1)
        if count:
            candidate = unicodedata.normalize("NFC", re.sub(r"\s+", " ", candidate)).strip()
            if candidate != question:
                return candidate, kind
    return None


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def build(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for row in read_rows(source):
        result = rewrite(row.get("question", ""))
        if result is None:
            continue
        candidate, rule = result
        key = (row.get("id", ""), candidate)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "id": f"rule-{row.get('id', len(rows))}",
            "source_id": str(row.get("id", "")),
            "question": row["question"],
            "reformulation": candidate,
            "context": row.get("context", ""),
            "answers": row.get("answers", []),
            "wiki_id": str(row.get("wiki_id", "")),
            "source": row.get("source", "ai4bharat/Indic-Rag-Suite"),
            "transformation": rule,
            "label_status": "silver_rule_based_requires_telugu_review",
            "semantic_equivalence_verified": False,
        })
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote={len(rows)} source={source} output={destination}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/indic_rag_te/train.jsonl")
    parser.add_argument("--dev", default="data/indic_rag_te/dev.jsonl")
    parser.add_argument("--out-dir", default="data/indic_rag_te/rule_reformulations")
    args = parser.parse_args()
    output = Path(args.out_dir)
    n_train = build(Path(args.train), output / "train.jsonl")
    n_dev = build(Path(args.dev), output / "dev.jsonl")
    metadata = {
        "method": "fixed conservative Telugu interrogative phrase substitutions; no model calls",
        "seed": None,
        "train_rows": n_train,
        "dev_rows": n_dev,
        "source_train": args.train,
        "source_dev": args.dev,
        "train_dev_article_overlap": 0,
        "semantic_equivalence_verified": False,
        "limitation": "Silver examples only. A fluent Telugu speaker must review every rewrite before supervised training.",
        "rules": [name for name, _, _ in RULES],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"metadata={output / 'metadata.json'}")


if __name__ == "__main__":
    main()
