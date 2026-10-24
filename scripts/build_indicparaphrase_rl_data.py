"""Expand the Telugu reformulation references from AI4Bharat IndicParaphrase.

Source: ``data/external/te_paraphrase/{train,dev,test}_te.jsonl`` pairs from
``ai4bharat/IndicParaphrase`` (Telugu split, noncommercial CC-BY-NC license —
research use only). The corpus contains ~596k paraphrase pairs overall, of
which ~47k have question-form text on both sides; those are the only usable
reformulation references for this project.

Output format matches ``data/indic_rag_te/user_provided_reformulations.jsonl``
so ``load_clean_splits`` (and therefore the REINFORCE/SFT trainers) can merge
them without any changes:

    {"original_question": str, "reformulations": [str, ...], "gold_answer": None}
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

_FOREIGN_SCRIPTS = re.compile(r"[\u0900-\u0bff\u0d00-\u0d7f]")
_TELUGU = re.compile(r"[\u0c00-\u0c7f]")
_QUESTION_WORD = re.compile(
    r"ఎవరు|ఏది|ఏమిటి|ఏమి|ఎక్కడ|ఎప్పుడు|ఎంత|ఎన్ని|ఎలా|ఎవరి|ఏవి|ఏపు|ఎందుక|ఏ తరహ|ఎందరు")


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text or "")).split()).strip()


def is_question(text: str) -> bool:
    text = text.strip()
    return bool(_QUESTION_WORD.search(text)) or text.endswith("?")


def clean(text: str, min_len: int, max_len: int) -> str | None:
    text = norm(text).rstrip(" .?").removesuffix("?")
    if not text:
        return None
    if len(text) < min_len or len(text) > max_len:
        return None
    if not _TELUGU.search(text) or _FOREIGN_SCRIPTS.search(text):
        return None
    return text + "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/external/te_paraphrase")
    ap.add_argument("--out", default="data/indic_rag_te/indic_paraphrase_question_pairs.jsonl")
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=140)
    ap.add_argument("--min-references", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    seen = set()
    written = pairs = 0
    with open(out, "w", encoding="utf-8") as stream:
        for split in ("train", "dev", "test"):
            path = Path(args.source) / f"{split}_te.jsonl"
            for line in open(path, encoding="utf-8"):
                record = json_parse(line)
                if record is None:
                    continue
                written += 1
                source = clean(record["input"], args.min_len, args.max_len)
                if source is None or not is_question(record["input"]):
                    continue
                references = []
                for text in [record["target"], *record.get("references", [])]:
                    cleaned = clean(text, args.min_len, args.max_len)
                    if cleaned and cleaned not in references:
                        references.append(cleaned)
                if len(references) < args.min_references:
                    continue
                key = unicodedata.normalize("NFC", source).casefold()
                if key in seen:
                    continue
                seen.add(key)
                stream.write(f"{json_line(source, references)}\n")
                pairs += 1
    print(f"scanned={written} question_pairs={pairs} written={out}")


def json_parse(line: str):
    import json
    try:
        return json.loads(line)
    except Exception:
        return None


def json_line(source: str, references: list[str]) -> str:
    import json
    return json.dumps({"original_question": source, "reformulations": references,
                       "gold_answer": None}, ensure_ascii=False)


if __name__ == "__main__":
    main()
