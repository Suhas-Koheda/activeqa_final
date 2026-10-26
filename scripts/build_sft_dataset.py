#!/usr/bin/env python3
"""Build the SFT pairs file from nlpctx/telugu-qa-codemixed — and nothing else.

This is the ONLY training data now (user decision: "use the dataset as it is"):

    https://huggingface.co/datasets/nlpctx/telugu-qa-codemixed

2,091 docs. Each doc holds one question rewritten through stacked levels:

    level_0   English seed            -> never a pair source or target
    level_1-4 Telugu code-mixed       -> paraphrase pairs within the doc
    level_5   native Telugu rewrite   -> the most fluent form when complete

The parquet is loaded as-is and its cells are never edited. The only
selection applied is mechanical:

  * a cell is dropped when it is not a complete question (single trailing "?"),
    is one of the known junk values (level_, nan, null, empty) or contains
    Tamil script;
  * translit pairs join complete Latin-script level_1..4 cells of the same doc;
  * native pairs join a level_1..4 source with its complete level_5 cell.

No regexes, no scoring, no filtering heuristics — just the dataset, shaped
for the trainer.

Output: <out>.jsonl {original_question, reformulations, sources, transformations}
        <out>_report.json with the pair/skip counts.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_URL = ("https://huggingface.co/datasets/nlpctx/telugu-qa-codemixed"
          "/resolve/main/data/train-00000-of-00001.parquet")
DEFAULT_PARQUET = ROOT / "data" / "external" / "telugu_qa_codemixed" / "train.parquet"

JUNK = {"", "level_", "nan", "null", "none"}


def has_script(s: str, lo: int, hi: int) -> bool:
    """True when s contains any character in the block [lo, hi]. No regex."""
    return any(lo <= ord(ch) <= hi for ch in s)


def complete_question(s) -> bool:
    """A usable source/target cell: complete, single trailing '?', no junk."""
    return (isinstance(s, str) and s.strip() not in JUNK
            and s.count("?") == 1 and s.rstrip().endswith("?"))


def usable_level(s) -> bool:
    """A level_1..4 cell: complete AND Latin script (no Telugu, no Tamil)."""
    return (complete_question(s) and not has_script(s, 0x0B80, 0x0C7F)
            and any(ch.isalpha() for ch in s))


def usable_native(s) -> bool:
    """A level_5 cell: complete, TELUGU script, no Tamil."""
    return (complete_question(s) and has_script(s, 0x0C00, 0x0C7F)
            and not has_script(s, 0x0B80, 0x0BFF))


def ensure_parquet(path: Path) -> Path:
    """Download the parquet verbatim (byte-for-byte) when it is missing."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading the dataset as-is from {HF_URL} …")
    urllib.request.urlretrieve(HF_URL, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET),
                    help="nlpctx/telugu-qa-codemixed parquet (downloaded "
                         "verbatim from Hugging Face when missing)")
    ap.add_argument("--out", default=str(ROOT / "data" / "indic_rag_te" / "sft_pairs_v2.jsonl"))
    args = ap.parse_args()

    import pandas as pd
    parquet = ensure_parquet(Path(args.parquet))
    df = pd.read_parquet(parquet)   # the dataset exactly as published
    print(f"loaded {len(df)} docs, columns: {list(df.columns)}")

    records: dict[str, dict] = {}
    skips = {"native_truncated": 0, "native_missing": 0, "native_tamil": 0,
             "level_incomplete_or_junk": 0, "docs_without_pairs": 0}

    for _, row in df.iterrows():
        levels = [row[f"level_{k}"] for k in range(1, 5)]

        # per-doc native Telugu target (level_5), used only when complete
        native = row["level_5"]
        if isinstance(native, str) and has_script(native, 0x0B80, 0x0BFF):
            skips["native_tamil"] += 1
            native = None
        elif native is None or (isinstance(native, float) and pd.isna(native)):
            skips["native_missing"] += 1
            native = None
        elif not complete_question(native):
            skips["native_truncated"] += 1
            native = None

        latin = [(i, v) for i, v in enumerate(levels, start=1) if usable_level(v)]
        if not latin:
            skips["docs_without_pairs"] += 1
        for i, src in latin:
            refs = [(j, v) for j, v in latin if j != i]
            if native is not None:
                refs.append((5, native))
            for j, tgt in refs:
                if not (usable_level(tgt) or usable_native(tgt)):
                    skips["level_incomplete_or_junk"] += 1
                    continue
                rec = records.setdefault(src, {
                    "original_question": src, "reformulations": [],
                    "sources": ["codemixed"], "transformations": {}})
                if tgt not in rec["reformulations"]:
                    rec["reformulations"].append(tgt)
                    rec["transformations"][tgt] = f"codemixed lvl{i}->{j}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records.values():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    per_q = [len(r["reformulations"]) for r in records.values()]
    report = {
        "dataset": "nlpctx/telugu-qa-codemixed (used as-is)",
        "parquet": str(parquet),
        "out_file": str(out_path),
        "questions": len(records),
        "pairs": sum(per_q),
        "skipped": {k: v for k, v in skips.items() if v},
        "pairs_per_question_max": max(per_q) if per_q else 0,
    }
    out_path.with_name(out_path.stem + "_report.json") \
        .write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n── sample pairs (translit) ──")
    shown = 0
    for rec in records.values():
        tgt = next((t for t in rec["reformulations"] if t == rec["reformulations"][0]), None)
        if tgt is None:
            continue
        shown += 1
        print(f"[{rec['transformations'].get(tgt, '?')}] "
              f"{rec['original_question']}\n   -> {tgt}")
        if shown >= 8:
            break
    print("\n── sample pairs (native Telugu targets) ──")
    shown = 0
    for rec in records.values():
        tgt = next((t for t in rec["reformulations"] if usable_native(t)), None)
        if tgt is None:
            continue
        shown += 1
        print(f"[{rec['transformations'].get(tgt, '?')}] "
              f"{rec['original_question']}\n   -> {tgt}")
        if shown >= 8:
            break


if __name__ == "__main__":
    main()
