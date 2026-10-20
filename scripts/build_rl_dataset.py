"""Build a reproducible, small Telugu QA + reformulation dataset for ActiveQA.

Source QA pairs come from the Telugu train config of AI4Bharat Indic-Rag-Suite.
The original held-out benchmark remains IndicQA and is never sampled for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = "https://datasets-server.huggingface.co"
DATASET = "ai4bharat/Indic-Rag-Suite"


def compact(text: str) -> str:
    """Unicode-aware comparison key that ignores whitespace and punctuation."""
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] in {"L", "N", "M"})


def get_json(url: str, retries: int = 4) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "activeqa-telugu/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def dataset_rows(seed: int, max_pages: int, page_size: int, cache_dir: Path):
    size_url = ROOT + "/size?" + urllib.parse.urlencode({"dataset": DATASET})
    sizes = get_json(size_url)["size"]["configs"]
    count = next(x["num_rows"] for x in sizes if x["config"] == "te")
    offsets = list(range(0, count, page_size))
    random.Random(seed).shuffle(offsets)
    for page_num, offset in enumerate(offsets[:max_pages], 1):
        cache_file = cache_dir / f"te_train_{offset}_{page_size}.json"
        if cache_file.exists():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            params = {"dataset": DATASET, "config": "te", "split": "train",
                      "offset": offset, "length": min(page_size, count - offset)}
            result = get_json(ROOT + "/rows?" + urllib.parse.urlencode(params))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(f"fetched page {page_num}/{min(max_pages, len(offsets))} (offset {offset})",
              flush=True)
        for item in result.get("rows", []):
            yield item["row"]


def normalize_row(row: dict, min_context_chars: int) -> dict | None:
    question = unicodedata.normalize("NFC", str(row.get("question") or "")).strip()
    answer = unicodedata.normalize("NFC", str(row.get("answer") or "")).strip()
    context = unicodedata.normalize("NFC", str(row.get("paragraph") or "")).strip()
    if not question or not answer or len(context) < min_context_chars or not row.get("wiki_id"):
        return None
    # Keep Telugu questions and answers with an extractable source answer so QA-F1
    # provides a meaningful reward instead of punishing impossible examples.
    if not re.search(r"[\u0c00-\u0c7f]", question):
        return None
    answer_key, context_key = compact(answer), compact(context)
    if len(answer_key) < 2 or answer_key not in context_key:
        return None
    signature = "\0".join((question, answer, context))
    row_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"indic-rag-te-{row_id}",
        "question": question,
        "context": context,
        "answers": [answer],
        "title": str(row.get("title") or ""),
        "wiki_id": str(row["wiki_id"]),
        "url": str(row.get("url") or ""),
        "source": DATASET,
        "source_config": "te",
        "source_split": "train",
        "source_model": str(row.get("model_name") or ""),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--dev-size", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Maximum 100-row random-offset pages fetched (default: 100).")
    parser.add_argument("--page-size", type=int, choices=(10, 20, 50, 100), default=100)
    parser.add_argument("--min-context-chars", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/indic_rag_te")
    args = parser.parse_args()
    if args.train_size < 1 or args.dev_size < 1:
        parser.error("train-size and dev-size must be positive")

    target = args.train_size + args.dev_size
    rows: dict[str, dict] = {}
    examined = 0
    for source_row in dataset_rows(args.seed, args.max_pages, args.page_size,
                                   Path("data/cache/indic_rag_api_pages")):
        examined += 1
        clean = normalize_row(source_row, args.min_context_chars)
        if clean:
            rows.setdefault(clean["id"], clean)
        if len(rows) >= target * 2:
            break
    if len(rows) < target:
        raise RuntimeError(f"Only {len(rows)} valid unique rows from {examined} examined; "
                           f"need {target}. Re-run with a larger --max-pages.")

    # Split by Wikipedia article to prevent near-duplicate paragraphs/questions
    # from one article appearing in both train and development.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows.values():
        groups[row["wiki_id"]].append(row)
    group_items = list(groups.items())
    random.Random(args.seed).shuffle(group_items)
    train, dev = [], []
    train_wikis, dev_wikis = [], []
    for wiki_id, group in group_items:
        if len(train) < args.train_size:
            train.extend(group)
            train_wikis.append(wiki_id)
        elif len(dev) < args.dev_size:
            dev.extend(group)
            dev_wikis.append(wiki_id)
        if len(train) >= args.train_size and len(dev) >= args.dev_size:
            break
    if len(dev) < args.dev_size:
        raise RuntimeError(f"Article-group split yielded only {len(dev)} dev rows; "
                           "increase --max-pages or reduce requested sizes.")
    train.sort(key=lambda r: r["id"])
    dev.sort(key=lambda r: r["id"])

    out = Path(args.output_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    metadata = {
        "dataset": DATASET,
        "dataset_url": "https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite",
        "viewer_api": ROOT,
        "config": "te",
        "source_split": "train",
        "source_model": "Meta-Llama-3.3-70B-Instruct (as listed in source records)",
        "license": "MIT (source dataset card)",
        "seed": args.seed,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_articles": len(train_wikis),
        "dev_articles": len(dev_wikis),
        "article_overlap": len(set(train_wikis) & set(dev_wikis)),
        "rows_examined": examined,
        "unique_valid_rows_sampled": len(rows),
        "answer_in_context_filter": "Unicode letters/numbers/marks compacted; answer must occur in paragraph",
        "minimum_context_characters": args.min_context_chars,
        "split_method": "seeded random-offset sampling, then article-grouped train/dev split",
        "limitation": "Synthetic QA from Telugu Wikipedia; sampled subset and automatic answer/context checks require human quality review.",
        "created_unix": int(time.time()),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
