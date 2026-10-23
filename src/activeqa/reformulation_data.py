"""Load and conservatively filter Telugu question-reformulation references."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

_FOREIGN_SCRIPTS = re.compile(r"[\u0900-\u0bff\u0d00-\u0d7f]")
_TELUGU = re.compile(r"[\u0c00-\u0c7f]")
_STOP = {"ఏది", "ఏమిటి", "ఏమి", "ఎవరు", "ఎప్పుడు", "ఎక్కడ", "ఎలా", "ఎన్ని",
         "ఎంత", "పేరు", "చెప్పండి", "తెలుపుము", "పేర్కొనండి", "ఏ", "లో", "కి",
         "కు", "ను", "తో", "అని", "గా", "మరియు", "ఒక", "ఈ", "ఆ", "దాని",
         "అది", "ఉన్న", "చేసిన", "చేశారు", "ఏమని", "నుండి", "మధ్య"}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text or "")).split()).strip()


def _question_form(text: str) -> str:
    text = _norm(text).rstrip(" .!؟")
    return text + "?" if text else ""


def _key(text: str) -> str:
    return _norm(text).casefold().rstrip(" ?.!؟")


def _kind(text: str) -> str:
    q = text.casefold()
    if re.search(r"శతాబ్ద", q): return "century"
    if re.search(r"ఏ తేదీ|తేదీన|తేదీ ఏమి|ఏ రోజున", q): return "date"
    if re.search(r"ఏ సంవత్సర|సంవత్సరం|సంవత్సరంలో", q): return "year"
    if re.search(r"ఎన్ని|సంఖ్య ఎంత|ఎంత మంది", q): return "count"
    if re.search(r"ఎంత|ఎన్ని", q): return "amount"
    if re.search(r"ఎవరు|ఎవరిని|ఎవరి", q): return "person"
    if re.search(r"పేరు\s*(?:చెప్పండి|తెలుపుము|పేర్కొనండి|ఏమిటి|ఏమని)", q):
        return "person" if re.search(r"వ్యక్తి|రచయిత|ప్రధాని|రాష్ట్రపతి|కళాకార|నిర్మించిన", q) else "thing"
    if re.search(r"ఎక్కడ|ఏ ప్రదేశ|ఏ నగరంలో", q): return "place"
    if re.search(r"ఎందుకు", q): return "reason"
    if re.search(r"ఎలా|ఏ విధంగా", q): return "manner"
    if re.search(r"ఎప్పుడు|ఏ కాలంలో", q): return "time"
    if re.search(r"ఏది|ఏమిటి|ఏమి|పేరు ఏమిటి|ఏ నగరం", q): return "thing"
    return "unknown"


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[\u0c00-\u0c7f]+|[A-Za-z0-9]+", text.casefold())
    return {w for w in words if len(w) > 2 and w not in _STOP and not w.isdigit()}


def valid_pair(question: str, target: str) -> tuple[bool, str]:
    q, t = _question_form(question), _question_form(target)
    if not q or not t: return False, "empty_question_or_target"
    if not re.search(r"ఏది|ఏమిటి|ఏమి|ఎవరు|ఎవరిని|ఎవరి|ఎప్పుడు|ఎక్కడ|ఎలా|ఎన్ని|ఎంత|ఎందుకు|ఏ సంవత్సర|ఏ తేదీ|శతాబ్ద|పేరు", q):
        return False, "source_not_question"
    # Declarative request forms like “... పేరు తెలుపుము.” count as questions
    # after normalization; statements without an interrogative/request are rejected.
    if not re.search(r"ఏది|ఏమిటి|ఏమి|ఎవరు|ఎవరిని|ఎవరి|ఎప్పుడు|ఎక్కడ|ఎలా|ఎన్ని|ఎంత|ఎందుకు|ఏ సంవత్సర|ఏ తేదీ|శతాబ్ద|పేరు|చెప్పండి|తెలుపుము|పేర్కొనండి", t):
        return False, "target_not_question"
    if _key(q) == _key(t): return False, "identity_copy"
    if _FOREIGN_SCRIPTS.search(t): return False, "foreign_script"
    if len(_TELUGU.findall(t)) < 4: return False, "too_little_telugu"
    qkind, tkind = _kind(q), _kind(t)
    if qkind == "unknown" or tkind == "unknown": return False, "unknown_question_type"
    # A person/name rewrite is fine, but changing a person question into a date,
    # or a century question into an unspecified time question, is not.
    compatible = {"person": {"person"}, "century": {"century"},
                  "year": {"year", "date"}, "date": {"date", "year"},
                  "time": {"time", "date", "year", "century"},
                  "count": {"count"}, "amount": {"amount", "count"},
                  "place": {"place"}, "reason": {"reason"},
                  "manner": {"manner"}, "thing": {"thing"}}
    if tkind not in compatible.get(qkind, {qkind}): return False, "answer_type_changed"
    anchors = _content_words(q)
    overlap = anchors & _content_words(t)
    # English titles and proper names should be copied verbatim. Telugu case
    # endings can vary, so also accept a shared stem of four characters.
    if anchors:
        matched = set(overlap)
        target_words = _content_words(t)
        for word in anchors - matched:
            if any(word[:4] == other[:4] for other in target_words
                   if len(word) >= 4 and len(other) >= 4):
                matched.add(word)
        if len(matched) / len(anchors) < 0.45:
            return False, "topic_anchor_dropped"
    return True, "accepted"


def load_clean_splits(input_path: str | Path, dev_fraction: float = 0.2,
                      seed: int = 42, output_dir: str | Path | None = None):
    """Filter questionable pairs before splitting by source question."""
    examples: dict[str, dict] = {}
    rejected: dict[str, int] = {}
    with Path(input_path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip(): continue
            row = json.loads(line)
            question = _question_form(row.get("original_question"))
            answer = _norm(row.get("gold_answer"))
            if not question or not answer:
                rejected["missing_question_or_answer"] = rejected.get("missing_question_or_answer", 0) + 1
                continue
            key = _key(question)
            item = examples.setdefault(key, {"question": question, "answer": answer,
                                             "references": [], "source_ids": []})
            for raw_target in row.get("reformulations", []):
                target = _question_form(raw_target)
                ok, reason = valid_pair(question, target)
                if not ok:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                if _key(target) not in {_key(x) for x in item["references"]}:
                    item["references"].append(target)
            item["source_ids"].append(str(row.get("id", line_no)))
    examples = {k: v for k, v in examples.items() if v["references"]}
    keys = sorted(examples)
    random.Random(seed).shuffle(keys)
    n_dev = max(1, round(len(keys) * dev_fraction)) if len(keys) > 1 else 0
    dev_keys = set(keys[:n_dev])
    splits = {"train": [], "dev": []}
    for key, item in examples.items():
        split = "dev" if key in dev_keys else "train"
        source_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        for i, reference in enumerate(item["references"], 1):
            splits[split].append({"id": f"{source_id}-{i}", "source_id": source_id,
                                  "question": item["question"], "reformulation": reference,
                                  "references": item["references"], "answers": [item["answer"]],
                                  "label_status": "user_provided_filtered_not_human_verified"})
    if output_dir:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        for split, rows in splits.items():
            with (out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out / "filter_report.json").open("w", encoding="utf-8") as stream:
            json.dump({"unique_questions": len(examples), "train_pairs": len(splits["train"]),
                       "dev_pairs": len(splits["dev"]), "rejected_pairs_by_reason": rejected,
                       "human_verified": False}, stream, ensure_ascii=False, indent=2)
    return splits["train"], splits["dev"], rejected
