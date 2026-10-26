"""Build Telugu reformulation SFT dataset v2 with generated, entity-safe targets.

The organic paraphrase corpus is not meaning-preserving (median content-word
Jaccard 0.143; entity corruption like "ట్రంప్ → ట్రూమాన్" appears in the data),
so its targets are never used. Instead every target is GENERATED:

  1. Copy the source question verbatim.
  2. Substitute only a whitelisted interrogative token/phrase, matched on
     token boundaries, so entities and content words cannot change.
  3. Reject the pair unless it passes a strict validator.

Token-boundary matching makes the indefinite forms (ఎక్కడా, ఎక్కడో,
ఎప్పుడైనా, ఎందుకంటే, ఎవరైనా, ఎంతో, ఎలాగైనా, …) unreachable by construction;
they are also kept in an explicit deny-list as a defensive guard.

Data sources (real questions first):
  A. train.jsonl                     804 real QA questions (dev excluded)
  B. user_provided_reformulations    curated multi-rewrite rows
  C. rule_reformulations train +     curated single pairs
     user_reformulations train
  D. organic corpus sources with     tier "corpus_rule", capped by
     rule-generated targets          --max-corpus-questions
  E. nlpctx/telugu-qa-codemixed      tier "codemixed_translit" + "codemixed_native":
     (Hugging Face) code-mixed       reformulation pairs mined across the
                                     level_1..4 chains and, when complete,
                                     into the level_5 native-Telugu rewrite.
                                     The dataset is consumed AS-IS: cells are
                                     never edited; validation only selects
                                     which pairs are usable (skips truncated,
                                     Tamil-script and junk cells).

Output: <out>.jsonl  {original_question, reformulations, gold_answer,
                      sources, transformations}
        <out>_report.json with counts, rule hits and rejection reasons.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "\u0C00-\u0C7F"          # Telugu block

# ── transformation rules ─────────────────────────────────────────────────────
# Telugu combining marks (matras, virama, anusvara) are category Mn and are
# NOT part of \w in Python's re, so \b cannot separate "ఎక్కడ" from
# "ఎక్కడికి". Boundaries are enforced explicitly over the Telugu block.
ZH = r"[\u200c\u200d]*"                                        # zero-width chars
TEL = r"[\u0C00-\u0C7F\u200c\u200d]"
TEXT_END = r"(?!" + TEL + r")"                                 # after a full word
LA = lambda w: fr"(?<!{TEL}){w}(?!{TEL})"                      # Telugu-word edges
# idiom guards: the following word must END here (a full word; a matra suffix
# like ఉన్నాయి disqualifies), so real question words stay unmolested
PHRASE_GUARD = r"(?!\s*(?:చూసినా|వెళ్ళినా|అయినా|నుంచి|నుండి)" + TEXT_END + r")"
ID_GUARD = r"(?!\s*(?:ఉన్నా|అయినా)" + TEXT_END + r")"          # ఎలా ఉన్నా … (idiom)
PERSON_END = r"(?![\s?,.]*$)"                                  # ఎవరు as predicate

RULES: list[tuple[str, re.Pattern, list[str]]] = [
    # location phrases (longest first)
    ("where_from", re.compile(LA("ఎక్కడి") + ZH + r"\s*నుంచి"), ["ఏ ప్రదేశం నుంచి"]),
    ("where_from", re.compile(LA("ఎక్కడ") + ZH + r"\s*నుంచి" + PHRASE_GUARD), ["ఏ ప్రదేశం నుంచి"]),
    ("where_from", re.compile(LA("ఎక్కడి") + ZH + r"\s*నుండి"), ["ఏ ప్రదేశం నుంచి"]),
    ("where_from", re.compile(LA("ఎక్కడ") + ZH + r"\s*నుండి" + PHRASE_GUARD), ["ఏ ప్రదేశం నుంచి"]),
    ("where_from", re.compile(LA("ఏ") + ZH + r"\s*ప్రదేశం\s*నుంచి"), ["ఎక్కడి నుంచి"]),
    ("where_from", re.compile(LA("ఏ") + ZH + r"\s*ప్రదేశం\s*నుండి"), ["ఎక్కడి నుంచి"]),
    ("where_to", re.compile(LA("ఎక్కడికి")), ["ఏ ప్రదేశానికి"]),
    ("where_to", re.compile(LA("ఏ") + ZH + r"\s*ప్రదేశానికి"), ["ఎక్కడికి"]),
    # where (token)
    ("where", re.compile(LA("ఎక్కడ") + PHRASE_GUARD),
     ["ఏ ప్రదేశంలో", "ఏ చోట", "ఏ ప్రాంతంలో"]),
    ("where", re.compile(LA("ఏ") + ZH + r"\s*ప్రదేశంలో"), ["ఎక్కడ"]),
    ("where", re.compile(LA("ఏ") + ZH + r"\s*చోట" + TEXT_END), ["ఎక్కడ"]),
    ("where", re.compile(LA("ఏ") + ZH + r"\s*ప్రాంతంలో"), ["ఎక్కడ"]),
    # when
    ("when", re.compile(LA("ఎప్పుడు") + PHRASE_GUARD), ["ఏ సమయంలో"]),
    ("when", re.compile(LA("ఏ") + ZH + r"\s*సమయంలో"), ["ఎప్పుడు"]),
    ("year", re.compile(LA("ఏ") + ZH + r"\s*సంవత్సరంలో"), ["ఏ ఏడాదిలో"]),
    ("year", re.compile(LA("ఏ") + ZH + r"\s*ఏడాదిలో"), ["ఏ సంవత్సరంలో"]),
    # why
    ("why", re.compile(LA("ఎందుకు") + PHRASE_GUARD), ["ఏ కారణంగా", "ఏ కారణంతో"]),
    ("why", re.compile(LA("ఏ") + ZH + r"\s*కారణంగా"), ["ఎందుకు"]),
    ("why", re.compile(LA("ఏ") + ZH + r"\s*కారణంతో"), ["ఎందుకు"]),
    # how / manner
    ("how", re.compile(LA("ఎలా") + ID_GUARD), ["ఏ విధంగా", "ఏ విధముగా", "ఏ పద్ధతిలో"]),
    ("how", re.compile(LA("ఏ") + ZH + r"\s*విధంగా"), ["ఎలా"]),
    ("how", re.compile(LA("ఏ") + ZH + r"\s*విధముగా"), ["ఎలా"]),
    ("how", re.compile(LA("ఏ") + ZH + r"\s*పద్ధతిలో"), ["ఎలా"]),
    # what kind of
    ("kind", re.compile(LA("ఎలాంటి")), ["ఏ రకమైన", "ఏ విధమైన"]),
    ("kind", re.compile(LA("ఏ") + ZH + r"\s*రకమైన"), ["ఎలాంటి"]),
    ("kind", re.compile(LA("ఏ") + ZH + r"\s*విధమైన"), ["ఎలాంటి"]),
    ("kind", re.compile(LA("ఏ") + ZH + r"\s*రకమైన"), ["ఏ విధమైన"]),
    # what
    ("what", re.compile(LA("ఏమిటి")), ["ఏంటి"]),
    ("what", re.compile(LA("ఏంటి")), ["ఏమిటి"]),
    ("what", re.compile(LA("ఏమి") + PHRASE_GUARD), ["ఏం"]),
    ("what", re.compile(LA("ఏం") + PHRASE_GUARD), ["ఏమి"]),
    # who / whom-family ("ఏ వ్యక్తి" only where ఎవరు is not the predicate)
    ("who", re.compile(LA("ఎవరు") + PERSON_END), ["ఏ వ్యక్తి"]),
    ("who", re.compile(LA("ఎవరికి")), ["ఏ వ్యక్తికి"]),
    ("who", re.compile(LA("ఏ") + ZH + r"\s*వ్యక్తికి"), ["ఎవరికి"]),
    ("who", re.compile(LA("ఎవరిని")), ["ఏ వ్యక్తిని"]),
    ("who", re.compile(LA("ఏ") + ZH + r"\s*వ్యక్తిని"), ["ఎవరిని"]),
    ("who", re.compile(LA("ఎవరితో")), ["ఏ వ్యక్తితో"]),
    ("who", re.compile(LA("ఏ") + ZH + r"\s*వ్యక్తితో"), ["ఎవరితో"]),
    # how many (people)
    ("quantity", re.compile(LA("ఎంతమంది")), ["ఎన్ని మంది"]),
    ("quantity", re.compile(LA("ఎన్ని") + ZH + r"\s*మంది"), ["ఎంతమంది"]),
]

# Indefinite/idiom tokens that must never be treated as interrogatives.
INDEFINITE = {
    "ఎక్కడా", "ఎక్కడో", "ఎక్కడైనా", "ఎప్పుడైనా", "ఎప్పుడో", "ఎందుకంటే",
    "ఎందుకో", "ఎవరైనా", "ఎవరూ", "ఎవరికీ", "ఎవరినీ", "ఎవరికైనా", "ఎంతో",
    "ఎంతగానో", "ఎంతైనా", "ఎలాగైనా", "ఎలాగూ", "ఏదైనా", "ఏదో", "ఏమీ",
}
# ── validator ────────────────────────────────────────────────────────────────

NUM_OR_LATIN = re.compile(r"[0-9]+|[A-Za-z]{2,}")
TOKEN = re.compile(fr"[{T}]+|[A-Za-z]+|[0-9]+")
RARE_DF = 5   # tokens in fewer than RARE_DF pool questions are "rare"

QTYPES = {
    "person": r"ఎవర|వ్యక్తి",
    "place": r"ఎక్కడ|ప్రదేశ|చోట|ప్రాంత",
    "time": r"ఎప్పుడ|సమయ|సంవత్సర|ఏడాది|శతాబ్ద|రోజ|నెల",
    "quantity": r"ఎంత|ఎన్ని",
    "reason": r"ఎందుక|కారణ",
    "manner": r"ఎలా|విధ|పద్ధతి|రకమ|తరహ",
    "what": r"ఏమి|ఏం|ఏది",
}


def question_type(text: str) -> frozenset[str]:
    return frozenset(k for k, pat in QTYPES.items() if re.search(pat, text))


# Tokens the generator may legally substitute for each other. Jaccard is
# computed over canonical forms so a whitelisted swap can't pass as drift.
CANON_GROUPS = (
    ("ఎక్కడ", "ఎక్కడి", "ఎక్కడికి", "ప్రదేశంలో", "ప్రదేశం", "చోట", "ప్రాంతంలో",
     "ప్రదేశానికి", "నుంచి", "నుండి"),
    ("ఎప్పుడు", "సమయంలో", "సమయం", "వేళ"),
    ("ఎలా", "విధంగా", "విధముగా", "పద్ధతిలో", "విధమైన", "రకమైన", "ఎలాంటి",
     "రకముగా"),
    ("ఎందుకు", "కారణంగా", "కారణంతో"),
    ("ఏమి", "ఏం", "ఏమిటి", "ఏంటి"),
    ("ఎవరు", "వ్యక్తి", "ఎవరికి", "వ్యక్తికి", "ఎవరిని", "వ్యక్తిని",
     "ఎవరితో", "వ్యక్తితో"),
    ("ఎంతమంది", "ఎన్నిమంది"),
)
CANON: dict[str, str] = {}
for _i, _group in enumerate(CANON_GROUPS):
    for _w in _group:
        CANON[_w] = f"§{_i}"


def content_tokens(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text) if len(t) > 2}


def normalize_ws(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return text.strip()


def text_ok(text: str) -> bool:
    """Usable question text: exactly one trailing '?', no junk punctuation."""
    if not text or text.count("?") != 1 or not text.endswith("?"):
        return False
    if re.search(r"\?\?|\? ?\.|\.\s*\?", text):
        return False
    if "  " in text:
        return False
    return True


def validate_pair(src: str, tgt: str, freq: Counter, curated: bool = False) -> str | None:
    """Return a rejection reason, or None when the pair is acceptable.

    Generated pairs are near-copies, so they get the strict thresholds.
    Curated pairs are true paraphrases, so Jaccard/rare-retention floors are
    relaxed (the other checks — entities, question type, punctuation,
    length — stay identical).
    """
    if not text_ok(src):
        return "bad_source_text"
    if not text_ok(tgt):
        return "bad_target_text"
    if tgt == src:
        return "exact_copy"
    j_min, r_min = (0.40, 0.50) if curated else (0.60, 0.80)
    ratio = len(tgt) / max(1, len(src))
    if not 0.7 <= ratio <= 1.4:
        return "length_ratio"
    digits_src = set(NUM_OR_LATIN.findall(src))
    digits_tgt = set(NUM_OR_LATIN.findall(tgt))
    if not digits_src <= digits_tgt:
        return "entity_dropped"
    if digits_tgt - digits_src:
        return "entity_invented"
    if question_type(src) != question_type(tgt):
        return "question_type"
    s_tokens, t_tokens = content_tokens(src), content_tokens(tgt)
    # Canonicalize interrogative synonyms so whitelisted swaps
    # (ఎలాంటి↔ఏ రకమైన, ఏ ప్రదేశం నుంచి↔ఎక్కడి నుంచి, …) don't read as drift.
    s_tokens = {CANON.get(t, t) for t in s_tokens}
    t_tokens = {CANON.get(t, t) for t in t_tokens}
    union = s_tokens | t_tokens
    jac = len(s_tokens & t_tokens) / len(union) if union else 0.0
    if jac < j_min:
        return "jaccard"
    rare = [t for t in content_tokens(src) if freq.get(t, 0) < RARE_DF]
    if rare:
        kept = sum(1 for t in rare if t in content_tokens(tgt)) / len(rare)
        if kept < r_min:
            return "rare_retention"
    # Sentence-final dots only: Telugu abbreviation dots sit inside a word
    # (టీ.శేషంపల్లె, డి.ఎస్.పి) with no following space. A real sentence
    # break is a dot, then whitespace, then another sentence.
    if re.search(r"[.।]\s+\S", tgt):
        return "multi_sentence"
    return None


# ── generation ───────────────────────────────────────────────────────────────

def variants_for(question: str, limit: int) -> list[tuple[str, str]]:
    """Up to `limit` (rewrite, rule_name) pairs; one substitution per variant."""
    out: list[tuple[str, str]] = []
    for name, pat, reps in RULES:
        m = pat.search(question)
        if not m:
            continue
        for rep in reps:
            cand = normalize_ws(question[: m.start()] + rep + question[m.end():])
            if cand != question and (cand, name) not in out:
                out.append((cand, name))
        if len(out) >= limit:
            break
    return out[:limit]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_answers(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [a for a in raw if a]
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            try:
                return [a for a in ast.literal_eval(s) if a]
            except (ValueError, SyntaxError):
                return [s] if s else []
        return [s] if s else []
    return []


# ── tier E: code-mixed↔native pairs from nlpctx/telugu-qa-codemixed ─────────
TEL_RE = re.compile(r"[\u0C00-\u0C7F]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_WORD = re.compile(r"[A-Za-z]+")
CM_STOP = {"emiti", "ela", "cheyali", "che", "kavali", "ante", "ayit",
           "yundhi", "vallaki", "kosam"}
JUNK_MARK = "LEVEL_"


def cm_valid(x) -> bool:
    """Usable dataset cell: a string, no junk marker, no Tamil script."""
    return (isinstance(x, str) and JUNK_MARK not in x
            and not TAMIL_RE.search(x) and bool(x.strip()))


def cm_matches_native_level5(x) -> bool:
    """The level_5 native-Telugu slice: Telugu script, complete, single '?'."""
    return (isinstance(x, str) and TEL_RE.search(x)
            and not isinstance(x, float) and cm_valid(x)
            and x.count("?") == 1 and x.rstrip().endswith("?"))


CM_HF_URL = ("https://huggingface.co/datasets/nlpctx/telugu-qa-codemixed"
             "/resolve/main/data/train-00000-of-00001.parquet")


def ensure_codemixed_parquet(parquet_path: Path) -> Path | None:
    """Locate the nlpctx/telugu-qa-codemixed parquet, downloading it verbatim
    (as-is, byte-for-byte from the Hugging Face repo) when it is missing."""
    if parquet_path.exists():
        return parquet_path
    import urllib.request
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[codemixed] downloading as-is from {CM_HF_URL}")
    try:
        urllib.request.urlretrieve(CM_HF_URL, parquet_path)
    except OSError as exc:
        print(f"[codemixed] download failed ({exc}); tier skipped")
        return None
    return parquet_path



def cm_anchor_kept(anchor: str, tgt_words: set[str]) -> bool:
    """Morphology-tolerant match: exact hit or prefix of/near-same-length word.

    Plural/suffix variants in code-mixed rewrites (witnesses→witness,
    complaints→complaint) must count as retained anchors. Requiring one side to
    be a prefix of the other, both ≥ 4 chars, with ≤ 3 chars of length
    difference, still rejects unrelated words (police vs visitor etc.).
    """
    if anchor in tgt_words:
        return True
    return any(
        (anchor.startswith(w) or w.startswith(anchor))
        and min(len(anchor), len(w)) >= 4
        and abs(len(anchor) - len(w)) <= 3
        for w in tgt_words
    )


def cm_anchor_retention(src: str, tgt: str) -> float:
    """Share of source Latin anchor words (len ≥ 4, not stopwords) kept."""
    anchors = {w.lower() for w in LATIN_WORD.findall(src)
               if len(w) >= 4 and w.lower() not in CM_STOP}
    if not anchors:
        return 1.0
    tgt_words = {w.lower() for w in LATIN_WORD.findall(tgt)}
    return sum(1 for a in anchors if cm_anchor_kept(a, tgt_words)) / len(anchors)


def add_codemixed_tier(records, tier_pair_counts, reject, samples,
                       is_dev, parquet_path) -> int:
    """Ingest nlpctx/telugu-qa-codemixed (2091 docs).

    Each doc chains one question across 6 levels: an English seed
    (level_0, never used as a source — that would teach translation),
    4 progressively Telugu code-mixed rewrites (level_1..4, Latin script)
    and a native-Telugu rewrite (level_5, present but frequently truncated).
    Emits two kinds of pairs:
      codemixed_translit: level_i -> level_j   (i, j in 1..4)
      codemixed_native:   level_i -> level_5   (i in 1..4, level_5 complete)
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    docs_used = 0
    for _, row in df.iterrows():
        cm_levels = [(k, row[f"level_{k}"]) for k in range(1, 5) if cm_valid(row[f"level_{k}"])]
        cm_levels = [(k, normalize_ws(v)) for k, v in cm_levels
                     if v.strip() and v.count("?") == 1 and v.rstrip().endswith("?")]
        clean_cm = [(k, v) for k, v in cm_levels if not TEL_RE.search(v)]
        if not clean_cm:
            continue
        native = row["level_5"]
        if not cm_matches_native_level5(native):
            native = None
        else:
            native = normalize_ws(native)
        if is_dev(native or clean_cm[0][1]):
            continue  # defense in depth: drop only if it matches a dev question
        docs_used += 1
        tier_kind = ("codemixed_translit", "codemixed_native")
        for k, src in clean_cm:
            targets = [(j, v) for j, v in clean_cm if j != k][:3]
            if native is not None:
                targets.append((5, native))
            for _j, tgt in targets:
                is_native = _j == 5
                tier = tier_kind[1 if is_native else 0]
                if src == tgt:
                    reject["exact_copy"] += 1
                    continue
                if not text_ok(src) or not text_ok(tgt):
                    reject["bad_text"] += 1
                    continue
                ratio = len(tgt) / max(1, len(src))
                if not 0.45 <= ratio <= 2.4:
                    reject["length_ratio"] += 1
                    continue
                s_tokens, t_tokens = set(LATIN_WORD.findall(src)), set(LATIN_WORD.findall(tgt))
                # Latin anchors are content bearings in translit; dropped anchors
                # mean the rewrite talks about something else.
                if not is_native and cm_anchor_retention(src, tgt) < 0.5:
                    reject["anchor_dropped"] += 1
                    continue
                if is_native:
                    # entities may transliterate to Telugu script, but any Latin
                    # word the target gains must come from the source
                    if {w.lower() for w in t_tokens} - {w.lower() for w in s_tokens}:
                        reject["entity_invented"] += 1
                        continue
                rec = records.setdefault(src, {
                    "original_question": src, "reformulations": [],
                    "sources": [], "transformations": {}, "gold_answer": []})
                rec["transformations"][tgt] = f"codemixed lvl{k}->{_j}"
                if tgt not in rec["reformulations"]:
                    rec["reformulations"].append(tgt)
                    tier_pair_counts[tier] += 1
                if tier not in rec["sources"]:
                    rec["sources"].append(tier)
    return docs_used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "data" / "indic_rag_te"
    ap.add_argument("--out", default=str(base / "sft_pairs_v2.jsonl"))
    ap.add_argument("--corpus", default=str(base / "indic_paraphrase_question_pairs.jsonl"))
    ap.add_argument("--codemixed",
                    default=str(ROOT / "data" / "external" / "telugu_qa_codemixed" / "train.parquet"),
                    help="nlpctx/telugu-qa-codemixed parquet; tier skipped if missing")
    ap.add_argument("--max-codemixed-pairs", type=int, default=0,
                    help="cap codemixed pairs (0 = unlimited)")
    ap.add_argument("--variants-per-train-question", type=int, default=4)
    ap.add_argument("--variants-per-corpus-question", type=int, default=2)
    ap.add_argument("--max-corpus-questions", type=int, default=12000,
                    help="cap corpus-tier source questions (0 = unlimited)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    for _name, pat, _reps in RULES:  # defensive: no rule key is an indefinite form
        assert not any(w in pat.pattern for w in INDEFINITE), INDEFINITE

    rnd = random.Random(args.seed)

    # dev questions must never appear in training, by id or by text
    dev_rows = load_jsonl(base / "dev.jsonl")
    dev_ids = {r.get("id") for r in dev_rows}
    dev_q = {normalize_ws(r["question"]) for r in dev_rows}

    def is_dev(q: str, source_id: str | None = None) -> bool:
        return normalize_ws(q) in dev_q or (source_id is not None and source_id in dev_ids)

    train_rows = load_jsonl(base / "train.jsonl")
    train_questions: dict[str, list[str]] = {}
    skipped_dev = 0
    for row in train_rows:
        if is_dev(row["question"], row.get("id")):
            skipped_dev += 1
            continue
        train_questions[normalize_ws(row["question"])] = parse_answers(row.get("answers"))

    # curated pairs: (source, tag) -> {target -> gold}
    curated: dict[str, dict[str, str]] = {}
    curated_gold: dict[str, list[str]] = {}
    curated_tier: dict[str, set[str]] = {}

    def add_curated(src: str, tgt: str, tier: str, gold: list[str]) -> None:
        if not src or not tgt or is_dev(src):
            return
        bucket = curated.setdefault(src, {})
        cur_tier = bucket.get(tgt)
        if cur_tier is None or tier == "curated_user":
            bucket[tgt] = tier
        if gold and src not in curated_gold:
            curated_gold[src] = gold

    for row in load_jsonl(base / "rule_reformulations" / "train.jsonl"):
        add_curated(normalize_ws(row["question"]), normalize_ws(row["reformulation"]),
                    "rule_curated", parse_answers(row.get("answers")))
    for row in load_jsonl(base / "user_reformulations" / "train.jsonl"):
        add_curated(normalize_ws(row["question"]), normalize_ws(row["reformulation"]),
                    "user_curated", parse_answers(row.get("answers")))
    for row in load_jsonl(base / "user_provided_reformulations.jsonl"):
        gold = [row["gold_answer"]] if row.get("gold_answer") else []
        for tgt in row.get("reformulations") or []:
            add_curated(normalize_ws(row["original_question"]), normalize_ws(tgt),
                        "curated_user", gold)
    for src, bucket in curated.items():
        curated_tier[src] = {tier for tier in bucket.values()}

    # corpus-tier source questions (targets from the corpus are ignored)
    corpus_rows = load_jsonl(Path(args.corpus))
    corpus_qs = list(dict.fromkeys(normalize_ws(r["original_question"])
                                   for r in corpus_rows))
    if args.max_corpus_questions and args.max_corpus_questions < len(corpus_qs):
        corpus_qs = rnd.sample(corpus_qs, args.max_corpus_questions)

    pool = list(dict.fromkeys(list(train_questions) + corpus_qs + list(curated)))
    pool = [q for q in pool if not is_dev(q)]
    freq: Counter = Counter()
    for q in pool:
        freq.update(content_tokens(q))

    records: dict[str, dict] = {}
    reject: Counter = Counter()
    samples: dict[str, list] = {}
    rule_hits: Counter = Counter()
    tier_pair_counts: Counter = Counter()

    def add_pair(q: str, tgt: str, tier: str, rule: str | None,
                 gold: list[str] | None) -> None:
        curated = tier not in ("qa_train_rule", "corpus_rule")
        reason = validate_pair(q, tgt, freq, curated=curated)
        if reason:
            reject[reason] += 1
            samples.setdefault(reason, [])
            if len(samples[reason]) < 5:
                samples[reason].append({"tier": tier, "src": q, "tgt": tgt})
            return
        rec = records.setdefault(q, {
            "original_question": q, "reformulations": [], "sources": [],
            "transformations": {}, "gold_answer": gold or []})
        if tgt not in rec["reformulations"]:
            rec["reformulations"].append(tgt)
            tier_pair_counts[tier] += 1
        if tgt not in rec["transformations"]:
            rec["transformations"][tgt] = rule or "curated"
        if rule:
            rule_hits[rule] += 1
        if tier not in rec["sources"]:
            rec["sources"].append(tier)
        if gold and not rec["gold_answer"]:
            rec["gold_answer"] = gold

    # tier A: rule variants generated on real train questions
    for q, gold in train_questions.items():
        for tgt, rule in variants_for(q, args.variants_per_train_question):
            add_pair(q, tgt, "qa_train_rule", rule, gold)

    # tier B/C: curated pairs through the same validator
    for src, bucket in curated.items():
        for tgt, tier in bucket.items():
            add_pair(src, tgt, tier, None, curated_gold.get(src, []))

    # tier D: same rules on organic-corpus sources
    for q in corpus_qs:
        if q in train_questions or is_dev(q):
            continue
        for tgt, rule in variants_for(q, args.variants_per_corpus_question):
            add_pair(q, tgt, "corpus_rule", rule, [])

    # tier E: NLP-ctx telugu-qa-codemixed — code-mixed↔native Telugu pairs
    # (dataset used as-is; content is never modified, only selected)
    cm_src = ensure_codemixed_parquet(Path(args.codemixed))
    if cm_src:
        n_cm = add_codemixed_tier(
            records, tier_pair_counts, reject, samples, is_dev, cm_src)
        if n_cm:
            print(f"[codemixed] {n_cm} docs contributed pairs from {cm_src.name}")
    else:
        print("[codemixed] parquet unavailable, tier skipped")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final = [r for r in records.values() if r["reformulations"]]
    rnd.shuffle(final)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in final:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tier_sizes: Counter = Counter()
    for rec in final:
        for tier in rec["sources"]:
            tier_sizes[tier] += 1
    per_q = [len(r["reformulations"]) for r in final]
    report = {
        "out_file": str(out_path),
        "questions": len(final),
        "pairs": sum(per_q),
        "tier_pairs": dict(tier_pair_counts),
        "tier_questions": {k: int(v) for k, v in tier_sizes.items()},
        "rule_hits": dict(Counter({k: v for k, v in rule_hits.items()})),
        "rejections": dict(reject),
        "dev_questions_skipped": skipped_dev,
        "train_questions": len(train_questions),
        "curated_sources": len(curated),
        "corpus_questions_used": len(corpus_qs),
        "pairs_per_question_max": max(per_q) if per_q else 0,
    }
    out_path.with_name(out_path.stem + "_report.json") \
        .write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if samples:
        printed = 0
        print("\n── rejection samples (per reason) ──")
        for reason, xs in samples.items():
            for s in xs[:2]:
                print(f"[{reason}|{s['tier']}] {s['src'][:90]}\n"
                      f"   -> {str(s['tgt'])[:90]}")
                printed += 1
                if printed >= 18:
                    break
            if printed >= 18:
                break

    print("\n── samples: tier A (real train questions) ──")
    shown = 0
    for rec in final:
        if "qa_train_rule" not in rec["sources"] or shown >= 10:
            continue
        shown += 1
        for tgt in rec["reformulations"][:2]:
            print(f"[{rec['transformations'].get(tgt, '?')}] "
                  f"{rec['original_question']}\n   -> {tgt}")
    print("\n── samples: curated tiers ──")
    shown = 0
    for rec in final:
        if not any(s in ("curated_user", "rule_curated", "user_curated")
                   for s in rec["sources"]) or shown >= 8:
            continue
        shown += 1
        for tgt in rec["reformulations"][:2]:
            print(f"[{rec['transformations'].get(tgt, '?')}] "
                  f"{rec['original_question']}\n   -> {tgt}")
    print("\n── samples: tier D (corpus) ──")
    shown = 0
    for rec in final:
        if "corpus_rule" not in rec["sources"] or shown >= 10:
            continue
        shown += 1
        for tgt in rec["reformulations"][:1]:
            print(f"[{rec['transformations'].get(tgt, '?')}] "
                  f"{rec['original_question']}\n   -> {tgt}")

    for cm_tier in ("codemixed_translit", "codemixed_native"):
        print(f"\n── samples: tier E ({cm_tier}) ──")
        shown = 0
        for rec in final:
            if cm_tier not in rec["sources"] or shown >= 10:
                continue
            # show targets actually written by this tier (native targets are
            # Telugu script; translit targets are Latin)
            want_tel = cm_tier == "codemixed_native"
            tgts = [t for t in rec["reformulations"]
                    if bool(TEL_RE.search(t)) == want_tel]
            if not tgts:
                continue
            shown += 1
            tgt = tgts[0]
            print(f"[{rec['transformations'].get(tgt, '?')}] "
                  f"{rec['original_question']}\n   -> {tgt}")


if __name__ == "__main__":
    main()
