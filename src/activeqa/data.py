from __future__ import annotations
import hashlib, json
from pathlib import Path
from .preprocessing import normalize

def _row(r):
    a=r.get("answers") or {}; texts=a.get("text", []) if isinstance(a,dict) else []
    return {"id":str(r.get("id", "")), "question":normalize(r.get("question")), "context":normalize(r.get("context")), "answers":[normalize(x) for x in texts if normalize(x)]}

def _load_indicqa_json(path):
    """Flatten IndicQA's published SQuAD-style JSON structure."""
    with open(path, encoding="utf-8") as f:
        raw=json.load(f)
    rows=[]
    for article in raw.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context=normalize(paragraph.get("context"))
            for qa in paragraph.get("qas", []):
                rows.append(_row({"id":qa.get("id"), "context":context,
                                  "question":qa.get("question"),
                                  "answers":{"text":[a.get("text", "") for a in qa.get("answers", [])]}}))
    return rows

def _load_jsonl_qa(path):
    """Load the project JSONL schema: id, question, context, answers."""
    rows=[]; seen=set()
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            item=json.loads(line)
            row={"id":str(item.get("id", "")), "question":normalize(item.get("question")),
                 "context":normalize(item.get("context")),
                 "answers":[normalize(x) for x in (item.get("answers") or []) if normalize(x)],
                 "source":str(item.get("source", "external jsonl")),
                 "wiki_id":str(item.get("wiki_id", ""))}
            if not row["question"] or not row["context"] or not row["answers"]: continue
            key=hashlib.sha256((row["question"]+"\0"+row["context"]+"\0"+"|".join(row["answers"])).encode()).hexdigest()
            if key in seen: continue
            seen.add(key); rows.append(row)
    return rows

def _looks_like_hf_dataset(source):
    text = str(source)
    if text.endswith(".jsonl") or text.startswith((".", "/", "~")):
        return False
    parts = text.split("/")
    return len(parts) == 2 and all(parts)


def _resolve_external_jsonl(source, split, cache_dir=None):
    """Resolve a local JSONL path or a Hugging Face dataset repo to a local file.

    Accepts ``data/indic_rag_te/train.jsonl`` or a dataset repo id such as
    ``nlpctx/activeqa-telugu-rl-data``. For a repo id, the requested split file
    (``train.jsonl`` or ``dev.jsonl``) is downloaded from the Hub.
    """
    path = Path(str(source)).expanduser()
    if path.is_file():
        return str(path)
    if not _looks_like_hf_dataset(source):
        raise FileNotFoundError(f"External QA source not found: {source}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=str(source), filename=f"{split}.jsonl",
                           repo_type="dataset", cache_dir=cache_dir)


def load_splits(cfg, include_full_train=False):
    """Load external Telugu training rows and a disjoint IndicQA test evaluation.

    The upstream repository provides a legacy custom loading script, which Datasets 5.x
    intentionally no longer executes. Fetch only the requested language JSON and parse
    the documented SQuAD-style structure directly. Without ``external_train_file``, retain
    the old exploratory split of IndicQA test rows for backwards compatibility.
    """
    from huggingface_hub import hf_hub_download
    dataset=cfg.get("dataset", "ai4bharat/IndicQA")
    language=cfg.get("config", "indicqa.te").split(".")[-1]
    if language != "te":
        raise ValueError(f"This implementation currently expects Telugu config indicqa.te, got {language!r}")
    cache_dir=cfg.get("cache_dir")
    filepath=hf_hub_download(repo_id=dataset, filename=f"data/indicqa.{language}.json",
                             repo_type="dataset", cache_dir=cache_dir)
    rows=[]; seen=set()
    for x in _load_indicqa_json(filepath):
        if not x["question"] or not x["context"] or not x["answers"]: continue
        key=hashlib.sha256((x["question"]+"\0"+x["context"]+"\0"+"|".join(x["answers"])).encode()).hexdigest()
        if key in seen: continue
        seen.add(key); rows.append(x)
    # Stable shuffle for reproducible held-out evaluation selection.
    import random
    rng=random.Random(42); rng.shuffle(rows)
    external_train=cfg.get("external_train_dataset") or cfg.get("external_train_file")
    if external_train:
        # Real training comes from the generated dataset: the published Hugging Face
        # repo by default, with the committed JSONL as an offline fallback. Keep all
        # IndicQA rows held out unless a smaller evaluation subset is configured, and
        # remove any exact context duplicates across the two sources.
        try:
            train_source=_resolve_external_jsonl(external_train, "train", cache_dir)
        except Exception as exc:
            fallback=cfg.get("external_train_file")
            if not fallback or fallback==external_train:
                raise
            print(f"external_dataset_unavailable={external_train} "
                  f"({type(exc).__name__}: {exc}); falling back to {fallback}")
            train_source=_resolve_external_jsonl(fallback, "train", cache_dir)
        print(f"external_train_source={train_source}")
        full_train=_load_jsonl_qa(train_source)
        ev=rows
        if cfg.get("eval_examples") is not None: ev=ev[:int(cfg["eval_examples"])]
        heldout_contexts={r["context"].casefold() for r in ev}
        before=len(full_train)
        full_train=[r for r in full_train if r["context"].casefold() not in heldout_contexts]
        print(f"external_train_rows={before}; dropped_exact_eval_context_duplicates={before-len(full_train)}")
        tr=full_train
        if cfg.get("train_examples") is not None: tr=tr[:int(cfg["train_examples"])]
        if include_full_train: return tr,ev,full_train
        return tr,ev
    # Legacy exploratory mode: the upstream source has no train split, so training
    # is drawn only from the disjoint remainder of its test rows.
    n_eval=max(1, int(len(rows)*cfg.get("eval_fraction", .2)))
    ev=rows[:n_eval]; full_train=rows[n_eval:]; tr=full_train
    if cfg.get("train_examples") is not None: tr=tr[:int(cfg["train_examples"])]
    if cfg.get("eval_examples") is not None: ev=ev[:int(cfg["eval_examples"])]
    if include_full_train:
        return tr,ev,full_train
    return tr,ev

def save_splits(train, evaluation, out="outputs/split_manifest.json"):
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps({"seed":42,"train_ids":[x["id"] for x in train],"eval_ids":[x["id"] for x in evaluation],"train_size":len(train),"eval_size":len(evaluation),"train_source":train[0].get("source","IndicQA exploratory") if train else None,"evaluation_source":"ai4bharat/IndicQA test"},ensure_ascii=False,indent=2),encoding="utf-8")
