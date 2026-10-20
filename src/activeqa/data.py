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

def load_splits(cfg, include_full_train=False):
    """Load only Telugu IndicQA test records; deterministically split for exploration.

    The upstream repository provides a legacy custom loading script, which Datasets 5.x
    intentionally no longer executes. Fetch only the requested language JSON and parse
    the documented SQuAD-style structure directly.
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
    # Stable shuffle, disjoint subsets. Since the source has no train split, this is exploratory
    # policy training on part of the benchmark test set; it is not an official test evaluation.
    import random
    rng=random.Random(42); rng.shuffle(rows)
    n_eval=max(1, int(len(rows)*cfg.get("eval_fraction", .2)))
    ev=rows[:n_eval]; full_train=rows[n_eval:]; tr=full_train
    if cfg.get("train_examples") is not None: tr=tr[:int(cfg["train_examples"])]
    if cfg.get("eval_examples") is not None: ev=ev[:int(cfg["eval_examples"])]
    if include_full_train:
        return tr,ev,full_train
    return tr,ev

def save_splits(train, evaluation, out="outputs/split_manifest.json"):
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps({"seed":42,"train_ids":[x["id"] for x in train],"eval_ids":[x["id"] for x in evaluation],"train_size":len(train),"eval_size":len(evaluation)},ensure_ascii=False,indent=2),encoding="utf-8")
