"""Calibrate the reference-free candidate ranker from collected pools.

Input:  ``outputs/selector_dataset.jsonl`` written by ``scripts/train.py`` with
        ``training.collect_selector_data: true`` (one record per question with
        per-candidate answers, confidences, validity and gold F1).
Output: ``models/selector_weights.json`` — logistic weights over the
        reference-free features in :mod:`activeqa.ranker` plus a calibrated
        switch threshold, ready for ``evaluation.selection: ranker``.

The gold labels exist only here, on the train split. The shipped artifact is
purely reference-free, so it stays deployable at evaluation time.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "src")

from activeqa.ranker import (FEATURE_NAMES, candidate_features,
                             feature_vector, fill_answer_features)


def build_points(records):
    """One training point per *valid* candidate; grouped by question id.

    Returns (X, y, points_by_question) where points_by_question keeps, for each
    question, the candidate features/f1s plus the original F1 for simulation.
    """
    X, y = [], []
    points_by_question = {}
    for record in records:
        valid_idx = [i for i, v in enumerate(record["validity"]) if v]
        if not valid_idx:
            continue
        pool_answers = [record["answers"][i] for i in valid_idx]
        pool_confs = [record["confidences"][i] for i in valid_idx]
        entry = {
            "question": record["question"],
            "original_f1": record["original_f1"],
            "vectors": [],
            "f1s": [],
        }
        for i in valid_idx:
            features = candidate_features(
                original_question=record["question"],
                original_conf=record["original_confidence"],
                candidate=record["candidates"][i],
                answer=record["answers"][i],
                confidence=record["confidences"][i],
                pool_answers=pool_answers,
                pool_confs=pool_confs,
                valid=True)
            fill_answer_features(features, record["answers"][i],
                                 record["original_answer"], pool_answers)
            vector = feature_vector(features)
            label = int(record["f1s"][i] > record["original_f1"] + 1e-9)
            X.append(vector)
            y.append(label)
            entry["vectors"].append(vector)
            entry["f1s"].append(record["f1s"][i])
        if entry["vectors"]:
            points_by_question[record["id"]] = entry
    return X, y, points_by_question


def fit_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=200)
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/selector_dataset.jsonl")
    parser.add_argument("--out", default="models/selector_weights.json")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    records = []
    with open(args.dataset, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))

    X, y, questions = build_points(records)
    if len(questions) < args.splits * 4 or len(set(y)) < 2:
        raise SystemExit("not enough usable data to calibrate the ranker")

    positive_rate = sum(y) / len(y)
    print(f"records={len(records)} questions={len(questions)} "
          f"points={len(X)} positive_rate={positive_rate:.3f}", flush=True)

    # Question-grouped cross-validation: fit on whole questions, simulate the
    # "switch only above threshold" policy on whole held-out questions.
    question_ids = sorted(questions)
    fold_of = {q: k % args.splits for k, q in enumerate(question_ids)}
    thresholds = [round(t * 0.05, 2) for t in range(2, 19)]
    totals = {t: {"delta": 0.0, "questions": 0, "switches": 0} for t in thresholds}
    for split in range(args.splits):
        train_qs = [q for q in question_ids if fold_of[q] != split]
        test_qs = [q for q in question_ids if fold_of[q] == split]
        train_X = [v for q in train_qs for v in questions[q]["vectors"]]
        train_y = [label for q in train_qs for label in
                   [int(f > questions[q]["original_f1"] + 1e-9) for f in questions[q]["f1s"]]]
        if len({label for label in train_y}) < 2:
            continue
        model = fit_logistic(train_X, train_y)
        for q in test_qs:
            entry = questions[q]
            probs = model.predict_proba(entry["vectors"])[:, 1]
            if len(probs) == 0:
                continue
            best = max(range(len(probs)), key=lambda k: probs[k])
            for t in thresholds:
                slot = totals[t]
                slot["questions"] += 1
                if best is None:
                    continue
                if probs[best] >= t:
                    slot["switches"] += 1
                    slot["delta"] += entry["f1s"][best] - entry["original_f1"]

    table = []
    for t in thresholds:
        slot = totals[t]
        table.append({
            "threshold": t,
            "mean_f1_delta_per_question": slot["delta"] / max(1, slot["questions"]),
            "switch_rate": slot["switches"] / max(1, slot["questions"]),
        })
        print(f"threshold={t:.2f} switch_rate={table[-1]['switch_rate']:.3f} "
              f"mean_f1_delta={table[-1]['mean_f1_delta_per_question']:+.4f}", flush=True)

    best = max(table, key=lambda r: r["mean_f1_delta_per_question"])
    print(f"chosen_threshold={best['threshold']} "
          f"switch_rate={best['switch_rate']:.3f} "
          f"mean_f1_delta={best['mean_f1_delta_per_question']:+.4f}", flush=True)

    final = fit_logistic(X, y)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    artifact = {
        "names": list(FEATURE_NAMES),
        "coef": [float(v) for v in final.coef_[0]],
        "intercept": float(final.intercept_[0]),
        "switch_threshold": best["threshold"],
        "meta": {
            "records": len(records),
            "questions": len(questions),
            "points": len(X),
            "positive_rate": positive_rate,
            "cv_switch_rate": best["switch_rate"],
            "cv_mean_f1_delta": best["mean_f1_delta_per_question"],
            "threshold_table": table,
        },
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2)
    print(f"weights_written path={args.out}", flush=True)


if __name__ == "__main__":
    main()
