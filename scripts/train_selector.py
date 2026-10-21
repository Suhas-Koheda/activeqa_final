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
        entry = points_by_question.setdefault(record["id"], {
            "question": record["question"],
            "original_f1": record["original_f1"],
            "pools": [],
        })
        pool = {"vectors": [], "f1s": []}
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
            pool["vectors"].append(vector)
            pool["f1s"].append(record["f1s"][i])
        if pool["vectors"]:
            # Keep each sampled pool intact. Treating candidates from different
            # epochs as one pool would let offline CV select an action that was
            # never available alongside the other actions it is compared with.
            entry["pools"].append(pool)
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
    if len(questions) < 8:
        raise SystemExit(f"not enough usable data to calibrate the ranker: "
                         f"need at least 8 distinct questions, got {len(questions)}")

    splits = min(args.splits, max(2, len(questions) // 4))

    positive_rate = sum(y) / len(y)
    print(f"records={len(records)} questions={len(questions)} "
          f"points={len(X)} positive_rate={positive_rate:.3f}", flush=True)

    # Question-grouped cross-validation: fit on whole questions, simulate the
    # "switch only above threshold" policy on whole held-out questions.
    question_ids = sorted(questions)
    fold_of = {q: k % splits for k, q in enumerate(question_ids)}
    # 1.01 explicitly represents the always-abstain policy. If every learned
    # switch hurts on cross-validation, calibration can choose the safe policy.
    thresholds = [round(t * 0.05, 2) for t in range(2, 19)] + [1.01]
    totals = {t: {"delta": 0.0, "questions": 0, "switches": 0} for t in thresholds}
    for split in range(splits):
        train_qs = [q for q in question_ids if fold_of[q] != split]
        test_qs = [q for q in question_ids if fold_of[q] == split]
        train_X = [v for q in train_qs for pool in questions[q]["pools"]
                   for v in pool["vectors"]]
        train_y = [int(f > questions[q]["original_f1"] + 1e-9)
                   for q in train_qs for pool in questions[q]["pools"]
                   for f in pool["f1s"]]
        if len({label for label in train_y}) < 2:
            continue
        model = fit_logistic(train_X, train_y)
        for q in test_qs:
            entry = questions[q]
            for pool in entry["pools"]:
                probs = model.predict_proba(pool["vectors"])[:, 1]
                if len(probs) == 0:
                    continue
                best = max(range(len(probs)), key=lambda k: probs[k])
                for t in thresholds:
                    slot = totals[t]
                    slot["questions"] += 1
                    if probs[best] >= t:
                        slot["switches"] += 1
                        slot["delta"] += pool["f1s"][best] - entry["original_f1"]

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

    # On ties prefer the more conservative switch threshold, including the
    # explicit no-switch option. This prevents an all-zero CV table from
    # accidentally choosing a permissive threshold just because it appeared first.
    best = max(table, key=lambda r: (r["mean_f1_delta_per_question"], r["threshold"]))
    print(f"chosen_threshold={best['threshold']} "
          f"switch_rate={best['switch_rate']:.3f} "
          f"mean_f1_delta={best['mean_f1_delta_per_question']:+.4f}", flush=True)

    if len(set(y)) < 2:
        # A single observed label cannot train a classifier. Emit a valid,
        # reference-free artifact that always falls back to the original.
        coef = [0.0] * len(FEATURE_NAMES)
        intercept = 0.0
        best = {"threshold": 1.01, "switch_rate": 0.0,
                "mean_f1_delta_per_question": 0.0}
    else:
        final = fit_logistic(X, y)
        coef = [float(v) for v in final.coef_[0]]
        intercept = float(final.intercept_[0])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    artifact = {
        "names": list(FEATURE_NAMES),
        "coef": coef,
        "intercept": intercept,
        "switch_threshold": best["threshold"],
        "meta": {
            "records": len(records),
            "questions": len(questions),
            "cross_validation_splits": splits,
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
