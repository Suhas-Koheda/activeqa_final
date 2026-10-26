"""Paired held-out QA evaluation for a reformulation policy.

The SFT script measures paraphrase quality (token-F1 against reference
rewrites). This module measures the thing that actually matters: does the
deployment pipeline answer a held-out question better after reformulation?

Every question is answered twice on its own passage:

* the original question (the do-nothing baseline), and
* the question chosen by the deployed confidence selector among the original
  plus the policy's sampled rewrites.

Selection is reference-free (span confidence + answer consistency); gold
answers are used only to compute token-F1 afterwards. ``generation_oracle_f1``
reports the best rewrite the policy produced, which separates a weak policy
from a weak selector: if the oracle rises but the selected score does not, the
selector is the bottleneck.
"""
from __future__ import annotations

import random
import re
import time

from .preprocessing import normalize
from .rewards import token_f1

TELUGU = re.compile(r"[\u0C00-\u0C7F]")
TAMIL = re.compile(r"[\u0B80-\u0BFF]")
KANNADA = re.compile(r"[\u0C80-\u0CFF]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")

CONSISTENCY_FLOOR = 0.5


def _script_counts(text):
    return {"te": len(TELUGU.findall(text)), "ta": len(TAMIL.findall(text)),
            "kn": len(KANNADA.findall(text)),
            "deva": len(DEVANAGARI.findall(text))}


def _bootstrap_ci(deltas, samples=1000):
    """Paired bootstrap interval for the mean per-question F1 delta."""
    if len(deltas) <= 1:
        value = round(sum(deltas) / max(1, len(deltas)), 4)
        return [value, value]
    rng = random.Random(0)
    boots = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
        for _ in range(samples))
    return [round(boots[int(0.025 * samples)], 4),
            round(boots[int(0.975 * samples) - 1], 4)]


def evaluate_qa_pairs(ref, qa, rows, *, candidates=4, limit=None,
                      max_input=96, max_new=64, do_sample=True,
                      confidence_override=0.5,
                      log=None, label="policy"):
    """Return (summary, records) for held-out QA with/without reformulation.

    ``selected_f1`` is the currently deployed rule: only rewrites whose answer
    is consistent with the original's answer may win on span confidence.
    ``corrective_f1`` additionally lets the single highest-confidence rewrite
    replace the original when it beats the original's confidence by
    ``confidence_override``. That is the case reformulation exists for (the
    original question produced a wrong span), but it is also the riskier case,
    so both numbers are reported instead of silently changing deployment.
    """
    rows = list(rows[:limit] if limit else rows)
    if not rows:
        raise ValueError("QA evaluation needs at least one row")
    if log:
        log(f"      QA eval [{label}]: {len(rows)} questions × "
            f"{candidates} candidates (confidence selection) …")
    started = time.perf_counter()
    records = []
    for index, row in enumerate(rows):
        question = normalize(row["question"])
        actions = ref.sample_actions(
            question, candidates, max_input, max_new, do_sample)
        texts = [question] + [action["candidate"] for action in actions]
        answers = qa.answer_batch(texts, row["context"])
        original_answer, original_confidence = answers[0][0], answers[0][2]
        original_f1 = token_f1(original_answer, row["answers"])

        candidate_f1s = []
        for action, (answer, _, confidence) in zip(actions, answers[1:]):
            candidate_f1s.append({
                "answer": answer,
                "confidence": confidence,
                "f1": token_f1(answer, row["answers"]),
                "valid": action["valid"],
                "consistent": token_f1(answer, [original_answer]) >= CONSISTENCY_FLOOR,
                "text": action["sampled_text"],
            })

        # Same rule as activeqa.selection.select_qa_score: the original is
        # always eligible; a rewrite must be valid and keep an answer that is
        # consistent with the original question's answer. Highest span
        # confidence wins, so this matches deployment.
        eligible = [0] + [i + 1 for i, cand in enumerate(candidate_f1s)
                          if cand["valid"] and cand["consistent"]]
        best = max(eligible, key=lambda i: answers[i][2])
        selected_question = texts[best]
        selected_answer, selected_confidence = answers[best][0], answers[best][2]
        selected_f1 = token_f1(selected_answer, row["answers"])

        valid_indices = [i + 1 for i, cand in enumerate(candidate_f1s)
                         if cand["valid"]]
        top = max(valid_indices, key=lambda i: answers[i][2], default=None)
        corrective = top if (top is not None
                             and not candidate_f1s[top - 1]["consistent"]
                             and answers[top][2]
                             >= original_confidence + confidence_override) else None
        corrective_index = corrective if corrective is not None else best
        corrective_question = texts[corrective_index]
        corrective_f1 = token_f1(answers[corrective_index][0], row["answers"])

        valid_f1s = [cand["f1"] for cand in candidate_f1s if cand["valid"]]
        oracle_f1 = max(valid_f1s) if valid_f1s else original_f1
        scripts = _script_counts(selected_question)
        records.append({
            "id": row.get("id"),
            "question": question,
            "selected_question": selected_question,
            "selected_index": best,
            "switched": best != 0,
            "references": row["answers"],
            "original_answer": original_answer,
            "selected_answer": selected_answer,
            "original_f1": original_f1,
            "selected_f1": selected_f1,
            "corrective_f1": corrective_f1,
            "corrective_question": corrective_question,
            "corrective_switch": corrective_index != 0,
            "original_confidence": original_confidence,
            "selected_confidence": selected_confidence,
            "candidate_f1s": [cand["f1"] for cand in candidate_f1s],
            "candidate_validity": [cand["valid"] for cand in candidate_f1s],
            "candidate_texts": [cand["text"] for cand in candidate_f1s],
            "generation_oracle_f1": oracle_f1,
            "valid": True if best == 0 else candidate_f1s[best - 1]["valid"],
            "scripts": scripts,
        })
        if log and (index + 1) % 25 == 0:
            done = records
            log(f"      QA eval [{label}] {index + 1}/{len(rows)}  "
                f"orig={sum(r['original_f1'] for r in done) / len(done):.4f}  "
                f"selected={sum(r['selected_f1'] for r in done) / len(done):.4f}")

    n = max(1, len(records))
    candidate_count = sum(len(r["candidate_f1s"]) for r in records)
    valid_count = sum(sum(r["candidate_validity"]) for r in records)
    switched = [r for r in records if r["switched"]]
    summary = {
        "label": label,
        "examples": len(records),
        "candidates_per_question": candidates,
        "original_f1": round(sum(r["original_f1"] for r in records) / n, 4),
        "selected_f1": round(sum(r["selected_f1"] for r in records) / n, 4),
        "corrective_f1": round(sum(r["corrective_f1"] for r in records) / n, 4),
        "generation_oracle_f1": round(
            sum(r["generation_oracle_f1"] for r in records) / n, 4),
        "improved_rate": round(
            sum(r["selected_f1"] > r["original_f1"] + 1e-9 for r in records) / n, 4),
        "harmed_rate": round(
            sum(r["selected_f1"] < r["original_f1"] - 1e-9 for r in records) / n, 4),
        "switch_rate": round(len(switched) / n, 4),
        "corrective_switch_rate": round(
            sum(r["corrective_switch"] for r in records) / n, 4),
        "corrective_harmed_rate": round(
            sum(r["corrective_f1"] < r["original_f1"] - 1e-9 for r in records) / n, 4),
        "confidence_override": confidence_override,
        "mean_confidence_delta": round(
            sum(r["selected_confidence"] - r["original_confidence"]
                for r in records) / n, 4),
        "valid_generation_rate": round(valid_count / max(1, candidate_count), 4),
        "distinct_selected_ratio": round(
            len({r["selected_question"] for r in records}) / n, 4),
        "ends_with_question_mark": round(
            sum(r["selected_question"].strip().endswith("?") for r in records) / n, 4),
        "tamil_spill_outputs": sum(1 for r in records if r["scripts"]["ta"]),
        "kannada_spill_outputs": sum(1 for r in records if r["scripts"]["kn"]),
        "devanagari_spill_outputs": sum(1 for r in records if r["scripts"]["deva"]),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    summary["delta_f1"] = round(summary["selected_f1"] - summary["original_f1"], 4)
    summary["corrective_delta_f1"] = round(
        summary["corrective_f1"] - summary["original_f1"], 4)
    summary["delta_ci95"] = _bootstrap_ci(
        [record["selected_f1"] - record["original_f1"] for record in records])
    summary["corrective_delta_ci95"] = _bootstrap_ci(
        [record["corrective_f1"] - record["original_f1"] for record in records])
    summary["generation_oracle_delta"] = round(
        summary["generation_oracle_f1"] - summary["original_f1"], 4)
    summary["switched_mean_lift"] = round(
        sum(r["selected_f1"] - r["original_f1"] for r in switched)
        / max(1, len(switched)), 4)
    return summary, records
