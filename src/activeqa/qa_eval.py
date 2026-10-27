"""Paired held-out QA evaluation for a reformulation policy.

Reference-paraphrase scoring measures paraphrase quality (token-F1 against
reference rewrites). This module measures the thing that actually matters:
does the deployment pipeline answer a held-out question better after
reformulation?

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
CONSENSUS_FLOOR = 0.5
CONSENSUS_MIN_SUPPORT = 2
CONFIDENCE_MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0)
# Margins for the corrective rule: the highest-confidence valid rewrite may
# replace the original only when it beats the original's span score by this
# much. 0.0 switches on any improvement; large values converge to abstention.
CORRECTIVE_MARGINS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)


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


def _quantile(values, q):
    """Nearest-rank quantile without adding a numpy dependency."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _auc(scores, labels):
    """Rank-based AUC (Mann–Whitney) for ranking positives above negatives.

    This answers a question that must come before any threshold sweep: does the
    signal order recoverable questions above the rest *at all*? An AUC of 0.5
    means no threshold on this signal can work, so a failed sweep is not a
    tuning problem. Uses the statistic directly, without a scipy dependency.
    """
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 4)


def deployable_rule(summary):
    """Best selection rule whose paired delta CI excludes zero, or None.

    Sweeping thresholds on one dev set inflates the maximum, so a rule is only
    deployable when its *lower* bootstrap bound is above zero. Returns
    ``(name, options)``; ``options`` maps every measured rule to
    ``(f1, ci95)`` so the report can show near misses.
    """
    original = summary["original_f1"]
    options = {
        "consistent": (summary["selected_f1"], summary.get("delta_ci95")),
        "corrective": (summary["corrective_f1"],
                       summary.get("corrective_delta_ci95")),
        "consensus": (summary["consensus_f1"],
                      summary.get("consensus_delta_ci95")),
    }
    for margin, row in summary.get("corrective_sweep", {}).items():
        options[f"corrective@{margin}"] = (row["selected_f1"],
                                           row.get("delta_ci95"))
    for key, row in summary.get("selective_sweep", {}).items():
        options[f"selective:{key}"] = (row["selected_f1"],
                                       row.get("delta_ci95"))
    passing = {
        name: value for name, value in options.items()
        if value[0] > original + 1e-9 and value[1] and value[1][0] > 0
    }
    if not passing:
        return None, options
    return max(passing, key=lambda name: passing[name][0]), options


def paired_delta_ci(records_a, records_b, key, samples=1000):
    """Bootstrap CI for mean(a[key] - b[key]) over aligned per-question rows.

    ``summary["delta_ci95"]`` compares one policy against the original
    question. That is not the interval for RL − start, where both policies are
    evaluated on the same rows and their per-question difference must be
    bootstrapped directly.
    """
    if len(records_a) != len(records_b):
        raise ValueError("paired CI needs the same rows for both policies")
    return _bootstrap_ci(
        [float(a[key]) - float(b[key]) for a, b in zip(records_a, records_b)],
        samples=samples)


def _margin_choice(answers, candidate_f1s, original_confidence, margin):
    """Conservative rule with an extra confidence margin over the original."""
    eligible = [0] + [i + 1 for i, cand in enumerate(candidate_f1s)
                      if cand["valid"] and cand["consistent"]
                      and cand["confidence"] >= original_confidence + margin]
    return max(eligible, key=lambda i: answers[i][2])


def _corrective_choice(answers, candidate_f1s, original_confidence, margin):
    """Highest-confidence valid rewrite, only past a confidence margin.

    The conservative rule can never change the answer, so it cannot fix a wrong
    original — the only case reformulation exists for. This rule can, but too
    small a margin makes it switch constantly and lose. Sweeping the margin
    finds out whether any amount of switching pays. Returns a 1-based index
    into ``answers`` (0 is the original).
    """
    valid = [i for i, cand in enumerate(candidate_f1s) if cand["valid"]]
    if not valid:
        return 0
    top = max(valid, key=lambda i: candidate_f1s[i]["confidence"])
    if candidate_f1s[top]["confidence"] >= original_confidence + margin:
        return top + 1
    return 0


def _consensus_choice(answers, candidate_f1s, original_answer,
                      original_confidence):
    """Pick a rewrite answer agreed on by at least two valid rewrites.

    The conservative rule can never correct a wrong original, because it
    requires the rewrite's answer to match the original's. The confidence
    override can switch, but a single confident wrong answer is enough to hurt
    (see ``corrective_f1``). Consensus is the middle ground: switch only when
    several independent rewrites land on the same different answer, and that
    answer is more confident than the original. Returns a 0-based index into
    ``candidate_f1s`` or None.
    """
    valid = [i for i, cand in enumerate(candidate_f1s) if cand["valid"]]
    if len(valid) < CONSENSUS_MIN_SUPPORT:
        return None
    clusters = []
    for index in valid:
        cand = candidate_f1s[index]
        for cluster in clusters:
            if token_f1(cand["answer"],
                        [cluster[0]["answer"]]) >= CONSENSUS_FLOOR:
                cluster.append({**cand, "index": index})
                break
        else:
            clusters.append([{**cand, "index": index}])
    best = None
    for cluster in clusters:
        if len(cluster) < CONSENSUS_MIN_SUPPORT:
            continue
        # Same-answer clusters are already covered by the conservative rule.
        if token_f1(cluster[0]["answer"],
                    [original_answer]) >= CONSISTENCY_FLOOR:
            continue
        leader = max(cluster, key=lambda cand: cand["confidence"])
        if leader["confidence"] <= original_confidence:
            continue
        rank = (leader["confidence"], len(cluster))
        if best is None or rank > best[0]:
            best = (rank, leader["index"])
    return None if best is None else best[1]


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
    original question produced a wrong span), but it is also the riskier case.
    ``consensus_f1`` switches only when at least two valid rewrites agree on a
    different answer that is more confident than the original.
    ``margin_sweep`` reports the conservative rule at several confidence
    margins over the original, so deployment does not have to guess a
    threshold. All are reported; none silently replaces the deployed rule.
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
        best = _margin_choice(answers, candidate_f1s, original_confidence, 0.0)
        selected_question = texts[best]
        selected_answer, selected_confidence = answers[best][0], answers[best][2]
        selected_f1 = token_f1(selected_answer, row["answers"])

        margin_f1, margin_switch = {}, {}
        for margin in CONFIDENCE_MARGINS:
            key = f"{margin:.2f}"
            index = _margin_choice(answers, candidate_f1s,
                                   original_confidence, margin)
            margin_f1[key] = token_f1(answers[index][0], row["answers"])
            margin_switch[key] = index != 0

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

        corrective_margin_f1, corrective_margin_switch = {}, {}
        for margin in CORRECTIVE_MARGINS:
            key = f"{margin:.2f}"
            index = _corrective_choice(answers, candidate_f1s,
                                       original_confidence, margin)
            corrective_margin_f1[key] = token_f1(answers[index][0],
                                                 row["answers"])
            corrective_margin_switch[key] = index != 0

        valid_actions = [i for i, cand in enumerate(candidate_f1s)
                         if cand["valid"]]
        best_rewrite = (max(valid_actions,
                            key=lambda i: candidate_f1s[i]["confidence"])
                        if valid_actions else None)
        best_rewrite_f1 = (candidate_f1s[best_rewrite]["f1"]
                           if best_rewrite is not None else original_f1)

        consensus = _consensus_choice(answers, candidate_f1s, original_answer,
                                      original_confidence)
        # _consensus_choice returns a 0-based candidate index; texts/answers
        # put the original question at 0.
        consensus_index = consensus + 1 if consensus is not None else best
        consensus_question = texts[consensus_index]
        consensus_f1 = token_f1(answers[consensus_index][0], row["answers"])

        valid_f1s = [cand["f1"] for cand in candidate_f1s if cand["valid"]]
        oracle_f1 = max(valid_f1s) if valid_f1s else original_f1
        # Upper bound for any selector over {original, rewrites}.
        ceiling_f1 = max(original_f1, oracle_f1)
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
            "consensus_f1": consensus_f1,
            "consensus_question": consensus_question,
            "consensus_switch": consensus_index != 0,
            "best_rewrite_index": best_rewrite,
            "best_rewrite_f1": best_rewrite_f1,
            "best_rewrite_confidence": (
                candidate_f1s[best_rewrite]["confidence"]
                if best_rewrite is not None else None),
            "confidence_gain": (
                candidate_f1s[best_rewrite]["confidence"]
                - original_confidence if best_rewrite is not None else 0.0),
            "disagreement_rate": (
                sum(1 for cand in candidate_f1s
                    if cand["valid"] and not cand["consistent"])
                / max(1, sum(1 for cand in candidate_f1s if cand["valid"]))),
            "corrective_margin_f1": corrective_margin_f1,
            "corrective_margin_switch": corrective_margin_switch,
            "candidate_f1s": [cand["f1"] for cand in candidate_f1s],
            "candidate_validity": [cand["valid"] for cand in candidate_f1s],
            "candidate_texts": [cand["text"] for cand in candidate_f1s],
            "candidate_answers": [cand["answer"] for cand in candidate_f1s],
            "candidate_confidences": [cand["confidence"] for cand in candidate_f1s],
            "margin_f1": margin_f1,
            "margin_switch": margin_switch,
            "generation_oracle_f1": oracle_f1,
            "ceiling_f1": ceiling_f1,
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
        "consensus_f1": round(sum(r["consensus_f1"] for r in records) / n, 4),
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
        "consensus_switch_rate": round(
            sum(r["consensus_switch"] for r in records) / n, 4),
        "consensus_harmed_rate": round(
            sum(r["consensus_f1"] < r["original_f1"] - 1e-9 for r in records) / n, 4),
        "consensus_improved_rate": round(
            sum(r["consensus_f1"] > r["original_f1"] + 1e-9 for r in records) / n, 4),
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
    summary["consensus_delta_f1"] = round(
        summary["consensus_f1"] - summary["original_f1"], 4)
    summary["delta_ci95"] = _bootstrap_ci(
        [record["selected_f1"] - record["original_f1"] for record in records])
    summary["corrective_delta_ci95"] = _bootstrap_ci(
        [record["corrective_f1"] - record["original_f1"] for record in records])
    summary["consensus_delta_ci95"] = _bootstrap_ci(
        [record["consensus_f1"] - record["original_f1"] for record in records])
    summary["generation_oracle_delta"] = round(
        summary["generation_oracle_f1"] - summary["original_f1"], 4)

    margin_sweep = {}
    for margin in CONFIDENCE_MARGINS:
        key = f"{margin:.2f}"
        values = [record["margin_f1"][key] for record in records]
        margin_sweep[key] = {
            "selected_f1": round(sum(values) / n, 4),
            "delta_f1": round(sum(values) / n - summary["original_f1"], 4),
            "switch_rate": round(
                sum(record["margin_switch"][key] for record in records) / n, 4),
            "improved_rate": round(
                sum(value > record["original_f1"] + 1e-9
                    for value, record in zip(values, records)) / n, 4),
            "harmed_rate": round(
                sum(value < record["original_f1"] - 1e-9
                    for value, record in zip(values, records)) / n, 4),
        }
    summary["margin_sweep"] = margin_sweep
    summary["best_margin"] = max(
        margin_sweep, key=lambda key: margin_sweep[key]["selected_f1"])

    # Corrective sweep. The conservative sweep above can never change the
    # answer, so by construction it cannot fix a wrong original. This one can:
    # it is the only rule with headroom toward ceiling_f1, and the question is
    # whether some margin captures the recoverable questions without dragging
    # in the harmless-but-wrong switches.
    corrective_sweep = {}
    for margin in CORRECTIVE_MARGINS:
        key = f"{margin:.2f}"
        values = [record["corrective_margin_f1"][key] for record in records]
        corrective_sweep[key] = {
            "selected_f1": round(sum(values) / n, 4),
            "delta_f1": round(sum(values) / n - summary["original_f1"], 4),
            "switch_rate": round(
                sum(record["corrective_margin_switch"][key]
                    for record in records) / n, 4),
            "improved_rate": round(
                sum(value > record["original_f1"] + 1e-9
                    for value, record in zip(values, records)) / n, 4),
            "harmed_rate": round(
                sum(value < record["original_f1"] - 1e-9
                    for value, record in zip(values, records)) / n, 4),
            "delta_ci95": _bootstrap_ci(
                [record["corrective_margin_f1"][key] - record["original_f1"]
                 for record in records]),
        }
    summary["corrective_sweep"] = corrective_sweep
    summary["best_corrective_margin"] = max(
        corrective_sweep,
        key=lambda key: corrective_sweep[key]["selected_f1"])

    # Gate × corrective margin: reformulate only uncertain originals *and* only
    # when the best rewrite is clearly more confident. This is the deployable
    # form of the ceiling: abstain by default, switch rarely.
    confidences_for_gate = [record["original_confidence"] for record in records]
    selective_sweep = {}
    for quantile in (0.25, 0.5, 0.75):
        threshold = _quantile(confidences_for_gate, quantile)
        for margin in CORRECTIVE_MARGINS:
            margin_key = f"{margin:.2f}"
            values, switches = [], 0
            for record in records:
                if (record["original_confidence"] <= threshold
                        and record["corrective_margin_switch"][margin_key]):
                    values.append(record["corrective_margin_f1"][margin_key])
                    switches += 1
                else:
                    values.append(record["original_f1"])
            selective_sweep[f"q{quantile:.2f}/m{margin_key}"] = {
                "selected_f1": round(sum(values) / n, 4),
                "delta_f1": round(sum(values) / n - summary["original_f1"], 4),
                "switch_rate": round(switches / n, 4),
                "delta_ci95": _bootstrap_ci(
                    [value - record["original_f1"]
                     for value, record in zip(values, records)]),
            }
    summary["selective_sweep"] = selective_sweep
    summary["best_selective"] = max(
        selective_sweep,
        key=lambda key: selective_sweep[key]["selected_f1"])

    # ── selective reformulation ─────────────────────────────────────────────
    # Can a deployable rule (gate on the original's own span confidence) find
    # the questions where a rewrite actually beats the original? This is the
    # only route left when the policy's candidate oracle is below the original:
    # abstain everywhere except the uncertain originals.
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    recoverable = [record for record in records
                   if record["ceiling_f1"] > record["original_f1"] + 1e-9]
    not_recoverable = [record for record in records
                       if record["ceiling_f1"] <= record["original_f1"] + 1e-9]
    summary["ceiling_f1"] = round(
        sum(record["ceiling_f1"] for record in records) / n, 4)
    summary["ceiling_delta_f1"] = round(
        summary["ceiling_f1"] - summary["original_f1"], 4)
    summary["recoverable_rate"] = round(len(recoverable) / n, 4)
    summary["recoverable_mean_gain"] = round(_mean(
        [record["ceiling_f1"] - record["original_f1"]
         for record in recoverable]), 4)
    summary["recoverable_original_confidence"] = round(_mean(
        [record["original_confidence"] for record in recoverable]), 4)
    summary["nonrecoverable_original_confidence"] = round(_mean(
        [record["original_confidence"] for record in not_recoverable]), 4)

    # Does any *deployable* signal rank the recoverable questions above the
    # rest? This is the precondition for every threshold rule below; AUC 0.5
    # means a failed sweep is not a tuning problem.
    labels = [record["ceiling_f1"] > record["original_f1"] + 1e-9
              for record in records]
    summary["signal_auc"] = {
        "low_original_confidence": _auc(
            [-record["original_confidence"] for record in records], labels),
        "confidence_gain": _auc(
            [record["confidence_gain"] for record in records], labels),
        "rewrite_disagreement": _auc(
            [record["disagreement_rate"] for record in records], labels),
    }

    confidences = [record["original_confidence"] for record in records]
    gate_sweep = {}
    for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        threshold = _quantile(confidences, quantile)
        gated = [record for record in records
                 if record["original_confidence"] <= threshold]
        # Inside the gate, use the most confident valid rewrite. The
        # conservative rule cannot help here by construction: a wrong original
        # requires changing the answer, which consistency forbids.
        values = [(record["best_rewrite_f1"]
                   if record["original_confidence"] <= threshold
                   else record["original_f1"]) for record in records]
        gated_recoverable = [record for record in gated
                             if record["ceiling_f1"]
                             > record["original_f1"] + 1e-9]
        gate_sweep[f"{quantile:.2f}"] = {
            "threshold": round(threshold, 4),
            "examples": len(gated),
            "gate_rate": round(len(gated) / n, 4),
            "selected_f1": round(sum(values) / n, 4),
            "delta_f1": round(sum(values) / n - summary["original_f1"], 4),
            "gated_original_f1": round(_mean(
                [record["original_f1"] for record in gated]), 4),
            "gated_selected_f1": round(_mean(
                [record["best_rewrite_f1"] for record in gated]), 4),
            # Is the headroom actually inside this gate? If the gated ceiling
            # is not above the gated original, no selector can help here.
            "gated_ceiling_f1": round(_mean(
                [record["ceiling_f1"] for record in gated]), 4),
            "gated_recoverable_rate": round(
                len(gated_recoverable) / max(1, len(gated)), 4),
            "switch_rate": round(sum(
                record["best_rewrite_index"] is not None
                for record in gated) / max(1, len(gated)), 4),
            "harmed_rate": round(sum(
                value < record["original_f1"] - 1e-9
                for value, record in zip(values, records)) / n, 4),
        }
    summary["gate_sweep"] = gate_sweep
    summary["best_gate"] = max(
        gate_sweep, key=lambda key: gate_sweep[key]["selected_f1"])
    summary["switched_mean_lift"] = round(
        sum(r["selected_f1"] - r["original_f1"] for r in switched)
        / max(1, len(switched)), 4)
    return summary, records
