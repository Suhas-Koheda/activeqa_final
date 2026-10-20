from __future__ import annotations

import json
import os
import time

import psutil

from .preprocessing import entity_signature
from .rewards import reward, token_f1
from .selection import select_oracle, select_qa_score


def evaluate(rows, reformulator, qa, retriever, cfg, outfile):
    """Evaluate on canonical contexts; report retrieval-only results separately.

    Gold answers are used only for metric calculation and the explicitly offline
    oracle. They never choose contexts or candidates for any deployable method.
    """
    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    started = time.perf_counter()
    predictions = []
    methods = ("original", "pretrained", "qa_selected", "oracle")
    scores = {name: [] for name in methods}
    rewards = {name: [] for name in methods}
    retrieval_scores = {name: [] for name in ("original", "pretrained", "qa_selected", "oracle")}
    valid = preserved = n_candidates = harmful = 0
    qa_seconds = 0.0

    def record_qa_time(elapsed):
        nonlocal qa_seconds
        qa_seconds += elapsed

    def answer_best(question, contexts):
        """Choose the answer/context by model confidence, never by references."""
        nonlocal qa_seconds
        best = (float("-inf"), "", None)
        for context in contexts:
            answer, elapsed, confidence = qa.answer_with_score(question, context["context"])
            record_qa_time(elapsed)
            if confidence > best[0]:
                best = (confidence, answer, context.get("id"))
        return best[1], best[0], best[2]

    for row in rows:
        peak_rss = max(peak_rss, proc.memory_info().rss)
        question = row["question"]
        actions = reformulator.sample_actions(
            question,
            cfg["generation"]["candidates_per_question"],
            cfg["generation"]["max_input_tokens"],
            cfg["generation"]["max_new_tokens"],
            False,
        )
        candidates = [action["candidate"] for action in actions] or [question]
        selection_candidates = list(dict.fromkeys([question, *candidates]))

        # The paragraph attached to this question is its canonical supporting
        # context. All primary QA comparisons use exactly this same context.
        canonical = [{"id": row["id"], "context": row["context"]}]
        retrieved = retriever.retrieve(
            question, cfg["retrieval"]["top_k"], row["id"],
            cfg["retrieval"]["max_context_chars"],
        )

        original_answer, original_confidence, _ = answer_best(question, canonical)
        pretrained_question = candidates[0]
        pretrained_answer, pretrained_confidence, _ = answer_best(pretrained_question, canonical)
        qa_selected_question = select_qa_score(selection_candidates, qa, canonical, record_qa_time)
        qa_selected_answer, qa_selected_confidence, _ = answer_best(qa_selected_question, canonical)
        oracle_question = select_oracle(selection_candidates, qa, canonical, row["answers"], record_qa_time)
        oracle_answer, _, _ = answer_best(oracle_question, canonical)

        chosen = {
            "original": (original_answer, True, True),
            "pretrained": (pretrained_answer, actions[0]["valid"] if actions else True,
                           entity_signature(question).issubset(entity_signature(actions[0]["sampled_text"])) if actions else True),
            "qa_selected": (qa_selected_answer,
                            next((a["valid"] for a in actions if a["candidate"] == qa_selected_question), True),
                            entity_signature(question).issubset(entity_signature(qa_selected_question))),
            "oracle": (oracle_answer,
                       next((a["valid"] for a in actions if a["candidate"] == oracle_question), True),
                       entity_signature(question).issubset(entity_signature(oracle_question))),
        }
        for name, (answer, is_valid, preserves_entities) in chosen.items():
            scores[name].append(token_f1(answer, row["answers"]))
            rewards[name].append(reward(answer, row["answers"], valid=is_valid,
                                        preserve=float(preserves_entities)))

        for action in actions:
            n_candidates += 1
            valid += int(action["valid"])
            preserves = entity_signature(question).issubset(entity_signature(action["sampled_text"]))
            preserved += int(preserves)
        harmful += int(token_f1(pretrained_answer, row["answers"]) <
                        token_f1(original_answer, row["answers"]))

        # Retrieval metrics are a separate diagnostic. If no corpus passages are
        # found, report an empty answer rather than quietly substituting gold context.
        retrieval_record = {}
        if retrieved:
            retrieval_original, retrieval_original_confidence, retrieval_original_id = answer_best(question, retrieved)
            retrieval_pretrained, _, retrieval_pretrained_id = answer_best(pretrained_question, retrieved)
            retrieval_question = select_qa_score(selection_candidates, qa, retrieved, record_qa_time)
            retrieval_selected, _, retrieval_selected_id = answer_best(retrieval_question, retrieved)
            retrieval_oracle_question = select_oracle(selection_candidates, qa, retrieved, row["answers"], record_qa_time)
            retrieval_oracle, _, retrieval_oracle_id = answer_best(retrieval_oracle_question, retrieved)
            retrieved_answers = {
                "original": retrieval_original,
                "pretrained": retrieval_pretrained,
                "qa_selected": retrieval_selected,
                "oracle": retrieval_oracle,
            }
            for name, answer in retrieved_answers.items():
                retrieval_scores[name].append(token_f1(answer, row["answers"]))
            retrieval_record = {
                "original_answer": retrieval_original,
                "original_context_id": retrieval_original_id,
                "pretrained_answer": retrieval_pretrained,
                "pretrained_context_id": retrieval_pretrained_id,
                "qa_selected_question": retrieval_question,
                "qa_selected_answer": retrieval_selected,
                "qa_selected_context_id": retrieval_selected_id,
                "oracle_answer": retrieval_oracle,
                "oracle_context_id": retrieval_oracle_id,
            }

        predictions.append({
            "id": row["id"],
            "question": question,
            "references": row["answers"],
            "canonical_context_id": row["id"],
            "sampled_candidates": [action["sampled_text"] for action in actions],
            "candidates": candidates,
            "candidate_validity": [action["valid"] for action in actions],
            "original_answer": original_answer,
            "original_confidence": original_confidence,
            "pretrained_answer": pretrained_answer,
            "pretrained_confidence": pretrained_confidence,
            "qa_selected_question": qa_selected_question,
            "qa_selected_answer": qa_selected_answer,
            "qa_selected_confidence": qa_selected_confidence,
            "oracle_question": oracle_question,
            "oracle_answer": oracle_answer,
            "retrieval": retrieval_record,
        })
        peak_rss = max(peak_rss, proc.memory_info().rss)

    metric_block = {
        "examples": len(rows),
        "context_protocol": "primary metrics use each example's canonical supporting context",
        "metrics": {
            name: {
                "exact_match": sum(score == 1.0 for score in values) / max(1, len(values)),
                "f1": sum(values) / max(1, len(values)),
                "mean_reward": sum(rewards[name]) / max(1, len(rewards[name])),
            }
            for name, values in scores.items()
        },
        "retrieval_metrics": {
            "context_protocol": "BM25 passages only; answer/context choice uses model confidence",
            "metrics": {
                name: {
                    "examples": len(values),
                    "exact_match": sum(score == 1.0 for score in values) / max(1, len(values)),
                    "f1": sum(values) / max(1, len(values)),
                }
                for name, values in retrieval_scores.items()
            },
        },
        "valid_generation_rate": valid / max(1, n_candidates),
        "entity_signature_preservation_rate": preserved / max(1, n_candidates),
        "harmful_reformulation_rate": harmful / max(1, len(rows)),
        "qa_seconds": round(qa_seconds, 2),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "peak_rss_mb": round(peak_rss / 1e6, 1),
    }
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as stream:
        json.dump({"metrics": metric_block, "predictions": predictions}, stream,
                  ensure_ascii=False, indent=2)
    return metric_block
