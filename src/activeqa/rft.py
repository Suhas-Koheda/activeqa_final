"""On-policy rejection-sampling fine-tuning (RFT) for Telugu reformulation.

The SFT adapter is a good paraphraser, but it was trained to imitate reference
paraphrases, not to make a downstream QA model answer correctly. This stage
closes that gap with the frozen Telugu QA model as the reward:

1. Sample ``candidates`` rewrites per question from the current policy.
2. Answer the original question and every rewrite on the same passage.
3. Keep rewrites that are valid, score at least as well as the original
   (``margin``), and whose answer span is at least as confident as the
   original's (``confidence_slack``, default 0.0 because the deployed selector
   only switches on ``confidence > original``). Training on rewrites the
   selector would never pick does not change held-out QA.
4. Teacher-force the best qualifying rewrite, accumulating until
   ``batch_questions`` targets, then take one optimizer step.

Unlike REINFORCE, a batch with no qualifying rewrite is skipped rather than
pushing every sampled action down, which is what made the earlier all-negative
REINFORCE runs flat or harmful.
"""
from __future__ import annotations

import math
import random
import time

import torch

from .preprocessing import entity_signature
from .policy_gradient import sft_batch_step
from .rewards import reward, token_f1


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def rft_on_policy(ref, qa, rows, *, candidates=6, batch_questions=8,
                  epochs=1, learning_rate=7e-5, margin=0.0,
                  confidence_slack=0.0, max_input=96, max_new=64,
                  do_sample=True, seed=42, generate_batch=8,
                  max_updates=None, after_update=None, skip_solved=True,
                  log=print):
    """Run RFT on ``rows`` (dicts with question/context/answers).

    ``skip_solved``: questions whose original answer is already correct
    (token-F1 ≥ 0.99) are skipped — there is no headroom to learn from and
    training on them only teaches safe paraphrasing of solved cases.
    ``after_update(n)`` is called after every optimizer step that reaches
    ``n`` updates; the pipeline uses it to snapshot + evaluate adapters so a
    late drift can never destroy an earlier improvement (checkpoint selection).
    """
    if not rows:
        raise ValueError("RFT needs at least one training question")
    trainable = [p for p in ref.model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("The reformulator has no trainable parameters; "
                           "did the LoRA adapter load?")
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate),
                                  weight_decay=0.0)

    log(f"      RFT: {len(rows)} questions × {epochs} epoch(s), "
        f"{candidates} candidates, {batch_questions} targets/update, "
        f"lr={learning_rate}, margin={margin}, conf_slack={confidence_slack}, "
        f"skip_solved={skip_solved}")
    log(f"      RFT trainable params: {sum(p.numel() for p in trainable):,}")

    pending = []          # [(question, token_ids)]
    losses = []
    records = []
    scanned = scored = valid_actions = 0
    selected_questions = 0
    solved_skipped = 0
    question_errors = 0
    update_skipped = 0
    original_reward_sum = 0.0
    selected_reward_sum = 0.0
    selected_original_reward_sum = 0.0
    candidate_reward_sum = 0.0
    candidate_count = 0
    started = time.perf_counter()
    ref.model.train()

    def flush():
        nonlocal update_skipped
        if not pending:
            return None
        try:
            logprobs = ref.sequence_logprob_batch(pending, max_input)
            loss = sft_batch_step(list(logprobs), optimizer, trainable, clip=1.0)
        except (FloatingPointError, RuntimeError) as exc:
            update_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            log(f"      RFT skipped a non-finite update ({type(exc).__name__}: {exc})")
            pending.clear()
            return None
        losses.append(loss)
        pending.clear()
        if after_update is not None:
            ref.model.train()
            after_update(len(losses))
        return loss

    for epoch in range(epochs):
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        log(f"      RFT epoch {epoch + 1}/{epochs} …")
        for chunk in _chunks(order, max(1, generate_batch)):
            questions = [rows[i]["question"] for i in chunk]
            try:
                actions_per_question = ref.sample_actions_batch(
                    questions, candidates, max_input, max_new, do_sample)
            except RuntimeError as exc:
                question_errors += len(chunk)
                log(f"      RFT generation failed for {len(chunk)} questions: {exc}")
                continue

            for row, question, actions in zip((rows[i] for i in chunk),
                                              questions, actions_per_question):
                scanned += 1
                try:
                    answers = qa.answer_batch(
                        [question] + [action["candidate"] for action in actions],
                        row["context"])
                except RuntimeError as exc:
                    question_errors += 1
                    log(f"      RFT QA scoring failed ({type(exc).__name__}): {exc}")
                    continue
                original_reward = reward(answers[0][0], row["answers"],
                                         valid=True, preserve=1.0)
                original_confidence = answers[0][2]
                original_reward_sum += original_reward
                if (skip_solved and margin >= 0.0
                        and token_f1(answers[0][0], row["answers"]) >= 0.99):
                    # The original already answers correctly: no headroom.
                    solved_skipped += 1
                    continue

                values = []
                confidences = []
                for action, (answer, _, confidence) in zip(actions, answers[1:]):
                    preserve = entity_signature(question).issubset(
                        entity_signature(action["sampled_text"]))
                    values.append(reward(answer, row["answers"],
                                         valid=action["valid"],
                                         preserve=float(preserve)))
                    confidences.append(confidence)
                    valid_actions += int(action["valid"])
                    candidate_reward_sum += values[-1]
                    candidate_count += 1
                scored += len(actions)

                eligible = [i for i in range(len(actions))
                            if actions[i]["valid"]
                            and values[i] + 1e-9 >= original_reward + margin
                            and confidences[i] >= original_confidence - confidence_slack]
                if not eligible:
                    continue
                best = max(eligible, key=lambda i: (values[i], confidences[i]))
                selected_questions += 1
                selected_reward_sum += values[best]
                selected_original_reward_sum += original_reward
                pending.append((question, actions[best]["token_ids"]))
                if len(records) < 2000:
                    records.append({
                        "epoch": epoch + 1,
                        "question": question,
                        "original_reward": original_reward,
                        "original_confidence": original_confidence,
                        "selected_reward": values[best],
                        "selected_confidence": confidences[best],
                        "candidate_rewards": values,
                        "candidate_validity": [a["valid"] for a in actions],
                        "selected_text": actions[best]["sampled_text"],
                        "selected_valid": actions[best]["valid"],
                        "reward_lift": values[best] - original_reward,
                    })
                if len(pending) >= batch_questions:
                    flush()
                    if max_updates is not None and len(losses) >= max_updates:
                        break

            if max_updates is not None and len(losses) >= max_updates:
                break
            if scanned and scanned % 24 == 0:
                recent = records[-24:]
                elapsed = time.perf_counter() - started
                rate = scanned / max(1e-9, elapsed)
                log(f"      RFT {scanned}/{len(rows) * epochs} scanned  "
                    f"updates={len(losses)}  "
                    f"valid={valid_actions / max(1, scored):.2f}  "
                    f"selected={selected_questions / scanned:.2f}  "
                    f"lift={sum(x['reward_lift'] for x in recent) / max(1, len(recent)):+.3f}  "
                    f"loss={(sum(losses[-20:]) / len(losses[-20:])) if losses else float('nan'):.3f}  "
                    f"{rate:.2f} q/s")
        if max_updates is not None and len(losses) >= max_updates:
            if pending:
                flush()
            break
    if pending:
        flush()

    elapsed = time.perf_counter() - started
    if question_errors and scanned and question_errors / max(1, scanned) > 0.10:
        raise RuntimeError(f"RFT could not score {question_errors}/{scanned} questions; "
                           "aborting instead of training on a broken loop")
    metrics = {
        "questions_seen": scanned,
        "candidate_actions": scored,
        "optimizer_steps": len(losses),
        "skipped_updates": update_skipped,
        "solved_skipped": solved_skipped,
        "solved_skip_rate": round(solved_skipped / max(1, scanned), 4),
        "question_errors": question_errors,
        "valid_generation_rate": round(valid_actions / max(1, scored), 4),
        "selected_rate": round(selected_questions / max(1, scanned), 4),
        "mean_original_reward": round(original_reward_sum / max(1, scanned), 4),
        "mean_candidate_reward": round(candidate_reward_sum / max(1, candidate_count), 4),
        "mean_selected_reward": round(selected_reward_sum / max(1, selected_questions), 4),
        "mean_reward_lift": round((selected_reward_sum - selected_original_reward_sum)
                                  / max(1, selected_questions), 4),
        "initial_loss": round(losses[0], 4) if losses else None,
        "final_loss": round(sum(losses[-20:]) / len(losses[-20:]), 4) if losses else None,
        "elapsed_seconds": round(elapsed, 1),
        "questions_per_second": round(scanned / max(1e-9, elapsed), 2),
        "settings": {
            "candidates_per_question": candidates,
            "batch_questions": batch_questions,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "min_reward_margin": margin,
            "confidence_slack": confidence_slack,
            "max_input_tokens": max_input,
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "seed": seed,
            "skip_solved_originals": skip_solved,
        },
    }
    log(f"      RFT done: {scanned} questions, {len(losses)} updates, "
        f"selected={metrics['selected_rate']:.2f}, "
        f"lift={metrics['mean_reward_lift']:+.3f}, "
        f"{elapsed / 60:.1f} min")
    return metrics, records
