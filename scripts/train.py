"""Policy improvement for the Telugu reformulation model.

Two objectives are supported:

``reinforce`` (default)
    Sample several rewrites per question and update their log-probabilities with
    a per-question baseline. ``advantage = reward(rewrite) - reward(original)``.

``rft`` (optional supervised alternative)
    Rejection-sampling fine-tuning. Sample several rewrites, score them on the
    canonical passage, keep the best *valid* rewrite that is at least as good as
    the original question, and teacher-force that sequence. Batches with no good
    sample are skipped, so the policy is never pushed away from every sample at
    once. This can be more stable than REINFORCE when the QA reward is sparse,
    but it is not a policy-gradient RL update.

Both methods use the canonical supporting passage only, batch QA scoring across
candidates, and write resumable LoRA checkpoints.
"""
import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, "src")
from activeqa.data import load_splits, save_splits
from activeqa.policy_gradient import reinforce_batch_step, sft_batch_step
from activeqa.preprocessing import entity_signature
from activeqa.qa_model import TeluguQA
from activeqa.reformulator import Reformulator
from activeqa.rewards import reward, token_f1
from activeqa.utils import load_config, setup_cpu, make_policy_optimizer, load_policy_state


def step_checkpoint_path(path, step):
    root, ext = os.path.splitext(path)
    return f"{root}_step{step}{ext or '.pt'}"


def save_checkpoint(path, ref, optimizer, update_step, action_step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state": ref.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update_step": update_step,
        "action_step": action_step,
        "step": action_step,  # backwards-compatible key for older tooling
        "base_model": ref.base_name,
        "torch_rng_state": torch.get_rng_state(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_cpu(cfg["seed"], cfg["cpu_threads"], cfg.get("device", "cpu"))

    method = str(cfg["training"].get("method", "reinforce")).lower()
    if method not in {"reinforce", "rft"}:
        raise ValueError(f"Unknown training.method={method!r}; use 'reinforce' or 'rft'")
    margin = float(cfg["training"].get("min_reward_margin", 0.0))
    collect_selector_data = bool(cfg["training"].get("collect_selector_data", False))
    selector_records = []
    selector_path = os.path.join("outputs", "selector_dataset.jsonl")

    if bool(cfg["training"].get("skip_training", False)):
        # Guard the eval-only phase: refuse to touch (or clobber) the trained
        # checkpoint; evaluation is driven separately by scripts/evaluate.py.
        checkpoint_exists = os.path.exists(cfg["training"]["checkpoint"])
        if not checkpoint_exists:
            raise FileNotFoundError(f"training.skip_training requires an existing checkpoint: "
                                    f"{cfg['training']['checkpoint']}")
        print("skip_training=true; nothing to do (checkpoint kept as is)", flush=True)
        return

    train, evaluation, _retrieval_corpus = load_splits(cfg["data"], include_full_train=True)
    save_splits(train, evaluation)

    candidates = int(cfg["generation"]["candidates_per_question"])
    epochs = int(cfg["training"]["epochs"])
    batch_questions = max(1, int(cfg["training"].get("batch_questions", 8)))
    requested = len(train) * epochs * candidates
    print(f"method={method} device={cfg.get('device', 'cpu')} train={len(train)} "
          f"heldout={len(evaluation)} candidates_per_question={candidates} epochs={epochs} "
          f"batch_questions={batch_questions} sampled_actions={requested}", flush=True)

    started = time.perf_counter()
    ref = Reformulator(cfg["models"]["reformulator"], cfg["models"]["cache_dir"],
                       cfg["models"].get("lora_rank"), cfg["models"].get("lora_alpha"),
                       device=cfg.get("device", "cpu"))
    print(f"reformulator_load_seconds={time.perf_counter()-started:.1f}; "
          f"trainable_parameters={ref.trainable_count}", flush=True)
    if bool(cfg["training"].get("sft_warmup", False)):
        if not args.resume:
            sft_checkpoint = cfg["training"].get(
                "sft_checkpoint", "outputs/external_rl_sft_policy.pt")
            if not os.path.isfile(sft_checkpoint):
                raise FileNotFoundError(
                    f"SFT warm-up checkpoint missing: {sft_checkpoint}. "
                    "Run scripts/sft_warmup.py first.")
            sft_saved = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
            if sft_saved.get("base_model") != ref.base_name or not sft_saved.get("complete"):
                raise ValueError("SFT warm-up checkpoint is incomplete or uses a different base model")
            load_policy_state(ref.model, sft_saved)
            print(f"loaded_reformulation_sft_warmup={sft_checkpoint}", flush=True)
    qa_started = time.perf_counter()
    qa = TeluguQA(cfg["models"]["qa"], cfg["models"]["cache_dir"], device=cfg.get("device", "cpu"))
    print(f"qa_load_seconds={time.perf_counter()-qa_started:.1f}", flush=True)

    optimizer = make_policy_optimizer(ref.trainable, cfg["training"]["learning_rate"])
    checkpoint = cfg["training"]["checkpoint"]
    checkpoint_every = int(cfg["training"].get("checkpoint_every_steps", 0))
    keep_step_checkpoints = bool(cfg["training"].get("keep_step_checkpoints", False))

    action_step = 0
    update_step = 0
    start_action = 0
    if args.resume:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Cannot resume: checkpoint does not exist: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("base_model") != ref.base_name:
            raise ValueError("Checkpoint base model does not match configured reformulator")
        load_policy_state(ref.model, saved)
        optimizer.load_state_dict(saved["optimizer"])
        start_action = int(saved.get("action_step", saved.get("step", 0)))
        update_step = int(saved.get("update_step", 0))
        if saved.get("torch_rng_state") is not None:
            torch.set_rng_state(saved["torch_rng_state"])
        print(f"resumed_from_action_step={start_action} update_step={update_step}", flush=True)

    started = time.perf_counter()
    process = __import__("psutil").Process()
    peak_rss = process.memory_info().rss
    action_log = []
    update_log = []
    elapsed_offset = 0.0
    metrics_path = os.path.join("outputs", "train_metrics.json")
    if args.resume and os.path.exists(metrics_path):
        # Keep the metrics collected before the checkpoint so a resumed run does
        # not replace them with an empty/partial log.
        try:
            with open(metrics_path, encoding="utf-8") as stream:
                previous = json.load(stream)
            action_log = previous.get("actions", [])
            update_log = previous.get("updates", [])
            elapsed_offset = float(previous.get("elapsed_seconds", 0.0))
            peak_rss = max(peak_rss, float(previous.get("peak_rss_mb", 0.0)) * 1e6)
            print(f"resumed_metrics actions={len(action_log)} "
                  f"updates={len(update_log)}", flush=True)
        except Exception as exc:
            print(f"could_not_load_previous_metrics={exc}", flush=True)
    if args.resume and collect_selector_data and os.path.exists(selector_path):
        with open(selector_path, encoding="utf-8") as stream:
            selector_records = [json.loads(line) for line in stream if line.strip()]
        print(f"resumed_selector_records={len(selector_records)}", flush=True)
    pending = []  # REINFORCE: (logprob, advantage, reward); RFT: (logprob, best, original)
    rows_in_batch = 0
    confidences = []  # span confidence per sampled rewrite (deployment alignment)

    def flush_batch(epoch):
        nonlocal update_step, peak_rss
        if not pending:
            return
        if method == "rft":
            loss = sft_batch_step([item[0] for item in pending], optimizer,
                                  ref.trainable, cfg["training"]["gradient_clip_norm"])
            update_log.append({
                "epoch": epoch,
                "update_step": update_step + 1,
                "targets": len(pending),
                "loss": loss,
                "mean_selected_reward": sum(item[1] for item in pending) / len(pending),
                "mean_original_reward": sum(item[2] for item in pending) / len(pending),
            })
        else:
            logprobs = [item[0] for item in pending]
            advantages = [item[1] for item in pending]
            rewards = [item[2] for item in pending]
            loss = reinforce_batch_step(
                logprobs, advantages, optimizer, ref.trainable,
                cfg["training"]["gradient_clip_norm"])
            update_log.append({
                "epoch": epoch,
                "update_step": update_step + 1,
                "actions": len(pending),
                "loss": loss,
                "mean_advantage": sum(advantages) / len(advantages),
                "mean_reward": sum(rewards) / len(rewards),
            })
        update_step += 1
        pending.clear()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if checkpoint_every > 0 and update_step % checkpoint_every == 0:
            save_checkpoint(checkpoint, ref, optimizer, update_step, action_step)
            if keep_step_checkpoints:
                save_checkpoint(step_checkpoint_path(checkpoint, update_step),
                                ref, optimizer, update_step, action_step)

    last_selector_write = 0

    def write_selector_dataset():
        nonlocal last_selector_write
        if not collect_selector_data or len(selector_records) == last_selector_write:
            return
        os.makedirs("outputs", exist_ok=True)
        with open(selector_path, "w", encoding="utf-8") as stream:
            for record in selector_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"selector_dataset_written questions={len(selector_records)} "
              f"path={selector_path}", flush=True)
        last_selector_write = len(selector_records)

    for epoch in range(epochs):
        progress = tqdm(train, desc=f"epoch {epoch+1}/{epochs}", unit="q", dynamic_ncols=True)
        for row in progress:
            # Resume at question granularity: checkpoints are written after a
            # whole optimizer batch, so start_action is a multiple of candidates.
            if action_step + candidates <= start_action:
                action_step += candidates
                continue

            sampled = ref.sample_actions(
                row["question"], candidates,
                cfg["generation"]["max_input_tokens"],
                cfg["generation"]["max_new_tokens"],
                bool(cfg["generation"].get("do_sample", True)),
            )

            # One batched QA pass scores the original question and every sampled
            # rewrite on the canonical context.
            qa_answers = qa.answer_batch(
                [row["question"]] + [action["candidate"] for action in sampled],
                row["context"])

            # Per-question baseline: reward of leaving the question untouched.
            original_reward = reward(qa_answers[0][0], row["answers"],
                                     valid=True, preserve=1.0)
            original_confidence = qa_answers[0][2]
            values = []
            confidences.clear()
            for action, (answer, _, confidence) in zip(sampled, qa_answers[1:]):
                preserves_entities = entity_signature(row["question"]).issubset(
                    entity_signature(action["sampled_text"]))
                # reward() already applies the entity bonus and invalid penalty.
                values.append(reward(answer, row["answers"], valid=action["valid"],
                                     preserve=float(preserves_entities)))
                confidences.append(confidence)

            if method == "rft":
                question_step = action_step // candidates + 1
                action_step += candidates
                # Deployment alignment: offline the reward is gold-labelled, but
                # at test time the answer is chosen by span confidence among
                # candidates whose extracted answer stays consistent. Teaching
                # the policy rewrites that only look better under the gold
                # label yet never win the confidence ranking produced the
                # observed RFT gap, so require the rewrite to also win (or tie)
                # the span-confidence contest against the original question.
                best_index = max(
                    (i for i in range(len(sampled))
                     if sampled[i]["valid"]
                     and values[i] >= original_reward + margin
                     and confidences[i] >= original_confidence),
                    key=lambda i: values[i], default=None)
                diag_index = max(range(len(sampled)), key=values.__getitem__)
                best_action = sampled[diag_index]
                best_reward = values[diag_index]
                best_confidence = confidences[diag_index]
                selected = False
                if best_index is not None:
                    # Gradient target uses the deployment-aligned winner.
                    best_action = sampled[best_index]
                    best_reward = values[best_index]
                    best_confidence = confidences[best_index]
                    selected = True

                action_log.append({
                    "epoch": epoch + 1,
                    "question_step": question_step,
                    "original_reward": original_reward,
                    "best_reward": best_reward,
                    "best_index": best_index,
                    "diag_index": diag_index,
                    "selected": selected,
                    "best_confidence": best_confidence,
                    "original_confidence": original_confidence,
                    "valid": best_action["valid"],
                    "candidate_rewards": values,
                    "candidate_validity": [a["valid"] for a in sampled],
                    "target_text": best_action["sampled_text"],
                })
                if collect_selector_data:
                    # Candidate pool for the offline ranker. Gold answers label
                    # these records, but the features stored here are all
                    # reference-free (see src/activeqa/ranker.py) — the same
                    # features the deployed selector will compute. Invalid
                    # candidates are excluded: at deployment they surface as the
                    # original question, so they are not real choices.
                    f1s = [token_f1(answer, row["answers"])
                           for (answer, _, _) in qa_answers[1:]]
                    selector_records.append({
                        "id": row["id"],
                        "epoch": epoch + 1,
                        "question": row["question"],
                        "original_answer": qa_answers[0][0],
                        "original_confidence": qa_answers[0][2],
                        "original_f1": token_f1(qa_answers[0][0], row["answers"]),
                        "candidates": [a["candidate"] for a in sampled],
                        "validity": [a["valid"] for a in sampled],
                        "answers": [x[0] for x in qa_answers[1:]],
                        "confidences": [x[2] for x in qa_answers[1:]],
                        "f1s": f1s,
                    })

                if selected:
                    logprob = ref.sequence_logprob_ids(
                        row["question"], best_action["token_ids"],
                        cfg["generation"]["max_input_tokens"])
                    pending.append((logprob, best_reward, original_reward))
                    rows_in_batch += 1
                    if rows_in_batch >= batch_questions:
                        flush_batch(epoch + 1)
                        rows_in_batch = 0
            else:
                # One batched teacher-forcing pass for all sampled actions.
                logprobs = ref.sequence_logprob_ids_batch(
                    row["question"], [action["token_ids"] for action in sampled],
                    cfg["generation"]["max_input_tokens"])
                for action, value, logprob in zip(sampled, values, logprobs):
                    pending.append((logprob, value - original_reward, value))
                    action_log.append({
                        "epoch": epoch + 1,
                        "action_step": action_step + 1,
                        "reward": value,
                        "original_reward": original_reward,
                        "advantage": value - original_reward,
                        "valid": action["valid"],
                        "preserves_entities": entity_signature(row["question"]).issubset(
                            entity_signature(action["sampled_text"])),
                        "preserves_question_type": action.get("question_type_preserved", True),
                        "sampled_text": action["sampled_text"],
                        "scored_question": action["candidate"],
                    })
                    action_step += 1
                rows_in_batch += 1
                if rows_in_batch >= batch_questions:
                    flush_batch(epoch + 1)
                    rows_in_batch = 0

            recent = action_log[-min(len(action_log), 50):]
            if method == "rft":
                progress.set_postfix(
                    updates=update_step,
                    best=f"{sum(x['best_reward'] for x in recent)/len(recent):.3f}",
                    orig=f"{sum(x['original_reward'] for x in recent)/len(recent):.3f}",
                    sel=f"{sum(x['selected'] for x in recent)/len(recent):.2f}",
                    valid=f"{sum(x['valid'] for x in recent)/len(recent):.2f}",
                    elapsed=f"{(time.perf_counter()-started)/60:.1f}m",
                    rss=f"{process.memory_info().rss/1e9:.2f}GB",
                )
            else:
                progress.set_postfix(
                    updates=update_step,
                    reward=f"{sum(x['reward'] for x in recent)/len(recent):.3f}",
                    adv=f"{sum(x['advantage'] for x in recent)/len(recent):+.3f}",
                    valid=f"{sum(x['valid'] for x in recent)/len(recent):.2f}",
                    elapsed=f"{(time.perf_counter()-started)/60:.1f}m",
                    rss=f"{process.memory_info().rss/1e9:.2f}GB",
                )

        flush_batch(epoch + 1)
        rows_in_batch = 0
        write_selector_dataset()

    flush_batch(epochs)
    write_selector_dataset()
    save_checkpoint(checkpoint, ref, optimizer, update_step, action_step)
    os.makedirs("outputs", exist_ok=True)
    elapsed = elapsed_offset + (time.perf_counter() - started)

    if method == "rft":
        candidate_count = sum(len(x["candidate_validity"]) for x in action_log)
        valid_rate = sum(sum(x["candidate_validity"]) for x in action_log) / max(1, candidate_count)
        valid_rate_questions = sum(x["valid"] for x in action_log) / max(1, len(action_log))
        mean_reward = sum(x["best_reward"] for x in action_log) / max(1, len(action_log))
        selected_rate = sum(x["selected"] for x in action_log) / max(1, len(action_log))
        selected_rows = [x for x in action_log if x["selected"]]
        mean_selected_reward = (sum(x["best_reward"] for x in selected_rows) /
                                max(1, len(selected_rows)))
        extra = {
            "valid_questions_rate": valid_rate_questions,
            "selected_rate": selected_rate,
            "mean_selected_reward": mean_selected_reward,
        }
    else:
        valid_rate = sum(x["valid"] for x in action_log) / max(1, len(action_log))
        mean_reward = sum(x["reward"] for x in action_log) / max(1, len(action_log))
        extra = {}

    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump({
            "method": method,
            "log_unit": "question" if method == "rft" else "action",
            "trainable_parameters": ref.trainable_count,
            "actions": action_log,
            "updates": update_log,
            "elapsed_seconds": elapsed,
            "peak_rss_mb": peak_rss / 1e6,
            "valid_generation_rate": valid_rate,
            "mean_reward": mean_reward,
            "optimizer_steps": update_step,
            **extra,
        }, stream, ensure_ascii=False, indent=2)
    print(f"training_seconds={elapsed:.1f} optimizer_steps={update_step} "
          f"peak_rss_mb={peak_rss/1e6:.1f} valid_generation_rate={valid_rate:.3f} "
          f"mean_reward={mean_reward:.3f}"
          + (f" selected_rate={extra['selected_rate']:.3f}" if method == "rft" else ""),
          flush=True)


if __name__ == "__main__":
    main()
