"""REINFORCE training for the Telugu reformulation policy.

The reward for a sampled rewrite is computed on the canonical supporting
passage only and is centred against the reward of the original question. That
per-question baseline removes the "is this question easy?" component from the
advantage, so the policy is pushed toward rewrites that actually beat not
rewriting. Gradients are accumulated over several questions before each
optimizer step to keep the update variance manageable.
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
from activeqa.policy_gradient import reinforce_batch_step
from activeqa.preprocessing import entity_signature
from activeqa.qa_model import TeluguQA
from activeqa.reformulator import Reformulator
from activeqa.rewards import reward
from activeqa.utils import load_config, setup_cpu


def save_checkpoint(path, ref, optimizer, update_step, action_step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from peft import get_peft_model_state_dict
    torch.save({
        "adapter": get_peft_model_state_dict(ref.model),
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

    train, evaluation, _retrieval_corpus = load_splits(cfg["data"], include_full_train=True)
    save_splits(train, evaluation)

    candidates = int(cfg["generation"]["candidates_per_question"])
    epochs = int(cfg["training"]["epochs"])
    batch_questions = max(1, int(cfg["training"].get("batch_questions", 8)))
    requested = len(train) * epochs * candidates
    print(f"device={cfg.get('device', 'cpu')} train={len(train)} heldout={len(evaluation)} "
          f"candidates_per_question={candidates} epochs={epochs} "
          f"batch_questions={batch_questions} sampled_actions={requested}", flush=True)

    started = time.perf_counter()
    ref = Reformulator(cfg["models"]["reformulator"], cfg["models"]["cache_dir"],
                       cfg["models"]["lora_rank"], cfg["models"]["lora_alpha"],
                       device=cfg.get("device", "cpu"))
    print(f"reformulator_load_seconds={time.perf_counter()-started:.1f}; "
          f"trainable_parameters={ref.trainable_count}", flush=True)
    qa_started = time.perf_counter()
    qa = TeluguQA(cfg["models"]["qa"], cfg["models"]["cache_dir"], device=cfg.get("device", "cpu"))
    print(f"qa_load_seconds={time.perf_counter()-qa_started:.1f}", flush=True)

    optimizer = torch.optim.AdamW(ref.trainable, lr=cfg["training"]["learning_rate"])
    checkpoint = cfg["training"]["checkpoint"]
    checkpoint_every = int(cfg["training"].get("checkpoint_every_steps", 0))

    action_step = 0
    update_step = 0
    start_action = 0
    if args.resume:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Cannot resume: checkpoint does not exist: {checkpoint}")
        from peft import set_peft_model_state_dict
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("base_model") != ref.base_name:
            raise ValueError("Checkpoint base model does not match configured reformulator")
        set_peft_model_state_dict(ref.model, saved["adapter"])
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
    pending = []  # (logprob tensor, advantage float, reward float)
    rows_in_batch = 0

    def flush_batch(epoch):
        nonlocal update_step, peak_rss
        if not pending:
            return
        logprobs = [item[0] for item in pending]
        advantages = [item[1] for item in pending]
        rewards = [item[2] for item in pending]
        loss = reinforce_batch_step(
            logprobs, advantages, optimizer, ref.trainable,
            cfg["training"]["gradient_clip_norm"])
        update_step += 1
        update_log.append({
            "epoch": epoch,
            "update_step": update_step,
            "actions": len(pending),
            "loss": loss,
            "mean_advantage": sum(advantages) / len(advantages),
            "mean_reward": sum(rewards) / len(rewards),
        })
        pending.clear()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if checkpoint_every > 0 and update_step % checkpoint_every == 0:
            save_checkpoint(checkpoint, ref, optimizer, update_step, action_step)

    for epoch in range(epochs):
        progress = tqdm(train, desc=f"epoch {epoch+1}/{epochs}", unit="q", dynamic_ncols=True)
        for row in progress:
            # Exact resume at question granularity: checkpoints are written at
            # the end of a whole batch, so start_action is a multiple of candidates.
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

            # Per-question baseline: the reward of leaving the question alone,
            # measured on the same canonical context with the same reward.
            original_reward = reward(qa_answers[0][0], row["answers"],
                                     valid=True, preserve=1.0)

            # One batched teacher-forcing pass for all sampled actions.
            logprobs = ref.sequence_logprob_ids_batch(
                row["question"], [action["token_ids"] for action in sampled],
                cfg["generation"]["max_input_tokens"])

            for action, (answer, _, _), logprob in zip(sampled, qa_answers[1:], logprobs):
                action_started = time.perf_counter()
                scored_question = action["candidate"]
                preserves_entities = entity_signature(row["question"]).issubset(
                    entity_signature(action["sampled_text"]))
                # reward() already applies the entity bonus and invalid penalty,
                # so it is added exactly once here.
                value = reward(answer, row["answers"], valid=action["valid"],
                               preserve=float(preserves_entities))
                advantage = value - original_reward
                pending.append((logprob, advantage, value))
                action_step += 1
                action_log.append({
                    "epoch": epoch + 1,
                    "action_step": action_step,
                    "reward": value,
                    "original_reward": original_reward,
                    "advantage": advantage,
                    "valid": action["valid"],
                    "preserves_entities": preserves_entities,
                    "preserves_question_type": action.get("question_type_preserved", True),
                    "action_seconds": time.perf_counter() - action_started,
                    "sampled_text": action["sampled_text"],
                    "scored_question": scored_question,
                })

            rows_in_batch += 1
            if rows_in_batch >= batch_questions:
                flush_batch(epoch + 1)
                rows_in_batch = 0

            recent = action_log[-min(len(action_log), 50):]
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

    flush_batch(epochs)
    save_checkpoint(checkpoint, ref, optimizer, update_step, action_step)
    os.makedirs("outputs", exist_ok=True)
    elapsed = elapsed_offset + (time.perf_counter() - started)
    valid_rate = sum(x["valid"] for x in action_log) / max(1, len(action_log))
    mean_reward = sum(x["reward"] for x in action_log) / max(1, len(action_log))
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump({
            "trainable_parameters": ref.trainable_count,
            "actions": action_log,
            "updates": update_log,
            "elapsed_seconds": elapsed,
            "peak_rss_mb": peak_rss / 1e6,
            "valid_generation_rate": valid_rate,
            "mean_reward": mean_reward,
            "optimizer_steps": update_step,
        }, stream, ensure_ascii=False, indent=2)
    print(f"training_seconds={elapsed:.1f} optimizer_steps={update_step} "
          f"peak_rss_mb={peak_rss/1e6:.1f} valid_generation_rate={valid_rate:.3f} "
          f"mean_reward={mean_reward:.3f}", flush=True)


if __name__ == "__main__":
    main()
