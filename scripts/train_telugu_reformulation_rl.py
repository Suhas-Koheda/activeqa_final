#!/usr/bin/env python3
"""Full-parameter reference-reward REINFORCE for Telugu question reformulation.

This script performs policy-gradient RL only: it does not do teacher-forced SFT
and does not use LoRA. A generated question is rewarded by similarity to the
clean, user-provided reformulations for that source question, plus strict checks
for Telugu script, answer type, topic anchors, and non-copying.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, "src")
from activeqa.preprocessing import normalize, tokens
from activeqa.policy_gradient import reinforce_batch_step
from activeqa.reformulation_data import load_clean_splits
from activeqa.reformulator import Reformulator
from activeqa.utils import load_config, make_policy_optimizer, setup_cpu


def reference_f1(prediction: str, references: list[str]) -> float:
    pred = tokens(prediction)
    best = 0.0
    for reference in references:
        ref = tokens(reference)
        if not pred or not ref:
            continue
        from collections import Counter
        overlap = sum((Counter(pred) & Counter(ref)).values())
        precision, recall = overlap / len(pred), overlap / len(ref)
        best = max(best, 2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0)
    return best


def action_reward(question: str, action: dict, references: list[str]) -> float:
    text = normalize(action["sampled_text"])
    similarity = reference_f1(text, references)
    if not action["valid"]:
        return -0.25 + 0.1 * similarity
    if normalize(text).rstrip(" ?") == normalize(question).rstrip(" ?"):
        return -0.25
    # Similarity provides the learning signal; the validity term makes malformed,
    # off-topic, or answer-type-changing strings strictly less attractive.
    return 0.15 + 0.85 * similarity


def grouped(rows: list[dict]) -> list[dict]:
    result: dict[str, dict] = {}
    for row in rows:
        group = result.setdefault(row["source_id"], {
            "source_id": row["source_id"], "question": row["question"],
            "references": [],
        })
        for ref in row.get("references", [row["reformulation"]]):
            if ref not in group["references"]:
                group["references"].append(ref)
    return list(result.values())


def evaluate_policy(ref: Reformulator, examples: list[dict], gen_cfg: dict,
                    count: int) -> tuple[dict, list[dict]]:
    scores, valid_n, total_n = [], 0, 0
    predictions = []
    ref.model.eval()
    for row in tqdm(examples, desc="held-out reference check", unit="q"):
        actions = ref.sample_actions(row["question"], count,
                                     int(gen_cfg["max_input_tokens"]),
                                     int(gen_cfg["max_new_tokens"]), False)
        values = [action_reward(row["question"], action, row["references"])
                  for action in actions]
        scores.extend(values)
        valid_n += sum(int(action["valid"]) for action in actions)
        total_n += len(actions)
        predictions.append({"question": row["question"],
                            "references": row["references"],
                            "candidates": [a["sampled_text"] for a in actions],
                            "valid": [a["valid"] for a in actions],
                            "reference_rewards": values})
    return {"questions": len(examples), "candidates": total_n,
            "mean_reference_reward": sum(scores) / max(1, len(scores)),
            "valid_generation_rate": valid_n / max(1, total_n)}, predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/colab_user_data_rl.yaml")
    ap.add_argument("--input", default="data/indic_rag_te/user_provided_reformulations.jsonl")
    ap.add_argument("--data-dir", default="data/indic_rag_te/rl_reference_pairs")
    ap.add_argument("--checkpoint", default="outputs/telugu_reference_reinforce.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_cpu(cfg["seed"], cfg["cpu_threads"], cfg.get("device", "cpu"))
    train_pairs, dev_pairs, rejected = load_clean_splits(
        args.input, seed=int(cfg["seed"]),
        dev_fraction=float(cfg["data"].get("eval_fraction", 0.2)),
        output_dir=args.data_dir)
    train, dev = grouped(train_pairs), grouped(dev_pairs)
    if not train or not dev:
        raise RuntimeError("Strict filtering left no train or held-out questions; inspect filter_report.json")
    print(f"filtered_reference_data train_questions={len(train)} train_pairs={len(train_pairs)} "
          f"dev_questions={len(dev)} dev_pairs={len(dev_pairs)} rejected={rejected}", flush=True)

    model_cfg, gen_cfg, tr_cfg = cfg["models"], cfg["generation"], cfg["training"]
    device = cfg.get("device", "cpu")
    policy = Reformulator(model_cfg["reformulator"], model_cfg.get("cache_dir", "models"),
                          device=device)
    # Enforce the stated full-parameter RL run. Refuse a future accidental adapter-only config.
    if not policy.trainable or policy.trainable_count < 100_000_000:
        raise RuntimeError(f"Expected full-model policy parameters; got {policy.trainable_count}")
    optimizer = make_policy_optimizer(policy.trainable,
                                      float(tr_cfg.get("reference_rl_learning_rate", 1e-6)))
    checkpoint = Path(args.checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint.with_name(checkpoint.stem + "_metrics.json")
    epochs = args.epochs or int(tr_cfg.get("reference_rl_epochs", 3))
    n_candidates = int(gen_cfg.get("candidates_per_question", 3))
    max_input, max_new = int(gen_cfg["max_input_tokens"]), int(gen_cfg["max_new_tokens"])
    start_epoch = 0
    if args.resume:
        if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("base_model") != policy.base_name:
            raise ValueError("Checkpoint base model does not match the configured base")
        policy.model.load_state_dict(saved["model_state"])
        if "optimizer" in saved: optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved.get("epoch", 0))

    started = time.perf_counter()
    all_rewards, step = [], 0
    checkpoint_every = max(1, int(tr_cfg.get("reference_rl_checkpoint_every", 20)))
    # Retain the pretrained policy unless an RL epoch improves held-out reference
    # reward. This prevents publishing the last epoch if policy-gradient updates
    # collapse generation quality.
    pretrained_metrics, _ = evaluate_policy(policy, dev, gen_cfg, n_candidates)
    best_metrics = pretrained_metrics
    best_score = best_metrics["mean_reference_reward"]
    best_state = {k: v.detach().cpu().clone()
                  for k, v in policy.model.state_dict().items()}
    print(f"pretrained_heldout_reference_reward={best_score:.4f}", flush=True)

    def save(epoch: int, complete: bool = False):
        torch.save({"model_state": policy.model.state_dict(),
                    "optimizer": optimizer.state_dict(), "base_model": policy.base_name,
                    "epoch": epoch, "complete": complete,
                    "method": "reference_reward_REINFORCE_full_parameter",
                    "torch_rng_state": torch.get_rng_state()}, checkpoint)

    for epoch in range(start_epoch, epochs):
        order = list(range(len(train)))
        random.Random(int(cfg["seed"]) + epoch).shuffle(order)
        progress = tqdm(order, desc=f"reference REINFORCE {epoch + 1}/{epochs}", unit="q")
        for pos in progress:
            row = train[pos]
            actions = policy.sample_actions(row["question"], n_candidates,
                                            max_input, max_new, do_sample=True)
            rewards = [action_reward(row["question"], action, row["references"])
                       for action in actions]
            seen_outputs = set()
            for i, action in enumerate(actions):
                key = normalize(action["sampled_text"]).casefold().rstrip(" ?")
                if key in seen_outputs:
                    rewards[i] -= 0.15
                seen_outputs.add(key)
            logprobs = policy.sequence_logprob_ids_batch(
                row["question"], [action["token_ids"] for action in actions], max_input)
            # Per-question REINFORCE baseline. The group mean reduces variance and
            # cannot reward every bad sample just because all candidates are poor.
            baseline = sum(rewards) / len(rewards)
            advantages = [score - baseline for score in rewards]
            loss = reinforce_batch_step(logprobs.unbind(), advantages, optimizer,
                                        policy.trainable,
                                        float(tr_cfg.get("gradient_clip_norm", 1.0)))
            step += 1
            all_rewards.extend(rewards)
            progress.set_postfix(reward=f"{sum(rewards)/len(rewards):.3f}",
                                 valid=f"{sum(a['valid'] for a in actions)/len(actions):.2f}",
                                 loss=f"{loss:.3f}")
            if step % checkpoint_every == 0:
                save(epoch, complete=False)
        epoch_metrics, _ = evaluate_policy(policy, dev, gen_cfg, n_candidates)
        print(f"epoch={epoch + 1} heldout_reference_reward="
              f"{epoch_metrics['mean_reference_reward']:.4f} "
              f"valid={epoch_metrics['valid_generation_rate']:.3f}", flush=True)
        if epoch_metrics["mean_reference_reward"] > best_score:
            best_score = epoch_metrics["mean_reference_reward"]
            best_metrics = epoch_metrics
            best_state = {k: v.detach().cpu().clone()
                          for k, v in policy.model.state_dict().items()}
            save(epoch + 1, complete=False)

    policy.model.load_state_dict(best_state)
    policy.model.eval()
    heldout, predictions = evaluate_policy(policy, dev, gen_cfg, n_candidates)
    save(epochs, complete=True)
    result = {"method": "reference_reward_REINFORCE_full_parameter",
              "sft_used": False, "lora_used": False,
              "train_questions": len(train), "train_pairs": len(train_pairs),
              "heldout_questions": len(dev), "heldout_pairs": len(dev_pairs),
              "epochs": epochs, "optimizer_steps": step,
              "mean_sampled_reference_reward": sum(all_rewards) / max(1, len(all_rewards)),
              "pretrained_heldout": pretrained_metrics,
              "heldout": heldout, "best_heldout_reward": best_score,
              "seconds": time.perf_counter() - started,
              "checkpoint": str(checkpoint), "filtered_data_dir": args.data_dir,
              "human_verified_labels": False}
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    pred_path = checkpoint.with_name(checkpoint.stem + "_heldout.json")
    with pred_path.open("w", encoding="utf-8") as stream:
        json.dump(predictions, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
