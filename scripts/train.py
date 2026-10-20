"""CPU REINFORCE training for the reformulation policy."""
import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, "src")
from activeqa.data import load_splits, save_splits
from activeqa.policy_gradient import RunningBaseline, reinforce_step
from activeqa.preprocessing import entity_signature
from activeqa.qa_model import TeluguQA
from activeqa.reformulator import Reformulator
from activeqa.retrieval import Retriever
from activeqa.rewards import reward
from activeqa.utils import load_config, setup_cpu


def save_checkpoint(path, ref, optimizer, baseline, step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from peft import get_peft_model_state_dict
    torch.save({
        "adapter": get_peft_model_state_dict(ref.model),
        "optimizer": optimizer.state_dict(),
        "baseline": baseline.value,
        "step": step,
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
    train, evaluation = load_splits(cfg["data"])
    save_splits(train, evaluation)
    requested = len(train) * cfg["training"]["epochs"] * cfg["generation"]["candidates_per_question"]
    print(f"device=cpu train={len(train)} heldout={len(evaluation)} sampled_actions={requested}; "
          f"upper-bound QA calls={requested * (cfg['retrieval']['top_k'] + 1)}", flush=True)

    started = time.perf_counter()
    ref = Reformulator(cfg["models"]["reformulator"], cfg["models"]["cache_dir"],
                       cfg["models"]["lora_rank"], cfg["models"]["lora_alpha"],
                       device=cfg.get("device", "cpu"))
    print(f"reformulator_load_seconds={time.perf_counter()-started:.1f}; "
          f"trainable_parameters={ref.trainable_count}", flush=True)
    qa_started = time.perf_counter()
    qa = TeluguQA(cfg["models"]["qa"], cfg["models"]["cache_dir"], device=cfg.get("device", "cpu"))
    print(f"qa_load_seconds={time.perf_counter()-qa_started:.1f}", flush=True)

    retriever = Retriever(train)
    optimizer = torch.optim.AdamW(ref.trainable, lr=cfg["training"]["learning_rate"])
    baseline = RunningBaseline(cfg["training"]["baseline_momentum"])
    step = 0
    checkpoint = cfg["training"]["checkpoint"]
    if args.resume:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Cannot resume: checkpoint does not exist: {checkpoint}")
        from peft import set_peft_model_state_dict
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("base_model") != ref.base_name:
            raise ValueError("Checkpoint base model does not match configured reformulator")
        set_peft_model_state_dict(ref.model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        baseline.value = saved.get("baseline")
        step = int(saved.get("step", 0))
        if saved.get("torch_rng_state") is not None:
            torch.set_rng_state(saved["torch_rng_state"])
        print(f"resumed_from_step={step}", flush=True)

    started = time.perf_counter()
    process = __import__("psutil").Process()
    peak_rss = process.memory_info().rss
    metrics = []
    actions_per_question = cfg["generation"]["candidates_per_question"]
    if args.resume and cfg["training"]["checkpoint_every_steps"] % actions_per_question:
        raise ValueError("For exact resume, checkpoint_every_steps must be divisible by candidates_per_question")
    global_action_index = 0
    for epoch in range(cfg["training"]["epochs"]):
        progress = tqdm(train, desc=f"epoch {epoch+1}", unit="question")
        for row in progress:
            if global_action_index + actions_per_question <= step:
                global_action_index += actions_per_question
                continue
            actions = ref.sample_actions(row["question"], actions_per_question,
                                         cfg["generation"]["max_input_tokens"],
                                         cfg["generation"]["max_new_tokens"], True)
            retrieved = retriever.retrieve(row["question"], cfg["retrieval"]["top_k"],
                                            row["id"], cfg["retrieval"]["max_context_chars"])
            # The gold supporting context is explicit reward supervision; the BM25 passages
            # come only from training rows. Evaluation never uses these rewards.
            reward_contexts = [row["context"]] + [x["context"] for x in retrieved]
            first_action = max(0, step - global_action_index)
            for action in actions[first_action:]:
                action_started = time.perf_counter()
                scored_question = action["candidate"]
                answers = [qa.answer(scored_question, context)[0] for context in reward_contexts]
                answer_reward = max((reward(answer, row["answers"], valid=True)
                                     for answer in answers), default=0.0)
                preserves = entity_signature(row["question"]).issubset(
                    entity_signature(action["sampled_text"]))
                # Invalid text is scored using the documented original-question fallback,
                # then receives the invalid-action penalty. Its likelihood remains the
                # likelihood of action['token_ids'], the exact sequence sampled above.
                value = answer_reward + (0.1 if preserves else 0.0) - (0.2 if not action["valid"] else 0.0)
                logprob = ref.sequence_logprob_ids(
                    row["question"], action["token_ids"], cfg["generation"]["max_input_tokens"])
                loss, advantage = reinforce_step(logprob, value, baseline, optimizer,
                                                 ref.trainable,
                                                 cfg["training"]["gradient_clip_norm"])
                step += 1
                metrics.append({"epoch": epoch + 1, "step": step, "reward": value,
                                "answer_reward": answer_reward, "advantage": advantage,
                                "loss": loss, "valid": action["valid"],
                                "preserves_entities": preserves,
                                "action_seconds": time.perf_counter() - action_started,
                                "sampled_text": action["sampled_text"],
                                "scored_question": scored_question})
                peak_rss = max(peak_rss, process.memory_info().rss)
                interval = cfg["training"]["checkpoint_every_steps"]
                if interval > 0 and step % interval == 0:
                    save_checkpoint(checkpoint, ref, optimizer, baseline, step)
            global_action_index += actions_per_question
            elapsed = time.perf_counter() - started
            progress.set_postfix(step=step, reward=f"{metrics[-1]['reward']:.3f}",
                                 elapsed=f"{elapsed/60:.1f}m",
                                 rss=f"{process.memory_info().rss/1e9:.2f}GB")

    save_checkpoint(checkpoint, ref, optimizer, baseline, step)
    os.makedirs("outputs", exist_ok=True)
    elapsed = time.perf_counter() - started
    with open("outputs/train_metrics.json", "w", encoding="utf-8") as stream:
        json.dump({"trainable_parameters": ref.trainable_count, "steps": metrics,
                   "elapsed_seconds": elapsed,
                   "peak_rss_mb": peak_rss / 1e6,
                   "valid_generation_rate": sum(x["valid"] for x in metrics) / max(1, len(metrics)),
                   "mean_reward": sum(x["reward"] for x in metrics) / max(1, len(metrics))},
                  stream, ensure_ascii=False, indent=2)
    print(f"training_seconds={elapsed:.1f} peak_rss_mb={peak_rss/1e6:.1f} "
          f"valid_generation_rate={sum(x['valid'] for x in metrics)/max(1,len(metrics)):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
