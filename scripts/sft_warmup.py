"""LoRA supervised warm-up on the uploaded Telugu reformulation train pairs."""
import argparse
import json
import os
import random
import sys
import time

import torch
import psutil
from tqdm import tqdm

sys.path.insert(0, "src")
from activeqa.data import load_reformulation_splits
from activeqa.policy_gradient import sft_batch_step
from activeqa.reformulator import Reformulator
from activeqa.utils import load_config, setup_cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colab_external_rl.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_cpu(cfg["seed"], cfg["cpu_threads"], cfg.get("device", "cpu"))
    training = cfg["training"]
    if not training.get("sft_warmup", False):
        print("sft_warmup=false; skipping reformulation-pair warm-up", flush=True)
        return

    train_pairs, dev_pairs = load_reformulation_splits(cfg["data"])
    if not train_pairs or not dev_pairs:
        raise RuntimeError("Reformulation train and dev splits must both be non-empty")

    ref = Reformulator(cfg["models"]["reformulator"], cfg["models"]["cache_dir"],
                       cfg["models"]["lora_rank"], cfg["models"]["lora_alpha"],
                       device=cfg.get("device", "cpu"))
    params = ref.trainable
    optimizer = torch.optim.AdamW(
        params, lr=float(training.get("sft_learning_rate", 5e-6)))
    checkpoint = training.get("sft_checkpoint", "outputs/external_rl_sft_policy.pt")
    checkpoint_every = max(1, int(training.get("sft_checkpoint_every_steps", 25)))
    start_step = 0
    losses = []

    if args.resume and os.path.isfile(checkpoint):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("base_model") != ref.base_name:
            raise ValueError("SFT checkpoint base model does not match configured reformulator")
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(ref.model, saved["adapter"])
        if saved.get("complete"):
            print(f"SFT warm-up already complete: {checkpoint}", flush=True)
            return
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved.get("step", 0))
        losses = saved.get("losses", [])
        print(f"resuming_sft_step={start_step}", flush=True)

    epochs = max(1, int(training.get("sft_warmup_epochs", 1)))
    max_input = int(cfg["generation"]["max_input_tokens"])
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)

    def save(step, complete=False):
        from peft import get_peft_model_state_dict
        torch.save({"adapter": get_peft_model_state_dict(ref.model),
                    "optimizer": optimizer.state_dict(), "step": step,
                    "complete": complete, "base_model": ref.base_name,
                    "losses": losses, "torch_rng_state": torch.get_rng_state()}, checkpoint)

    step = 0
    for epoch in range(epochs):
        order = list(range(len(train_pairs)))
        random.Random(int(cfg["seed"]) + epoch).shuffle(order)
        progress = tqdm(order, desc=f"reformulation SFT {epoch + 1}/{epochs}",
                        unit="pair", dynamic_ncols=True)
        for row_index in progress:
            if step < start_step:
                step += 1
                continue
            row = train_pairs[row_index]
            # Average by target token count so long targets do not receive a
            # larger update merely because their summed log likelihood is more
            # negative.
            logprob = ref.sequence_mean_logprob(
                row["question"], row["reformulation"], max_input=max_input,
                max_new=int(cfg["generation"]["max_new_tokens"]))
            loss = sft_batch_step([logprob], optimizer, params,
                                  float(training.get("gradient_clip_norm", 1.0)))
            losses.append(loss)
            step += 1
            if step % checkpoint_every == 0:
                save(step)
            peak_rss = max(peak_rss, process.memory_info().rss)
            recent = losses[-min(50, len(losses)):]
            progress.set_postfix(loss=f"{sum(recent) / len(recent):.3f}",
                                 elapsed=f"{(time.perf_counter() - started) / 60:.1f}m")

    save(step, complete=True)
    summary = {"method": "teacher_forced_silver_reformulation_warmup",
               "train_pairs": len(train_pairs), "heldout_pairs": len(dev_pairs),
               "epochs": epochs, "optimizer_steps": len(losses),
               "mean_loss": sum(losses) / max(1, len(losses)),
               "elapsed_seconds": time.perf_counter() - started,
               "peak_rss_mb": peak_rss / 1e6,
               "checkpoint": checkpoint,
               "semantic_labels_human_verified": False}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sft_warmup_metrics.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
