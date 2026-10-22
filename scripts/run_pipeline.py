"""Answer one Telugu question with the trained ActiveQA reformulation policy."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "src")
from activeqa.data import load_splits
from activeqa.inference import run_question
from activeqa.qa_model import TeluguQA
from activeqa.reformulator import Reformulator
from activeqa.retrieval import Retriever
from activeqa.utils import load_config, setup_cpu, load_policy_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Telugu question to answer")
    parser.add_argument("--config", default="configs/colab_external_rl.yaml")
    parser.add_argument("--checkpoint", help="LoRA policy checkpoint (defaults to config)")
    parser.add_argument("--base-policy", action="store_true",
                        help="Skip loading the trained adapter")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_cpu(cfg["seed"], cfg["cpu_threads"], cfg.get("device", "cpu"))
    train, _evaluation = load_splits(cfg["data"])
    ref = Reformulator(cfg["models"]["reformulator"], cfg["models"]["cache_dir"],
                       cfg["models"].get("lora_rank"), cfg["models"].get("lora_alpha"),
                       device=cfg.get("device", "cpu"))
    checkpoint = args.checkpoint or cfg["training"].get("checkpoint")
    if not args.base_policy and checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"Trained adapter not found: {checkpoint}. Run the training workflow first "
                "or pass --base-policy to use the pretrained reformulator."
            )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("base_model") != ref.base_name:
            raise ValueError("Checkpoint base model does not match configured reformulator")
        load_policy_state(ref.model, saved)

    qa = TeluguQA(cfg["models"]["qa"], cfg["models"]["cache_dir"],
                  device=cfg.get("device", "cpu"))
    result = run_question(args.question, ref, qa, Retriever(train), cfg)
    result["policy_checkpoint"] = None if args.base_policy else checkpoint
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
