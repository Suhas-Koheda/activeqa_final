import argparse
import json
import os
import sys
import torch

sys.path.insert(0, "src")
from activeqa.data import load_splits
from activeqa.evaluation import evaluate
from activeqa.qa_model import TeluguQA
from activeqa.reformulator import Reformulator
from activeqa.retrieval import Retriever
from activeqa.utils import load_config, setup_cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_cpu(cfg["seed"], cfg["cpu_threads"], cfg.get("device", "cpu"))
    train, heldout, retrieval_corpus = load_splits(cfg["data"], include_full_train=True)
    ref = Reformulator(cfg["models"]["reformulator"], cfg["models"]["cache_dir"],
                       cfg["models"]["lora_rank"], cfg["models"]["lora_alpha"],
                       device=cfg.get("device", "cpu"))
    qa = TeluguQA(cfg["models"]["qa"], cfg["models"]["cache_dir"], device=cfg.get("device", "cpu"))
    retriever = Retriever(retrieval_corpus)

    baseline_path = "outputs/eval_pretrained.json"
    torch.manual_seed(cfg["seed"])
    pretrained = evaluate(heldout, ref, qa, retriever, cfg, baseline_path)
    combined = {"pretrained": pretrained}
    if args.checkpoint:
        from peft import set_peft_model_state_dict
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("base_model") != ref.base_name:
            raise ValueError("Checkpoint base model does not match configured reformulator")
        set_peft_model_state_dict(ref.model, saved["adapter"])
        torch.manual_seed(cfg["seed"])
        trained = evaluate(heldout, ref, qa, retriever, cfg, "outputs/eval_trained.json")
        combined["reinforce_trained"] = trained
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_metrics.json", "w", encoding="utf-8") as stream:
        json.dump(combined, stream, ensure_ascii=False, indent=2)
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
