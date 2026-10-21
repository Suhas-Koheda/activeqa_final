import argparse
import json
import os
import sys
import torch

sys.path.insert(0, "src")
from activeqa.data import load_reformulation_splits, load_splits
from activeqa.evaluation import evaluate, evaluate_reformulation_pairs
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
    _rewrite_train, rewrite_dev = load_reformulation_splits(cfg["data"])
    ref = Reformulator(cfg["models"]["reformulator"], cfg["models"]["cache_dir"],
                       cfg["models"]["lora_rank"], cfg["models"]["lora_alpha"],
                       device=cfg.get("device", "cpu"))
    qa = TeluguQA(cfg["models"]["qa"], cfg["models"]["cache_dir"], device=cfg.get("device", "cpu"))
    retriever = Retriever(retrieval_corpus)

    baseline_path = "outputs/eval_pretrained.json"
    torch.manual_seed(cfg["seed"])
    pretrained = evaluate(heldout, ref, qa, retriever, cfg, baseline_path, label="pretrained")
    combined = {"pretrained": pretrained}
    pair_metrics = {
        "pretrained": evaluate_reformulation_pairs(
            rewrite_dev, ref, qa, cfg, "outputs/reformulations_pretrained.json",
            label="reformulation_dev_pretrained")
    }
    sft_checkpoint = cfg["training"].get("sft_checkpoint")
    if sft_checkpoint and os.path.isfile(sft_checkpoint):
        from peft import set_peft_model_state_dict
        sft_saved = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
        if sft_saved.get("base_model") != ref.base_name or not sft_saved.get("complete"):
            raise ValueError("SFT warm-up checkpoint is incomplete or uses a different base model")
        set_peft_model_state_dict(ref.model, sft_saved["adapter"])
        torch.manual_seed(cfg["seed"])
        sft_qa = evaluate(heldout, ref, qa, retriever, cfg,
                          "outputs/eval_sft_warmup.json", label="sft_warmup")
        combined["sft_warmup"] = sft_qa
        torch.manual_seed(cfg["seed"])
        pair_metrics["sft_warmup"] = evaluate_reformulation_pairs(
            rewrite_dev, ref, qa, cfg, "outputs/reformulations_sft_warmup.json",
            label="reformulation_dev_sft_warmup")
    if args.checkpoint:
        from peft import set_peft_model_state_dict
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("base_model") != ref.base_name:
            raise ValueError("Checkpoint base model does not match configured reformulator")
        set_peft_model_state_dict(ref.model, saved["adapter"])
        torch.manual_seed(cfg["seed"])
        trained = evaluate(heldout, ref, qa, retriever, cfg, "outputs/eval_trained.json", label="reinforce_trained")
        combined["reinforce_trained"] = trained
        pair_metrics["trained"] = evaluate_reformulation_pairs(
            rewrite_dev, ref, qa, cfg, "outputs/reformulations_trained.json",
            label="reformulation_dev_trained")
    combined["reformulation_pair_eval"] = pair_metrics
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_metrics.json", "w", encoding="utf-8") as stream:
        json.dump(combined, stream, ensure_ascii=False, indent=2)
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
