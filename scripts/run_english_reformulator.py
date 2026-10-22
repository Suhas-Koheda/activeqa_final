"""Generate three English question reformulations with the trained SearchQA adapter."""
import argparse, json, sys
sys.path.insert(0,"src")
from activeqa.utils import load_config, setup_cpu
from train_searchqa_rl import EnglishReformulator

def main():
    p=argparse.ArgumentParser(); p.add_argument("--question",required=True); p.add_argument("--config",default="configs/colab_english_activeqa.yaml"); p.add_argument("--checkpoint",default="outputs/english_searchqa_rl_policy.pt"); a=p.parse_args()
    cfg=load_config(a.config); setup_cpu(cfg["seed"],cfg["cpu_threads"],cfg.get("device","cuda")); m=cfg["models"]
    ref=EnglishReformulator(m["reformulator"],m["cache_dir"],cfg.get("device","cuda"))
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if state.get("base_model") != ref.base_name:
        raise ValueError("RL checkpoint uses a different English base model")
    ref.model.load_state_dict(state["model_state"]); ref.model.eval()
    acts=ref.sample_actions(a.question,3,int(cfg["generation"]["max_input_tokens"]),int(cfg["generation"]["max_new_tokens"]))
    print(json.dumps({"question":a.question,"reformulations":[x["candidate"] for x in acts],"valid":[x["valid"] for x in acts],"checkpoint":a.checkpoint},indent=2))
if __name__=="__main__": main()
