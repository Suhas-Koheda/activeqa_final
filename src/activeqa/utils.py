from __future__ import annotations
import os, random, time
from pathlib import Path
import numpy as np
import yaml

def load_config(path):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)

def setup_cpu(seed: int, threads: int, device: str = "cpu"):
    """Configure deterministic seeds and CPU threads, preserving optional CUDA use."""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    # Keep Hugging Face metadata caches within the writable project tree. Some
    # Transformers releases probe optional adapter files and cache their 404 refs.
    hf_home = Path(os.environ.get("HF_HOME", Path("data/hf_home").resolve()))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.set_num_threads(threads); torch.set_num_interop_threads(max(1, min(threads, 2)))
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is false. Enable a GPU runtime.")
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def elapsed_since(start): return round(time.perf_counter() - start, 3)

def ensure_parent(path): Path(path).parent.mkdir(parents=True, exist_ok=True)

def make_policy_optimizer(params, lr):
    """Memory efficient optimizer for full-model policy-gradient updates."""
    from transformers.optimization import Adafactor
    return Adafactor(params, lr=float(lr), scale_parameter=False,
                     relative_step=False, warmup_init=False)

def load_policy_state(model, saved):
    """Load current full-model policy checkpoints and legacy LoRA checkpoints."""
    if "model_state" in saved:
        model.load_state_dict(saved["model_state"])
        return
    if "adapter" in saved:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, saved["adapter"])
        return
    raise ValueError("Checkpoint has no model_state or legacy adapter weights")
