"""Train an English question reformulator with QA-reward REINFORCE on SearchQA."""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, "src")
from activeqa.qa_model import TeluguQA
from activeqa.policy_gradient import reinforce_batch_step
from activeqa.preprocessing import normalize, tokens
from activeqa.rewards import reward
from activeqa.utils import load_config, setup_cpu, make_policy_optimizer


class EnglishReformulator:
    """T5 paraphrase policy with LoRA; stores exact generated tokens for REINFORCE."""
    def __init__(self, name, cache_dir, device):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=cache_dir).to(device)
        self.device = torch.device(device)
        self.base_name = name
        if hasattr(self.model,"gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache=False
        self.model.train()
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]

    def encode(self, question, max_input):
        return {k:v.to(self.device) for k,v in self.tokenizer(
            "paraphrase: " + normalize(question), return_tensors="pt", truncation=True,
            max_length=max_input).items() if k in {"input_ids", "attention_mask"}}

    def sample_actions(self, question, n, max_input, max_new):
        src=self.encode(question,max_input)
        was_training=self.model.training; old_cache=self.model.config.use_cache
        self.model.eval(); self.model.config.use_cache=True
        if hasattr(self.model,"gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        with torch.inference_mode():
            seq=self.model.generate(**src, max_new_tokens=max_new, do_sample=True,
                top_p=.9, temperature=.9, num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3)
        self.model.config.use_cache=old_cache
        if hasattr(self.model,"gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.train(was_training)
        actions=[]
        for ids in seq:
            text=normalize(self.tokenizer.decode(ids,skip_special_tokens=True))
            valid=valid_rewrite(question,text, self.tokenizer.eos_token_id in ids.tolist())
            policy_ids=ids.tolist()
            if policy_ids and policy_ids[0] == self.model.config.decoder_start_token_id:
                policy_ids=policy_ids[1:]
            actions.append({"candidate":text if valid else question,"sampled_text":text,
                            "valid":valid,"token_ids":policy_ids})
        return actions

    def score_actions(self, question, token_ids):
        src=self.encode(question,96); n=len(token_ids)
        src={k:v.expand(n,-1) for k,v in src.items()}
        width=max(map(len,token_ids)); labels=torch.full((n,width),-100,dtype=torch.long,device=self.device)
        mask=torch.zeros((n,width),dtype=torch.float32,device=self.device)
        for i,ids in enumerate(token_ids):
            labels[i,:len(ids)]=torch.tensor(ids,device=self.device); mask[i,:len(ids)]=1
        was_training=self.model.training; self.model.eval()
        try:
            logits=self.model(**src,labels=labels).logits
        finally: self.model.train(was_training)
        selected=torch.log_softmax(logits,dim=-1).gather(-1,labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (selected*mask).sum(dim=1)


_TYPE = {
    "who":r"\bwho(m|se)?\b", "when":r"\bwhen\b", "where":r"\bwhere\b",
    "why":r"\bwhy\b", "how":r"\bhow\b", "which":r"\bwhich\b",
    "what":r"\bwhat\b", "how_many":r"\bhow many\b", "how_much":r"\bhow much\b",
}
def _types(text):
    low=text.casefold(); return {k for k,p in _TYPE.items() if re.search(p,low)}

def valid_rewrite(source, target, terminated=True):
    if not terminated or len(target)<8 or len(target)>max(300,4*len(source)) or target==source: return False
    sw=set(tokens(source)); tw=set(tokens(target))
    if len(sw & tw)/max(1,len(sw)) < .25: return False
    if not _types(source).issubset(_types(target)): return False
    nums=set(re.findall(r"\d+(?:[./:-]\d+)*",source))
    if not nums.issubset(set(re.findall(r"\d+(?:[./:-]\d+)*",target))): return False
    return True

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/colab_english_activeqa.yaml"); a=ap.parse_args()
    cfg=load_config(a.config); setup_cpu(cfg["seed"],cfg["cpu_threads"],cfg.get("device","cuda"))
    ds=load_dataset(cfg["data"]["dataset"],split="train",cache_dir=cfg["data"].get("cache_dir"))
    # Reproducible random split because this public mirror publishes one combined MRQA split.
    ds=ds.shuffle(seed=cfg["seed"])
    ev_n=min(int(cfg["data"]["eval_examples"]),max(1,len(ds)//10))
    train_ds=ds.select(range(ev_n,len(ds))); eval_ds=ds.select(range(ev_n))
    if cfg["data"].get("train_examples"): train_ds=train_ds.select(range(min(len(train_ds),int(cfg["data"]["train_examples"]))))
    print(f"dataset={cfg['data']['dataset']} train={len(train_ds)} eval={len(eval_ds)} split=seeded_holdout_from_combined_MRQA")
    def row(x):
        ans=x.get("answers",[]); ans=ans.get("text",[]) if isinstance(ans,dict) else ans
        return {"id":str(x.get("id", "")),"question":normalize(x.get("question")),
                "context":normalize(x.get("context")),"answers":[normalize(y) for y in ans if normalize(y)]}
    train=[r for r in (row(x) for x in train_ds) if r["question"] and r["context"] and r["answers"]]
    evaluation=[r for r in (row(x) for x in eval_ds) if r["question"] and r["context"] and r["answers"]]
    device=cfg.get("device","cuda"); models=cfg["models"]; gen=cfg["generation"]; trcfg=cfg["training"]
    policy=EnglishReformulator(models["reformulator"],models["cache_dir"],device)
    qa=TeluguQA(models["qa"],models["cache_dir"],device=device)
    opt=make_policy_optimizer(policy.trainable,trcfg["learning_rate"])
    checkpoint=Path(trcfg["checkpoint"]); checkpoint.parent.mkdir(parents=True,exist_ok=True)
    outdir=Path(trcfg.get("model_dir",str(checkpoint.with_suffix(""))))
    actions_per=int(gen["candidates_per_question"]); losses=[]; records=[]; step=0; started=time.time()
    pbar=tqdm(train,desc="SearchQA REINFORCE",unit="q")
    for row_i,r in enumerate(pbar):
        acts=policy.sample_actions(r["question"],actions_per,int(gen["max_input_tokens"]),int(gen["max_new_tokens"]))
        questions=[r["question"]]+[x["candidate"] for x in acts]
        answers=qa.answer_batch(questions,r["context"])
        rs=[reward(ans[0],r["answers"],valid=True) for ans in answers]
        baseline=rs[0]; rew=[reward(answers[i+1][0],r["answers"],valid=x["valid"]) for i,x in enumerate(acts)]
        adv=[v-baseline for v in rew]
        logp=policy.score_actions(r["question"],[x["token_ids"] for x in acts])
        loss=reinforce_batch_step(list(logp.unbind()),adv,opt,policy.trainable,float(trcfg["gradient_clip_norm"]))
        losses.append(loss); step+=1
        records.append({"id":r["id"],"question":r["question"],"answer":r["answers"][0],
                        "baseline_reward":baseline,"candidates":[x["candidate"] for x in acts],"rewards":rew})
        if step%100==0:
            torch.save({"model_state":policy.model.state_dict(),"base_model":policy.base_name,"step":step},checkpoint)
        pbar.set_postfix(mean_adv=f"{sum(adv)/len(adv):.3f}",valid=f"{sum(x['valid'] for x in acts)/len(acts):.2f}")
    policy.model.save_pretrained(outdir)
    torch.save({"model_state":policy.model.state_dict(),"base_model":policy.base_name,"step":step},checkpoint)
    metrics={"method":"REINFORCE","dataset":cfg["data"]["dataset"],"train_examples":len(train),"eval_examples":len(evaluation),
             "seeded_holdout":True,"steps":step,"mean_loss":sum(losses)/max(1,len(losses)),"seconds":time.time()-started,
             "checkpoint":str(checkpoint),"full_model_export":str(outdir),"qa_model":models["qa"],"reformulator_base":models["reformulator"]}
    Path(cfg["outputs"]).mkdir(parents=True,exist_ok=True)
    Path(cfg["outputs"],"train_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    with open(Path(cfg["outputs"],"sampled_rewrites.jsonl"),"w",encoding="utf-8") as f:
        for x in records[:1000]: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    # Fast held-out sanity evaluation: QA F1 reward on originals versus three
    # sampled, guarded reformulations for the first configured eval examples.
    eval_records=[]; original_rewards=[]; rewrite_rewards=[]; valid_count=0; total=0
    for r in tqdm(evaluation[:min(100,len(evaluation))],desc="SearchQA held-out",unit="q"):
        acts=policy.sample_actions(r["question"],actions_per,int(gen["max_input_tokens"]),int(gen["max_new_tokens"]))
        qs=[r["question"]]+[x["candidate"] for x in acts]
        ans=qa.answer_batch(qs,r["context"])
        original_rewards.append(reward(ans[0][0],r["answers"]))
        rr=[reward(ans[i+1][0],r["answers"],valid=a["valid"]) for i,a in enumerate(acts)]
        rewrite_rewards.append(max(rr)); valid_count+=sum(a["valid"] for a in acts); total+=len(acts)
        eval_records.append({"id":r["id"],"question":r["question"],"answers":r["answers"],
                             "candidates":[a["candidate"] for a in acts],"original_reward":original_rewards[-1],
                             "best_rewrite_reward":rewrite_rewards[-1]})
    metrics.update({"heldout_scored_examples":len(eval_records),
                    "original_mean_reward":sum(original_rewards)/max(1,len(original_rewards)),
                    "best_of_n_rewrite_mean_reward":sum(rewrite_rewards)/max(1,len(rewrite_rewards)),
                    "valid_generation_rate":valid_count/max(1,total)})
    with open(Path(cfg["outputs"],"heldout_reformulations.jsonl"),"w",encoding="utf-8") as f:
        for x in eval_records: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    Path(cfg["outputs"],"train_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
