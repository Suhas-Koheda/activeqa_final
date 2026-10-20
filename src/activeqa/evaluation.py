from __future__ import annotations
import json, os, time, psutil
from .rewards import token_f1, reward
from .selection import select_qa_score,select_oracle
from .preprocessing import entity_signature

def evaluate(rows,reformulator,qa,retriever,cfg,outfile):
    proc=psutil.Process(); ram0=proc.memory_info().rss; peak_rss=ram0; start=time.perf_counter(); preds=[]
    flds={k:[] for k in ("original","pretrained","qa_selected","oracle")}; reward_values={k:[] for k in flds}; valid=preserve=harmful=0; ncan=0; qa_sec=0.
    for row in rows:
        peak_rss=max(peak_rss,proc.memory_info().rss)
        q=row["question"]
        actions=reformulator.sample_actions(q,cfg["generation"]["candidates_per_question"],cfg["generation"]["max_input_tokens"],cfg["generation"]["max_new_tokens"],False)
        cand=[x["candidate"] for x in actions]
        contexts=retriever.retrieve(q,cfg["retrieval"]["top_k"],row["id"],cfg["retrieval"]["max_context_chars"]) or [{"context":row["context"],"id":row["id"]}]
        def qa_best(question):
            nonlocal qa_sec
            aa=[]
            for ctx in contexts:
                a,t=qa.answer(question,ctx["context"]); qa_sec+=t; aa.append(a)
            return max(aa,key=lambda a:token_f1(a,row["answers"]))
        original=qa_best(q); base=cand[0] if cand else q; prep=qa_best(base)
        qsel=select_qa_score(cand,qa,contexts) if cand else q; qa_selected=qa_best(qsel)
        oracle=select_oracle(cand,qa,contexts,row["answers"]) if cand else q; oracle_ans=qa_best(oracle)
        selected_index=next((i for i,c in enumerate(cand) if c==qsel),0)
        oracle_index=next((i for i,c in enumerate(cand) if c==oracle),0)
        scored=[(original,True,True),(prep,actions[0]["valid"],entity_signature(q).issubset(entity_signature(actions[0]["sampled_text"]))),(qa_selected,actions[selected_index]["valid"],entity_signature(q).issubset(entity_signature(actions[selected_index]["sampled_text"]))),(oracle_ans,actions[oracle_index]["valid"],entity_signature(q).issubset(entity_signature(actions[oracle_index]["sampled_text"])))]
        for key,(answer,is_valid,is_preserved) in zip(flds,scored):
            flds[key].append(token_f1(answer,row["answers"]))
            reward_values[key].append(reward(answer,row["answers"],valid=is_valid,preserve=float(is_preserved)))
        for action in actions:
            ncan+=1; valid+=int(action["valid"])
            preserve+=int(entity_signature(q).issubset(entity_signature(action["sampled_text"])))
        sc=token_f1(prep,row["answers"]); harmful+=int(sc<token_f1(original,row["answers"]))
        preds.append({"id":row["id"],"question":q,"sampled_candidates":[x["sampled_text"] for x in actions],"candidates":cand,"candidate_validity":[x["valid"] for x in actions],"original_answer":original,"pretrained_answer":prep,"qa_selected_question":qsel,"qa_selected_answer":qa_selected,"oracle_answer":oracle_ans})
        peak_rss=max(peak_rss,proc.memory_info().rss)
    metrics={"examples":len(rows),"metrics":{k:{"exact_match":sum(x==1 for x in v)/max(1,len(v)),"f1":sum(v)/max(1,len(v)),"mean_reward":sum(reward_values[k])/max(1,len(reward_values[k]))} for k,v in flds.items()},"valid_generation_rate":valid/max(1,ncan),"entity_signature_preservation_rate":preserve/max(1,ncan),"harmful_reformulation_rate":harmful/max(1,len(rows)),"qa_seconds":round(qa_sec,2),"wall_seconds":round(time.perf_counter()-start,2),"peak_rss_mb":round(peak_rss/1e6,1)}
    os.makedirs(os.path.dirname(outfile) or ".",exist_ok=True)
    with open(outfile,"w",encoding="utf-8") as f: json.dump({"metrics":metrics,"predictions":preds},f,ensure_ascii=False,indent=2)
    return metrics
