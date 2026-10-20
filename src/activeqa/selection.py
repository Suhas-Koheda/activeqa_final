def select_qa_score(candidates, qa, contexts):
    scored=[]
    for c in candidates:
        answers=[qa.answer(c,x["context"])[0] for x in contexts]
        # Inference-time score is extractive span confidence proxy; no gold answers used.
        score=sum(len(a.strip())>0 for a in answers)/max(1,len(answers))
        scored.append((score,c))
    return max(scored,key=lambda x:x[0])[1] if scored else ""

def select_oracle(candidates,qa,contexts,references):
    from .rewards import token_f1
    return max(candidates,key=lambda c:max((token_f1(qa.answer(c,x["context"])[0],references) for x in contexts),default=0))
