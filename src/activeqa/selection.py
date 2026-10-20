def select_qa_score(candidates, qa, contexts, on_qa_time=None):
    """Select a candidate using only model span confidence, never gold answers."""
    scored=[]
    for c in candidates:
        scores=[]
        for x in contexts:
            if hasattr(qa, "answer_with_score"):
                _, elapsed, score = qa.answer_with_score(c, x["context"])
            else:
                answer, elapsed = qa.answer(c, x["context"])
                score = float(bool(answer.strip()))
            if on_qa_time is not None:
                on_qa_time(elapsed)
            scores.append(score)
        scored.append((sum(scores)/max(1,len(scores)),c))
    return max(scored,key=lambda x:x[0])[1] if scored else ""

def select_oracle(candidates,qa,contexts,references,on_qa_time=None):
    from .rewards import token_f1
    def score_candidate(candidate):
        scores=[]
        for context in contexts:
            answer, elapsed = qa.answer(candidate, context["context"])
            if on_qa_time is not None:
                on_qa_time(elapsed)
            scores.append(token_f1(answer,references))
        return max(scores,default=0)
    return max(candidates,key=score_candidate)
