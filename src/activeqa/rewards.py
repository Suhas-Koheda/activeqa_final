from .preprocessing import tokens

def token_f1(prediction, references):
    p=tokens(prediction); best=0.0
    for ref in references or [""]:
        g=tokens(ref)
        if not p and not g: score=1.
        elif not p or not g: score=0.
        else:
            from collections import Counter
            overlap=sum((Counter(p)&Counter(g)).values()); precision=overlap/len(p); recall=overlap/len(g)
            score=2*precision*recall/(precision+recall) if precision+recall else 0.
        best=max(best,score)
    return best

def reward(prediction,references,valid=True,preserve=1.0,em_weight=.1,preserve_weight=.1,invalid_penalty=.2):
    f1=token_f1(prediction,references); em=float(any(" ".join(tokens(prediction))==" ".join(tokens(x)) for x in references))
    return f1+em_weight*em+preserve_weight*preserve-(0 if valid else invalid_penalty)
