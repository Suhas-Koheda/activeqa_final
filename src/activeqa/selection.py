def select_majority_vote(candidates, qa, contexts, on_qa_time=None):
    """Reference-free self-consistency selection.

    Answers every candidate on the shared context pool, groups candidates by
    their extracted answer, and prefers the answer with the widest support
    (number of agreeing candidates); ties break on span confidence. Unlike
    :func:`select_qa_score`, this can elect a rewrite whose answer differs from
    the original, but only when several rewrites agree on it.
    """
    from .rewards import token_f1
    from .preprocessing import normalize

    fallback_question = candidates[0] if candidates else ""
    fallback_answers = []
    for context in contexts:
        if hasattr(qa, "answer_with_score"):
            answer, elapsed, _ = qa.answer_with_score(fallback_question, context["context"])
        else:
            answer, elapsed = qa.answer(fallback_question, context["context"])
        if on_qa_time is not None:
            on_qa_time(elapsed)
        fallback_answers.append(answer)

    per_candidate = []  # (candidate, answer, confidence)
    for c in candidates:
        scores = []; answers = []
        for x in contexts:
            if hasattr(qa, "answer_with_score"):
                answer, elapsed, score = qa.answer_with_score(c, x["context"])
            else:
                answer, elapsed = qa.answer(c, x["context"])
                score = float(bool(answer.strip()))
            if on_qa_time is not None:
                on_qa_time(elapsed)
            scores.append(score); answers.append(answer)
        best = max(range(len(contexts)), key=lambda i: scores[i])
        per_candidate.append((c, answers[best], scores[best]))

    groups = {}  # normalized answer -> list of (candidate, confidence)
    for c, answer, score in per_candidate:
        groups.setdefault(normalize(answer), []).append((c, score))
    if not groups:
        return ""

    def key(entry):
        return (len(entry[1]), max(score for _, score in entry[1]))

    answer_key = max(groups.items(), key=key)[0]
    chosen = max(groups[answer_key], key=lambda t: t[1])[0]
    # A lone-vote answer that shares nothing with the original's answer is
    # treated as too risky to deploy; prefer the original in that case.
    if chosen != fallback_question and len(groups[answer_key]) == 1:
        consistency = token_f1(answers_per_candidate(per_candidate, chosen),
                               fallback_answers)
        if consistency == 0.0:
            chosen = fallback_question

    return chosen


def answers_per_candidate(per_candidate, chosen):
    return next(a for c, a, _ in per_candidate if c == chosen)


def select_qa_score(candidates, qa, contexts, on_qa_time=None):
    """Select a candidate using only model span confidence, never gold answers."""
    from .rewards import token_f1

    fallback_question = candidates[0] if candidates else ""
    fallback_answers = []
    for context in contexts:
        if hasattr(qa, "answer_with_score"):
            answer, elapsed, _ = qa.answer_with_score(fallback_question, context["context"])
        else:
            answer, elapsed = qa.answer(fallback_question, context["context"])
        if on_qa_time is not None:
            on_qa_time(elapsed)
        fallback_answers.append(answer)

    scored=[]
    for c in candidates:
        scores=[]; answers=[]
        for index, x in enumerate(contexts):
            if hasattr(qa, "answer_with_score"):
                answer, elapsed, score = qa.answer_with_score(c, x["context"])
            else:
                answer, elapsed = qa.answer(c, x["context"])
                score = float(bool(answer.strip()))
            if on_qa_time is not None:
                on_qa_time(elapsed)
            scores.append(score)
            answers.append(answer)
        # A high-confidence but answer-changing rewrite may be a semantic
        # hallucination. Without a Telugu semantic model, require its extracted
        # answers to remain reasonably consistent with the original question.
        consistency = sum(token_f1(a, [b]) for a, b in zip(answers, fallback_answers)) / max(1, len(contexts))
        if c != fallback_question and consistency < 0.5:
            continue
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
