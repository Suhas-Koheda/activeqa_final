def run_question(question,ref,qa,retriever,cfg):
    cs=ref.sample(question,cfg["generation"]["candidates_per_question"],cfg["generation"]["max_input_tokens"],cfg["generation"]["max_new_tokens"],False)
    selection_candidates=list(dict.fromkeys([question,*cs]))
    passages=retriever.retrieve(question,cfg["retrieval"]["top_k"],max_chars=cfg["retrieval"]["max_context_chars"])
    if not passages: return {"question":question,"candidates":cs,"answer":"","contexts":[]}
    from .selection import select_qa_score
    chosen=select_qa_score(selection_candidates,qa,passages)
    answers=[qa.answer_with_score(chosen,p["context"]) for p in passages]
    answer,_,confidence=max(answers,key=lambda item:item[2])
    return {"question":question,"candidates":cs,"selected_question":chosen,
            "answer":answer,"answer_confidence":confidence,"contexts":passages}
