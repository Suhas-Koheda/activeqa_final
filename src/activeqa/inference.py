def run_question(question,ref,qa,retriever,cfg):
    cs=ref.sample(question,cfg["generation"]["candidates_per_question"],cfg["generation"]["max_input_tokens"],cfg["generation"]["max_new_tokens"],False)
    passages=retriever.retrieve(question,cfg["retrieval"]["top_k"],max_chars=cfg["retrieval"]["max_context_chars"])
    if not passages: return {"question":question,"candidates":cs,"answer":"","contexts":[]}
    from .selection import select_qa_score
    chosen=select_qa_score(cs,qa,passages)
    answers=[qa.answer(chosen,p["context"])[0] for p in passages]
    return {"question":question,"candidates":cs,"selected_question":chosen,"answer":max(answers,key=len),"contexts":passages}
