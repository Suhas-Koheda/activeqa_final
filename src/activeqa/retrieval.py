from rank_bm25 import BM25Okapi
from .preprocessing import tokens

class Retriever:
    def __init__(self, rows):
        self.rows=list(rows); self.bm25=BM25Okapi([tokens(x["context"]) or ["<empty>"] for x in self.rows])
    def retrieve(self, query, top_k=3, exclude_id=None, max_chars=1800):
        scores=self.bm25.get_scores(tokens(query) or ["<empty>"])
        ids=sorted(range(len(scores)),key=lambda i:(scores[i],-i),reverse=True)
        out=[]
        for i in ids:
            row=self.rows[i]
            if exclude_id is not None and row.get("id")==exclude_id: continue
            if not row["context"]: continue
            out.append({"id":row.get("id"),"context":row["context"][:max_chars],"score":float(scores[i])})
            if len(out)>=top_k: break
        return out
