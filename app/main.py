import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from app.rag_pipeline import build_pipeline

load_dotenv()

app = FastAPI(title="RAG API")

rag = build_pipeline(
    data_dir="data/documents",
    top_k=3
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        answer, contexts = rag.answer(req.question)
        return {
            "answer": answer,
            "sources": [
                {
                    "document": c["document"],
                    "snippet": c["chunk"][:300]
                }
                for c in contexts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
