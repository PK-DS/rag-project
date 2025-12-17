import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

from app.rag_pipeline import build_pipeline
from app.embeddings import embed_texts
from app.llm import generate_answer

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

index, meta = build_pipeline()

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    qv = embed_texts([req.question])
    _, idx = index.search(qv, 3)
    ctx = [meta[i] for i in idx[0]]

    answer = generate_answer(req.question, ctx)

    return {
        "answer": answer,
        "sources": [
            {"document": c["document"], "snippet": c["chunk"]}
            for c in ctx
        ]
    }
