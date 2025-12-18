import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from pypdf import PdfReader
import docx
import google.generativeai as genai

from app.embeddings import embed_texts
from app.llm import generate_answer
from app.vector_store import VectorStore


# =====================
# RAG PIPELINE CLASS
# =====================
class RAGPipeline:
    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        self.vs = vector_store
        self.top_k = top_k

    def answer(self, question: str):
        # embed query
        q_embedding = embed_texts([question])

        # retrieve contexts
        contexts = self.vs.search(q_embedding, self.top_k)

        # generate answer
        answer = generate_answer(question, contexts)

        return answer, contexts


# =====================
# LOAD DOCUMENTS
# =====================
def load_documents(path: str):
    docs = []

    for f in Path(path).glob("*"):
        text = ""

        if f.suffix.lower() == ".pdf":
            reader = PdfReader(f)
            text = "\n".join(
                p.extract_text() for p in reader.pages if p.extract_text()
            )

        elif f.suffix.lower() == ".docx":
            d = docx.Document(f)
            text = "\n".join(p.text for p in d.paragraphs)

        elif f.suffix.lower() == ".txt":
            text = f.read_text()

        if text.strip():
            docs.append({
                "text": text,
                "source": f.name
            })

    return docs


# =====================
# CHUNKING
# =====================
def chunk_text(text: str, size: int = 500, overlap: int = 100):
    chunks = []
    i = 0

    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap

    return chunks


# =====================
# PIPELINE BUILDER
# =====================
def build_pipeline(data_dir: str, top_k: int = 3):
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    docs = load_documents(data_dir)
    if not docs:
        raise RuntimeError("No documents found in data directory")

    chunks = []
    meta = []

    for d in docs:
        for c in chunk_text(d["text"]):
            chunks.append(c)
            meta.append({
                "document": d["source"],
                "chunk": c
            })

    embeddings = embed_texts(chunks)

    vector_store = VectorStore(embeddings, meta)
    return RAGPipeline(vector_store, top_k)
