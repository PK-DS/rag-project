import os
from dotenv import load_dotenv
import google.generativeai as genai

def generate_answer(question: str, contexts: list[dict]) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    context_text = "\n\n".join(c["chunk"] for c in contexts)

    prompt = f"""
You are a strict RAG system.

RULES:
- Answer ONLY using the provided context.
- If the answer is not present, say exactly: Not found.
- Do NOT use external knowledge.
- Do NOT hallucinate.

Context:
{context_text}

Question:
{question}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()
