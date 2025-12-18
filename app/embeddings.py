import numpy as np
import google.generativeai as genai

def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = []

    for t in texts:
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=t
        )
        embeddings.append(res["embedding"])

    return np.array(embeddings, dtype="float32")
