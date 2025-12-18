import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings: np.ndarray, metadata: list):
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be numpy array")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.metadata = metadata
        self.embeddings = embeddings

    def search(self, query_embedding: np.ndarray, k: int):
        _, idx = self.index.search(query_embedding, k)
        return [self.metadata[i] for i in idx[0]]
