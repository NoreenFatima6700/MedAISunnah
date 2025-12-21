from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.sources = []

    def build(self, documents):
        self.texts = documents["text"].tolist()
        self.sources = documents["source"].tolist()

        embeddings = self.model.encode(self.texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        q_embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(q_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.texts[idx],
                "source": self.sources[idx]
            })
        return results
