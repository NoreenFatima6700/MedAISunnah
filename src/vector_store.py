from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.kb = None
        self.embeddings = None
        self.index = None

    def build(self, docs: pd.DataFrame):
        """
        Build the vector store from a DataFrame with 'text' and 'source'.
        """
        self.kb = docs
        texts = self.kb["text"].tolist()
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        print(f"Vector store built with {len(texts)} documents.")

    def search(self, query: str):
        """
        Return ONLY the most relevant document.
        """
        query_vec = self.model.encode([query], convert_to_numpy=True)

        D, I = self.index.search(query_vec, 1)
        best_index = I[0][0]

        return [{
            "text": self.kb.iloc[best_index]["text"],
            "source": self.kb.iloc[best_index]["source"]
        }]
