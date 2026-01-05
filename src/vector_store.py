from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from src.knowledge_base import load_knowledge_base

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
    # Convert question into vector
    query_vec = self.model.encode([query], convert_to_numpy=True)

    # Ask FAISS for the single best match
    D, I = self.index.search(query_vec, 1)

    # Get index of best document
    best_index = I[0][0]

    # Return only one result
    return [{
        "text": self.kb.iloc[best_index]["text"],
        "source": self.kb.iloc[best_index]["source"]
    }]
