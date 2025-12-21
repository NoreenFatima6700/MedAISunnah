from transformers import pipeline
from src.vector_store import VectorStore
from src.knowledge_base import load_knowledge_base

class MedAISunnahQA:
    def __init__(self):
        # Load knowledge base
        docs = load_knowledge_base()

        # Build vector store
        self.vector_store = VectorStore()
        self.vector_store.build(docs)

        # Initialize local LLM (can replace with other offline models if needed)
        self.llm = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_length=300
        )

    def answer(self, question: str):
        # Retrieve top 5 relevant documents
        retrieved = self.vector_store.search(question, top_k=5)

        # Combine retrieved texts for context
        context = ""
        citations = []
        for r in retrieved:
            context += f"- {r['text']} ({r['source']})\n"
            citations.append(r["source"])

        # Prompt to LLM
        prompt = f"""
Answer the question using ONLY the context below.
Cite sources clearly.

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate answer
        response = self.llm(prompt)[0]["generated_text"]

        return {
            "answer": response,
            "citations": list(set(citations))
        }
