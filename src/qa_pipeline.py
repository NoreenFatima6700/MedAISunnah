from transformers import pipeline
from src.knowledge_base import load_knowledge_base
from src.vector_store import VectorStore

class MedAISunnahQA:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_length=300
        )

        docs = load_knowledge_base()
        self.vector_store.build(docs)

    def answer(self, question):
        retrieved = self.vector_store.search(question)

        context = ""
        citations = []
        for r in retrieved:
            context += f"- {r['text']} ({r['source']})\n"
            citations.append(r["source"])

        prompt = f"""
Answer the question using ONLY the context below.
Cite sources clearly.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm(prompt)[0]["generated_text"]

        return {
            "answer": response,
            "citations": list(set(citations))
        }
