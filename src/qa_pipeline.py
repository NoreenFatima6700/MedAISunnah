from transformers import pipeline
from src.vector_store import VectorStore
from src.knowledge_base import load_knowledge_base

class MedAISunnahQA:
    def __init__(self):
        docs = load_knowledge_base()

        self.vector_store = VectorStore()
        self.vector_store.build(docs)

        self.llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=300
        )

    def answer(self, question):
        retrieved = self.vector_store.search(question)

        context = ""
        citations = []

        for r in retrieved:
            context += f"{r['text']} ({r['source']})\n"
            citations.append(r["source"])

        prompt = f"""
Answer the question using ONLY the context below.

For each relevant source:
- Quote the exact text from the source.
- Briefly explain it in your own words.
- Mention the source after the quote.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm(prompt)[0]["generated_text"].strip()

        return {
            "answer": response,
            "citations": list(set(citations))
        }
