def answer(self, question):
    retrieved = self.vector_store.search(question)

    context = ""
    citations = []

    for r in retrieved:
        context += f"{r['text']} ({r['source']})\n"
        citations.append(r["source"])

    # ✅ UPDATED PROMPT GOES HERE
    prompt = f"""
Answer the question using ONLY the context below.

For each relevant source:
1. First QUOTE the exact text as it appears.
2. Then briefly explain it in your own words.
3. Mention the source explicitly.

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
