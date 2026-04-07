import os
from typing import List
from dotenv import load_dotenv
from groq import Groq
from src.vectorstore.vectordb import VectorDB
from langchain_core.documents import Document

class GroqLLM:
    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        load_dotenv()

        api_key = os.getenv("key")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(self, query: str, context_docs: List[Document]) -> str:
        context = "\n\n".join(
            [
                f"(Page {doc.metadata.get('page')}): {doc.page_content}"
                for doc in context_docs
            ]
        )

        prompt = f"""
        You are an AI assistant answering questions based ONLY on the provided context.

        Context:
        {context}

        Question:
        {query}

        Answer:
        - If possible, list items clearly (e.g., bullet points)
        - Use the context to construct the best possible answer
        - Include page references when relevant
        """
        return prompt

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate answer using Groq LLM.
        """

        prompt = self._build_prompt(query, context_docs)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        """
                        You are a helpful AI assistant answering questions using the provided context.

                        Rules:
                        - Use the context as your primary source of truth
                        - If the context contains partial information, you may reasonably infer or complete the answer
                        - If the exact answer is not explicitly stated, combine relevant pieces from the context
                        - Only say "I don't know" if the context provides no useful information at all
                        - Be clear, concise, and helpful
                        - When possible, include references to page numbers
                        """
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    vectordb = VectorDB()
    vectordb.load()

    llm = GroqLLM()

    query = "What is overfitting in machine learning?"

    docs = vectordb.search(query, top_k=5)

    answer = llm.generate_answer(query, docs)

    print("\nAnswer:\n")
    print(answer)