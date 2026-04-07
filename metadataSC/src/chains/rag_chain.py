from typing import Dict, Any

from src.vectorstore.vectordb import VectorDB
from src.llm.llm import GroqLLM


class RAGPipeline:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

        # Initialize components
        self.vectordb = VectorDB()
        self.vectordb.load()

        self.llm = GroqLLM()

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Process user query through full RAG pipeline.
        """

        # Step 1: Retrieve relevant chunks
        docs = self.vectordb.search(query, top_k=self.top_k)

        # Step 2: Generate answer using LLM
        answer = self.llm.generate_answer(query, docs)

        # Step 3: Return structured response
        return {
            "answer": answer,
            "sources": docs,
        }

if __name__ == "__main__":
    rag = RAGPipeline(top_k=5)

    query = "Explain overfitting in simple terms"

    result = rag.ask(query)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:\n")
    for i, doc in enumerate(result["sources"]):
        print(f"Source {i+1} (Page {doc.metadata.get('page')}):")
        print(doc.page_content[:200])
        print("-" * 50)