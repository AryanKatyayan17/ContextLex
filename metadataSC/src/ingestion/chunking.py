from typing import List
from src.ingestion.loader import PDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class SmartChunker:
    """
    Splits documents into semantically meaningful chunks
    while preserving metadata and adding chunk-level metadata.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)

        enriched_chunks = []
        for idx, chunk in enumerate(chunks):
            # Preserve original metadata
            metadata = dict(chunk.metadata)

            # Add new metadata
            metadata["chunk_id"] = idx
            metadata["chunk_size"] = len(chunk.page_content)

            enriched_chunk = Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )

            enriched_chunks.append(enriched_chunk)

        return enriched_chunks


if __name__ == "__main__":
    pdf_path = "data/books/handson_ml.pdf"

    loader = PDFLoader(pdf_path)
    docs = loader.load()

    chunker = SmartChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}\n")

    sample = chunks[0]
    print("Sample Chunk:\n")
    print(sample.page_content[:500])
    print("\nMetadata:", sample.metadata)