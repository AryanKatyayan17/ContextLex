import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.file_path = file_path

    def load(self) -> List[Document]:
        """
        Load PDF and return documents with metadata.
        """
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()

        # Enhance metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(self.file_path)
            doc.metadata["page"] = doc.metadata.get("page", None)

        return documents


if __name__ == "__main__":
    pdf_path = "data/books/handson_ml.pdf"

    loader = PDFLoader(pdf_path)
    docs = loader.load()

    print(f"Total pages loaded: {len(docs)}\n")

    sample_doc = docs[0]
    print("Sample Document:\n")
    print(sample_doc.page_content[:500]) 
    print("\nMetadata:", sample_doc.metadata)