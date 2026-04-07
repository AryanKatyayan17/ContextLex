from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

class HuggingFaceEmbedder:
    """
    Custom embedding class using HuggingFace models.
    Designed for retrieval tasks.
    """

    def __init__(self, model_name: str = "perplexity-ai/pplx-embed-v1-0.6b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.model.to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.
        """
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_text(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        Convert list of texts into embeddings.
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            batch_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            batch_embeddings = batch_embeddings.cpu().numpy()

            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        """
        return self.embed_text([query])[0]


if __name__ == "__main__":
    embedder = HuggingFaceEmbedder()

    texts = [
        "Machine learning is amazing.",
        "Neural networks are powerful models."
    ]

    vectors = embedder.embed_text(texts)

    print(f"Generated {len(vectors)} embeddings")
    print(f"Embedding dimension: {len(vectors[0])}")