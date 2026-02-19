"""
Embedding utilities using HuggingFace Transformers with BGE-micro model
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Union


class EmbeddingModel:
    """
    Wrapper for BGE-micro embedding model from HuggingFace
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedding model.

        Note: Using bge-small-en-v1.5 as bge-micro might not be available.
        If bge-micro is needed, we can use "BAAI/bge-micro-v2" or similar.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded embedding model: {model_name} on {self.device}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) into embedding vectors.

        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for processing multiple texts

        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state
                # Mean pooling: average over sequence length dimension
                attention_mask = encoded["attention_mask"]
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            # Convert to numpy and normalize
            embeddings = embeddings.cpu().numpy()
            # L2 normalize for better cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        # Concatenate all batches
        result = np.vstack(all_embeddings)
        return result


# Global model instance (lazy loading)
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the global embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        # Try bge-micro first, fallback to bge-small
        try:
            _embedding_model = EmbeddingModel("BAAI/bge-micro-v2")
        except:
            try:
                _embedding_model = EmbeddingModel("BAAI/bge-small-en-v1.5")
            except:
                # Final fallback
                _embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def encode_text(text: str) -> np.ndarray:
    """
    Convenience function to encode a single text.

    Args:
        text: Text string to encode

    Returns:
        Embedding vector as numpy array
    """
    model = get_embedding_model()
    embeddings = model.encode(text)
    return embeddings[0]  # Return first (and only) embedding

