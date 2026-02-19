"""
Custom Vector Database Implementation
Supports insert, search with multiple distance metrics (cosine, dot product, euclidean)
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from enum import Enum


class DistanceMetric(Enum):
    """Supported distance metrics"""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class VectorDatabase:
    """
    A simple vector database implementation using a flat index.
    Stores vectors and metadata for similarity search.
    """

    def __init__(self):
        """Initialize an empty vector database"""
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.ids: List[str] = []

    def insert(self, id: str, vector: np.ndarray, metadata: Dict) -> None:
        """
        Insert a vector with associated metadata into the database.

        Args:
            id: Unique identifier for the entry
            vector: Embedding vector (numpy array)
            metadata: Dictionary containing metadata (e.g., text content)
        """
        # Convert to numpy array if not already
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        # Normalize vector for consistent distance calculations
        vector = vector.astype(np.float32)

        self.vectors.append(vector)
        self.metadata.append(metadata)
        self.ids.append(id)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE
    ) -> List[Tuple[Dict, float]]:
        """
        Search for the top-k most similar vectors.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            metric: Distance metric to use

        Returns:
            List of tuples (id, metadata, score) sorted by similarity (highest first)
        """
        if len(self.vectors) == 0:
            return []

        # Convert query vector to numpy array
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        else:
            query_vector = query_vector.astype(np.float32)

        # Calculate distances/similarities
        scores = []
        for vector in self.vectors:
            score = self._calculate_similarity(query_vector, vector, metric)
            scores.append(score)

        # Get top-k indices
        scores = np.array(scores)
        top_k_indices = np.argsort(scores)[::-1][:k]  # Sort descending

        # Return results with id, metadata and scores
        results = []
        for idx in top_k_indices:
            results.append((self.ids[idx], self.metadata[idx].copy(), float(scores[idx])))

        return results

    def _calculate_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: DistanceMetric
    ) -> float:
        """
        Calculate similarity/distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector
            metric: Distance metric to use

        Returns:
            Similarity score (higher is better for all metrics)
        """
        if metric == DistanceMetric.COSINE:
            # Cosine similarity: dot product of normalized vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(vec1, vec2) / (norm1 * norm2)

        elif metric == DistanceMetric.DOT_PRODUCT:
            # Dot product similarity
            return np.dot(vec1, vec2)

        elif metric == DistanceMetric.EUCLIDEAN:
            # Euclidean distance (convert to similarity by negating)
            # Higher score = closer vectors
            distance = np.linalg.norm(vec1 - vec2)
            return -distance  # Negate so higher is better

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def size(self) -> int:
        """Return the number of entries in the database"""
        return len(self.vectors)

    def clear(self) -> None:
        """Clear all entries from the database"""
        self.vectors = []
        self.metadata = []
        self.ids = []

    def save(self, filepath: str) -> None:
        """
        Save the database to disk.

        Args:
            filepath: Path to save the database
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # Save vectors as numpy array file for efficiency
        vectors_path = filepath + '.vectors.npy'
        if self.vectors:
            np.save(vectors_path, np.array(self.vectors))
        else:
            # Create empty file
            np.save(vectors_path, np.array([]))

        # Save metadata and ids using pickle
        data = {
            'metadata': self.metadata,
            'ids': self.ids
        }
        with open(filepath + '.data.pkl', 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'VectorDatabase':
        """
        Load the database from disk.

        Args:
            filepath: Path to load the database from

        Returns:
            Loaded VectorDatabase instance
        """
        db = cls()

        vectors_path = filepath + '.vectors.npy'
        data_path = filepath + '.data.pkl'

        if not os.path.exists(vectors_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"Database files not found at {filepath}")

        # Load vectors
        vectors_array = np.load(vectors_path, allow_pickle=True)
        if vectors_array.size > 0:
            # Convert numpy array to list of numpy arrays
            if vectors_array.ndim == 1:
                # Single vector (shouldn't happen in practice, but handle it)
                db.vectors = [vectors_array.astype(np.float32)]
            elif vectors_array.ndim == 2:
                # Multiple vectors: convert each row to a numpy array
                db.vectors = [vec.astype(np.float32) for vec in vectors_array]
            else:
                raise ValueError(f"Unexpected vector array shape: {vectors_array.shape}")
        else:
            db.vectors = []

        # Load metadata and ids
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            db.metadata = data['metadata']
            db.ids = data['ids']

        return db

    @staticmethod
    def exists(filepath: str) -> bool:
        """
        Check if a saved database exists.

        Args:
            filepath: Path to check

        Returns:
            True if database files exist, False otherwise
        """
        vectors_path = filepath + '.vectors.npy'
        data_path = filepath + '.data.pkl'
        return os.path.exists(vectors_path) and os.path.exists(data_path)

