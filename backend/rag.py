"""
RAG (Retrieval Augmented Generation) implementation using OpenAI API
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from vector_db import VectorDatabase, DistanceMetric
from embeddings import encode_text

# Load environment variables
# Try to load from parent directory (root) or current directory
import os
env_paths = [
    os.path.join(os.path.dirname(__file__), "..", ".env"),
    os.path.join(os.path.dirname(__file__), ".env"),
    ".env"
]
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break
else:
    load_dotenv()  # Fallback to default behavior

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)


def create_rag_prompt(query: str, context_texts: List[str]) -> str:
    """
    Create a prompt for RAG that includes retrieved context.

    Args:
        query: User's question
        context_texts: List of relevant text snippets from blog posts

    Returns:
        Formatted prompt string
    """
    context = "\n\n".join([
        f"[Context {i+1}]: {text}"
        for i, text in enumerate(context_texts)
    ])

    prompt = f"""You are a helpful assistant that answers questions based on the provided context from blog posts.

Context from blog posts:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so. Use the context to provide accurate and relevant information.

Answer:"""

    return prompt


def generate_rag_response(
    query: str,
    db: VectorDatabase,
    k: int = 5,
    model: str = "gpt-3.5-turbo"
) -> Dict[str, any]:
    """
    Generate a RAG response by retrieving relevant context and calling OpenAI.

    Args:
        query: User's question
        db: Vector database instance
        k: Number of relevant documents to retrieve
        model: OpenAI model to use

    Returns:
        Dictionary with answer and retrieved context
    """
    try:
        # Retrieve relevant blog posts
        query_vector = encode_text(query)
        results = db.search(query_vector, k=k, metric=DistanceMetric.COSINE)

        # Extract text from results
        context_texts = [metadata["text"] for entry_id, metadata, score in results]

        # Create prompt
        prompt = create_rag_prompt(query, context_texts)

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "retrieved_context": [
                {
                    "text": metadata["text"],
                    "score": float(score)
                }
                for entry_id, metadata, score in results
            ],
            "num_contexts": len(context_texts)
        }

    except Exception as e:
        return {
            "error": str(e),
            "answer": "Sorry, I encountered an error while generating a response."
        }

