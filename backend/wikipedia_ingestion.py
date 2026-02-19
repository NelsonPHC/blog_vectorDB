"""
Wikipedia content ingestion pipeline
Fetches Wikipedia page content and adds it to the vector database
"""

import wikipedia
import uuid
from typing import Dict, Optional
from vector_db import VectorDatabase
from embeddings import get_embedding_model, encode_text


def fetch_wikipedia_content(search_term: str, language: str = "en") -> Dict[str, any]:
    """
    Fetch content from Wikipedia for a given search term.

    Args:
        search_term: Search term to look up on Wikipedia
        language: Wikipedia language code (default: "en")

    Returns:
        Dictionary with page title, content, url, and summary
    """
    try:
        # Set language
        wikipedia.set_lang(language)

        # Search for the page
        search_results = wikipedia.search(search_term, results=1)
        if not search_results:
            raise ValueError(f"No Wikipedia page found for: {search_term}")

        # Get the page
        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)

        return {
            "title": page.title,
            "content": page.content,
            "url": page.url,
            "summary": page.summary,
            "page_id": str(page.pageid) if hasattr(page, 'pageid') else None
        }

    except wikipedia.exceptions.DisambiguationError as e:
        # If there's a disambiguation, use the first option
        page = wikipedia.page(e.options[0], auto_suggest=False)
        return {
            "title": page.title,
            "content": page.content,
            "url": page.url,
            "summary": page.summary,
            "page_id": str(page.pageid) if hasattr(page, 'pageid') else None
        }

    except wikipedia.exceptions.PageError:
        raise ValueError(f"Wikipedia page not found for: {search_term}")

    except Exception as e:
        raise ValueError(f"Error fetching Wikipedia content: {str(e)}")


def add_wikipedia_to_database(
    search_term: str,
    db: VectorDatabase,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict[str, any]:
    """
    Fetch Wikipedia content and add it to the vector database.
    Content is chunked into smaller pieces for better retrieval.

    Args:
        search_term: Search term to look up on Wikipedia
        db: VectorDatabase instance to add content to
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Overlap between chunks to preserve context

    Returns:
        Dictionary with status, number of chunks added, and page info
    """
    try:
        # Fetch Wikipedia content
        page_data = fetch_wikipedia_content(search_term)

        # Chunk the content
        content = page_data["content"]
        chunks = chunk_text(content, chunk_size, chunk_overlap)

        # Get embedding model
        embedding_model = get_embedding_model()

        # Embed and insert each chunk
        chunk_ids = []
        texts = [chunk["text"] for chunk in chunks]

        # Batch encode embeddings
        embeddings = embedding_model.encode(texts, batch_size=16)

        # Insert chunks into database
        for i, chunk in enumerate(chunks):
            # Generate unique ID for this chunk
            chunk_id = f"wikipedia_{page_data.get('page_id', uuid.uuid4())}_{i}"
            chunk_ids.append(chunk_id)

            # Create metadata
            metadata = {
                "text": chunk["text"],
                "source": "wikipedia",
                "title": page_data["title"],
                "url": page_data["url"],
                "summary": page_data.get("summary", ""),
                "chunk_index": i,
                "total_chunks": len(chunks)
            }

            # Insert into database
            db.insert(
                id=chunk_id,
                vector=embeddings[i],
                metadata=metadata
            )

        return {
            "status": "success",
            "page_title": page_data["title"],
            "page_url": page_data["url"],
            "num_chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "total_database_size": db.size()
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Overlap between chunks (in characters)

    Returns:
        List of dictionaries with 'text' and 'start_index'
    """
    if len(text) <= chunk_size:
        return [{"text": text, "start_index": 0}]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the end
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.7:  # If we found a good break point
                chunk_text = chunk_text[:break_point + 1]
                end = start + break_point + 1

        chunks.append({
            "text": chunk_text.strip(),
            "start_index": start
        })

        # Move start position with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break

    return chunks

