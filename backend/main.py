"""
FastAPI backend for vector search system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json
import os
from pathlib import Path

from vector_db import VectorDatabase, DistanceMetric
from embeddings import get_embedding_model, encode_text
from rag import generate_rag_response
from wikipedia_ingestion import add_wikipedia_to_database

app = FastAPI(title="Vector Search API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global database instance
db = VectorDatabase()
embedding_model = None


class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str = Field(..., description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    metric: str = Field(default="cosine", description="Distance metric: cosine, dot_product, or euclidean")


class SearchResult(BaseModel):
    """Response model for search results"""
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    results: List[SearchResult]
    total_results: int


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., description="User's question")
    k: int = Field(default=5, ge=1, le=20, description="Number of context documents to retrieve")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    retrieved_context: List[Dict]
    num_contexts: int


class WikipediaIngestRequest(BaseModel):
    """Request model for Wikipedia ingestion endpoint"""
    search_term: str = Field(..., description="Wikipedia search term")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Overlap between chunks")


class WikipediaIngestResponse(BaseModel):
    """Response model for Wikipedia ingestion endpoint"""
    status: str
    page_title: Optional[str] = None
    page_url: Optional[str] = None
    num_chunks: Optional[int] = None
    total_database_size: Optional[int] = None
    error: Optional[str] = None


def get_blog_data_path(file_path: str = "blog.json") -> str:
    """Find the blog.json file path"""
    paths_to_try = [
        file_path,
        os.path.join(os.path.dirname(__file__), "..", file_path),
        os.path.join(os.path.dirname(__file__), file_path),
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Could not find blog.json in any of: {paths_to_try}")


def load_blog_data(file_path: str = "blog.json") -> List[dict]:
    """Load blog data from JSON file"""
    path = get_blog_data_path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_database_path() -> str:
    """Get the path for saving/loading the database"""
    backend_dir = os.path.dirname(__file__)
    return os.path.join(backend_dir, "data", "vector_db")


def should_rebuild_database() -> bool:
    """Check if database needs to be rebuilt"""
    db_path = get_database_path()

    # If database doesn't exist, need to build it
    if not VectorDatabase.exists(db_path):
        return True

    # Check if blog.json has been modified since database was created
    try:
        blog_path = get_blog_data_path()
        blog_mtime = os.path.getmtime(blog_path)

        # Check database file modification time
        db_vectors_path = db_path + '.vectors.npy'
        if os.path.exists(db_vectors_path):
            db_mtime = os.path.getmtime(db_vectors_path)
            # If blog.json is newer, rebuild
            if blog_mtime > db_mtime:
                return True
    except Exception as e:
        print(f"Error checking file timestamps: {e}, will rebuild database")
        return True

    return False


def initialize_database():
    """Initialize the vector database with blog data"""
    global db, embedding_model

    db_path = get_database_path()

    # Check if we can load from disk
    if not should_rebuild_database():
        try:
            print("Loading database from disk...")
            db = VectorDatabase.load(db_path)
            print(f"Loaded {db.size()} blog entries from disk!")
            # Still need to load embedding model for queries
            embedding_model = get_embedding_model()
            return
        except Exception as e:
            print(f"Error loading database from disk: {e}")
            print("Will rebuild database...")
            db = VectorDatabase()  # Reset database

    # Build database from scratch
    print("Loading blog data...")
    blogs = load_blog_data()
    print(f"Found {len(blogs)} blog entries")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Generating embeddings and indexing...")
    texts = [blog["metadata"]["text"] for blog in blogs]

    # Encode all texts in batches
    embeddings = embedding_model.encode(texts, batch_size=16)

    # Insert into database
    for i, blog in enumerate(blogs):
        db.insert(
            id=blog["id"],
            vector=embeddings[i],
            metadata=blog["metadata"]
        )

    print(f"Indexed {db.size()} blog entries successfully!")

    # Save to disk
    try:
        print("Saving database to disk...")
        db.save(db_path)
        print(f"Database saved to {db_path}")
    except Exception as e:
        print(f"Warning: Could not save database to disk: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    initialize_database()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vector Search API",
        "status": "running",
        "database_size": db.size()
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "database_size": db.size()}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search endpoint: finds the top-k most relevant blog entries for a query.

    Args:
        request: SearchRequest with query, k, and metric

    Returns:
        SearchResponse with list of results
    """
    try:
        # Validate metric
        metric_map = {
            "cosine": DistanceMetric.COSINE,
            "dot_product": DistanceMetric.DOT_PRODUCT,
            "euclidean": DistanceMetric.EUCLIDEAN
        }

        if request.metric not in metric_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric. Must be one of: {list(metric_map.keys())}"
            )

        metric = metric_map[request.metric]

        # Encode query
        query_vector = encode_text(request.query)

        # Search database
        results = db.search(query_vector, k=request.k, metric=metric)

        # Format results
        formatted_results = []
        for entry_id, metadata, score in results:
            formatted_results.append(SearchResult(
                id=entry_id,
                text=metadata.get("text", ""),
                score=round(score, 4)
            ))

        return SearchResponse(
            results=formatted_results,
            total_results=len(formatted_results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG chat endpoint: answers questions using retrieved context from blog posts.

    Args:
        request: ChatRequest with query and k

    Returns:
        ChatResponse with answer and retrieved context
    """
    try:
        result = generate_rag_response(
            query=request.query,
            db=db,
            k=request.k
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return ChatResponse(
            answer=result["answer"],
            retrieved_context=result["retrieved_context"],
            num_contexts=result["num_contexts"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/wikipedia", response_model=WikipediaIngestResponse)
async def ingest_wikipedia(request: WikipediaIngestRequest):
    """
    Wikipedia ingestion endpoint: fetches Wikipedia content and adds it to the vector database.

    Args:
        request: WikipediaIngestRequest with search_term and optional chunk parameters

    Returns:
        WikipediaIngestResponse with status and details
    """
    try:
        result = add_wikipedia_to_database(
            search_term=request.search_term,
            db=db,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))

        # Save database to disk after adding content
        try:
            db_path = get_database_path()
            db.save(db_path)
            print(f"Database saved after adding Wikipedia content: {request.search_term}")
        except Exception as e:
            print(f"Warning: Could not save database after Wikipedia ingestion: {e}")

        return WikipediaIngestResponse(
            status=result["status"],
            page_title=result.get("page_title"),
            page_url=result.get("page_url"),
            num_chunks=result.get("num_chunks"),
            total_database_size=result.get("total_database_size")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

