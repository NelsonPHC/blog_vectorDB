# Vector Search System

A complete vector search system with custom vector database, semantic search, and RAG (Retrieval Augmented Generation) chat interface.

## Features

- **Custom Vector Database**: Built from scratch with support for cosine similarity, dot product, and euclidean distance
- **Semantic Search**: Search blog posts using embeddings from HuggingFace Transformers (BGE model)
- **RAG Chat Interface**: Ask questions and get answers based on retrieved context from blog posts using OpenAI
- **Wikipedia Ingestion Pipeline**: Add Wikipedia page content to the vector database with automatic chunking
- **Persistent Storage**: Database is saved to disk and automatically loaded on startup
- **Modern UI**: React-based frontend with clean, responsive design

## Project Structure

```
blog_vectorDB/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── vector_db.py         # Custom vector database implementation
│   ├── embeddings.py         # Embedding utilities using HuggingFace
│   ├── rag.py               # RAG implementation with OpenAI
│   ├── wikipedia_ingestion.py # Wikipedia content ingestion pipeline
│   ├── requirements.txt     # Python dependencies
│   └── data/                # Saved vector database (auto-generated)
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app component
│   │   ├── Search.jsx       # Search interface
│   │   ├── Chat.jsx         # Chat interface
│   │   └── main.jsx         # React entry point
│   ├── package.json         # Node dependencies
│   └── vite.config.js       # Vite configuration
├── blog.json                # Blog data
└── .env                     # Environment variables (OPENAI_API_KEY)
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure you have a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Make sure `blog.json` is in the root directory (one level up from backend)

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Start the Backend

From the `backend` directory:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will:
- Load the embedding model (this may take a minute on first run)
- Load vector database from disk (if exists) or initialize from `blog.json`
- Start the API server on `http://localhost:8000`

**Note**: On first run, the database will be built and saved to `backend/data/`. Subsequent runs will load from disk much faster.

### Start the Frontend

From the `frontend` directory:
```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

## API Endpoints

### Search
- **POST** `/search`
- **Body**:
  ```json
  {
    "query": "your search query",
    "k": 10,
    "metric": "cosine"
  }
  ```
- **Response**: List of top-k matching blog posts with similarity scores

### Chat (RAG)
- **POST** `/chat`
- **Body**:
  ```json
  {
    "query": "your question",
    "k": 5
  }
  ```
- **Response**: Answer generated using retrieved context and OpenAI

### Wikipedia Ingestion
- **POST** `/ingest/wikipedia`
- **Body**:
  ```json
  {
    "search_term": "Artificial Intelligence",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
  ```
- **Response**: Status, page title, URL, number of chunks added, and total database size
- **Description**: Fetches Wikipedia page content, chunks it, embeds it, and adds it to the vector database. The database is automatically saved to disk after ingestion.

### Health Check
- **GET** `/health`
- Returns database status and size

## Distance Metrics

The vector database supports three distance metrics:

1. **Cosine Similarity** (default): Measures the cosine of the angle between vectors
2. **Dot Product**: Direct dot product of vectors
3. **Euclidean Distance**: L2 distance between vectors (converted to similarity)

## Technologies Used

- **Backend**: Python, FastAPI, HuggingFace Transformers, NumPy, OpenAI, Wikipedia API
- **Frontend**: React, Vite
- **Vector Database**: Custom implementation with flat index with disk persistence
- **Embeddings**: BGE-small-en-v1.5 (or bge-micro-v2 if available)

## Wikipedia Ingestion Usage

You can add Wikipedia content to the vector database using the API:

```bash
curl -X POST http://localhost:8000/ingest/wikipedia \
  -H "Content-Type: application/json" \
  -d '{
    "search_term": "Machine Learning",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

The system will:
1. Search Wikipedia for the term
2. Fetch the full page content
3. Chunk the content into overlapping pieces (for better retrieval)
4. Generate embeddings for each chunk
5. Add all chunks to the vector database
6. Save the updated database to disk

After ingestion, you can search for the Wikipedia content using the `/search` endpoint.

## Notes

- The embedding model will be downloaded on first run (may take a few minutes)
- The vector database uses a flat index for simplicity (suitable for this assignment)
- Database is automatically persisted to disk and loaded on startup
- Wikipedia content is chunked to improve retrieval quality
- For production use, consider using optimized vector databases like Pinecone, Weaviate, or Qdrant

