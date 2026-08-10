"""
FastAPI REST endpoint for Telecom RAG system.
Exposes query, retrieval, and health endpoints.
"""
import logging
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

from src.rag_chain import RAGChain, RAGResponse, QueryType
from src.security import sanitize_query, SecurityError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Telecom RAG API",
    description="Retrieval-Augmented Generation for 3GPP Specifications",
    version="1.0.0",
)

# Initialize RAG chain (lazy load)
_rag_chain = None


def get_rag_chain() -> RAGChain:
    """Get or initialize RAG chain."""
    global _rag_chain
    if _rag_chain is None:
        logger.info("Initializing RAG chain...")
        _rag_chain = RAGChain()
    return _rag_chain


# ─────────────────────────────────────────────────────────────────────────────
#  REQUEST/RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────


class RAGRequest(BaseModel):
    """RAG query request."""
    query: str
    k: int = 5
    use_hybrid: bool = False
    use_llm: bool = True


class SourceInfo(BaseModel):
    """Citation source."""
    source: str
    section: str
    title: str
    preview: str


class RAGResultResponse(BaseModel):
    """RAG result response."""
    query: str
    answer: str
    reasoning: str
    sources: List[SourceInfo]
    query_type: str
    confidence: float
    retrieved_chunks: int
    timestamp: str


class RetrievalRequest(BaseModel):
    """Pure retrieval request."""
    query: str
    k: int = 5
    use_hybrid: bool = False


class RetrievalResultResponse(BaseModel):
    """Pure retrieval result."""
    query: str
    results: List[SourceInfo]
    count: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_store_loaded: bool
    llm_available: bool
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        chain = get_rag_chain()
        return HealthResponse(
            status="healthy",
            vector_store_loaded=chain.vector_store is not None,
            llm_available=chain.llm_engine is not None,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=RAGResultResponse)
@app.post("/rag/query", response_model=RAGResultResponse)
async def rag_query(request: RAGRequest):
    """
    Query the RAG system.

    Args:
        query: Natural language query
        k: Number of documents to retrieve
        use_hybrid: Use hybrid search (semantic + keyword)
        use_llm: Generate LLM response (vs just retrieve)

    Returns:
        Structured RAG response with answer, reasoning, and sources
    """
    try:
        clean_query = sanitize_query(request.query)
        chain = get_rag_chain()

        # Process query
        response: RAGResponse = chain.process_query(
            query=clean_query,
            k=request.k,
            use_hybrid=request.use_hybrid,
            use_llm=request.use_llm,
        )

        # Format sources
        sources = [
            SourceInfo(
                source=source["source"],
                section=source["section"],
                title=source["title"],
                preview=source["preview"],
            )
            for source in response.sources
        ]

        return RAGResultResponse(
            query=request.query,
            answer=response.answer,
            reasoning=response.reasoning,
            sources=sources,
            query_type=response.query_type.value,
            confidence=response.confidence,
            retrieved_chunks=response.retrieved_chunks,
            timestamp=datetime.now().isoformat(),
        )

    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieval", response_model=RetrievalResultResponse)
async def retrieve_only(request: RetrievalRequest):
    """
    Pure retrieval endpoint (no LLM generation).

    Args:
        query: Search query
        k: Number of results to retrieve
        use_hybrid: Use hybrid search

    Returns:
        List of retrieved documents with metadata
    """
    try:
        clean_query = sanitize_query(request.query)
        chain = get_rag_chain()

        # Retrieve documents
        docs = chain.retrieve(
            query=clean_query,
            k=request.k,
            use_hybrid=request.use_hybrid,
        )

        # Format results
        results = [
            SourceInfo(
                source=doc.metadata.get("source", "unknown"),
                section=doc.metadata.get("section", "N/A"),
                title=doc.metadata.get("section_title", "N/A"),
                preview=doc.page_content[:200] + "...",
            )
            for doc in docs
        ]

        return RetrievalResultResponse(
            query=request.query,
            results=results,
            count=len(results),
            timestamp=datetime.now().isoformat(),
        )

    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    try:
        chain = get_rag_chain()

        if chain.vector_store is None:
            raise HTTPException(status_code=500, detail="Vector store not loaded")

        docs = list(chain.vector_store.docstore._dict.values())
        sections = set(d.metadata.get("section") for d in docs)
        sources = set(d.metadata.get("source") for d in docs)

        return {
            "total_chunks": len(docs),
            "unique_sections": len(sections),
            "unique_sources": len(sources),
            "sources": list(sources),
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query-types")
async def get_query_types():
    """Get supported query types and keywords."""
    from src.rag_chain import QueryRouter

    return {
        "types": [
            {
                "name": qtype.value,
                "keywords": keywords,
            }
            for qtype, keywords in QueryRouter.KEYWORDS.items()
        ]
    }


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Telecom RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "rag_query": "/rag/query (POST)",
            "retrieval": "/retrieval (POST)",
            "stats": "/stats (GET)",
            "query_types": "/query-types (GET)",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"

    logger.info(f"Starting Telecom RAG API on {host}:{port}")
    logger.info("Access API documentation at http://localhost:8000/docs")

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
