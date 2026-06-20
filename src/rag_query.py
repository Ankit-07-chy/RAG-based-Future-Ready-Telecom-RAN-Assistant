"""
RAG Query Interface
Query the vector store and retrieve relevant 3GPP document chunks.
Provides structured response parsing with mandatory ANSWER/REASONING/SOURCES fields.
"""
from pathlib import Path
from typing import Dict, List, Optional
import logging
import importlib.util
import re
import time
from dataclasses import dataclass


class ParseError(Exception):
    """Raised when LLM response is missing required structured fields."""
    pass

# Dynamic import of embedding module
def _load_embed_module():
    spec = importlib.util.spec_from_file_location(
        "embed_vectorstore",
        Path(__file__).parent / "3gpp_embed_vectorstore.py"
    )
    embed_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(embed_module)
    return embed_module

embed_module = _load_embed_module()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RAGResult:
    """Single retrieved document chunk."""
    content: str
    section: str
    section_title: str
    source: str
    word_count: int
    relevance_score: float


class TelecomRAG:
    """Query interface for Telecom RAG system."""

    def __init__(self, vector_store_path: Optional[Path] = None):
        """
        Initialize RAG system.

        Args:
            vector_store_path: Path to FAISS index (default: data/vectorstore/faiss_index)
        """
        if vector_store_path is None:
            vector_store_path = PROJECT_ROOT / "data" / "vectorstore" / "faiss_index"

        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.embeddings = None
        self._load_vector_store()

    def _load_vector_store(self):
        """Load FAISS vector store from disk."""
        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}\n"
                f"Run `python src/pipeline.py` to build it first."
            )

        logger.info(f"Loading vector store from {self.vector_store_path}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_module.MODEL_NAME,
            model_kwargs={"device": embed_module.MODEL_DEVICE},
            encode_kwargs={"normalize_embeddings": embed_module.NORMALIZE_EMBEDDINGS},
        )
        self.vector_store = FAISS.load_local(
            str(self.vector_store_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Loaded {len(self.vector_store.docstore._dict)} documents")

    def query(self, question: str, k: int = 5, threshold: float = 0.3) -> List[RAGResult]:
        """
        Query the vector store for relevant 3GPP chunks.

        Args:
            question: Natural language question
            k: Number of results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of RAGResult objects with retrieved chunks and scores
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not loaded. Call _load_vector_store() first.")

        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)

        results = []
        for doc, score in docs_with_scores:
            # FAISS returns distances, convert to similarity (1 / (1 + distance))
            similarity = 1 / (1 + score)

            if similarity >= threshold:
                results.append(RAGResult(
                    content=doc.page_content,
                    section=doc.metadata.get("section", "N/A"),
                    section_title=doc.metadata.get("section_title", "N/A"),
                    source=doc.metadata.get("source", "N/A"),
                    word_count=doc.metadata.get("word_count", 0),
                    relevance_score=similarity,
                ))

        return results

    def retrieve_by_section(self, section_num: str) -> List[str]:
        """
        Retrieve all chunks from a specific 3GPP section.

        Args:
            section_num: Section number (e.g., '5.3.1')

        Returns:
            List of chunk contents
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not loaded.")

        all_docs = list(self.vector_store.docstore._dict.values())
        matching = [
            doc.page_content
            for doc in all_docs
            if doc.metadata.get("section") == section_num
        ]
        return matching

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        if self.vector_store is None:
            return {"error": "Vector store not loaded"}

        docs = list(self.vector_store.docstore._dict.values())
        sections = set(d.metadata.get("section") for d in docs)

        return {
            "total_chunks": len(docs),
            "unique_sections": len(sections),
            "sections": sorted(list(sections)),
            "embedding_model": embed_module.MODEL_NAME,
            "store_path": str(self.vector_store_path),
        }


def format_results(results: List[RAGResult], show_preview: bool = True) -> str:
    """Format RAG results for display."""
    if not results:
        return "No relevant documents found."

    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n[{i}] §{result.section} — {result.section_title}")
        output.append(f"    Relevance: {result.relevance_score:.2%} | Words: {result.word_count} | Source: {result.source}")
        if show_preview:
            preview = result.content[:300].replace("\n", " ")
            output.append(f"    Preview: {preview}...")
    return "\n".join(output)


_QUERY_TYPE_KEYWORDS = {
    "spec_qa":      ["what is", "define", "specification", "standard", "ts 38", "release", "3gpp"],
    "rca":          ["root cause", "why did", "failure", "fault", "cause", "rca"],
    "anomaly":      ["anomaly", "alarm", "alert", "abnormal", "kpi", "threshold exceeded"],
    "optimization": ["optimize", "improve", "throughput", "latency", "configuration", "tune"],
}


def detect_query_type(question: str) -> str:
    """Classify query as spec_qa / rca / anomaly / optimization."""
    q = question.lower()
    for qtype, keywords in _QUERY_TYPE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return qtype
    return "spec_qa"


def parse_structured_response(raw: str) -> Dict[str, str]:
    """
    Extract ANSWER / REASONING / SOURCES from LLM output.
    Raises ParseError if any field is missing.
    """
    patterns = {
        "answer":    r"ANSWER\s*:\s*(.*?)(?=REASONING\s*:|SOURCES\s*:|$)",
        "reasoning": r"REASONING\s*:\s*(.*?)(?=ANSWER\s*:|SOURCES\s*:|$)",
        "sources":   r"SOURCES\s*:\s*(.*?)(?=ANSWER\s*:|REASONING\s*:|$)",
    }
    extracted: Dict[str, str] = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if match:
            extracted[field] = match.group(1).strip()

    missing = [f.upper() for f in ("answer", "reasoning", "sources") if not extracted.get(f)]
    if missing:
        raise ParseError(
            f"LLM response missing required field(s): {', '.join(missing)}.\n"
            f"Raw response:\n{raw[:500]}"
        )
    return extracted


class StructuredRAGQuery:
    """
    Full retrieval → chain → parse pipeline with structured output enforcement.
    Returns {answer, reasoning, sources, query_type, latency_ms}.
    """

    def __init__(self, rag: "TelecomRAG"):
        self.rag = rag

    def run(
        self,
        question: str,
        llm_generate_fn,
        k: int = 5,
        threshold: float = 0.3,
    ) -> Dict[str, object]:
        """
        Args:
            question: User query.
            llm_generate_fn: Callable(prompt: str) -> str from llm_engine.
            k: Number of chunks to retrieve.
            threshold: Minimum similarity threshold.

        Returns:
            {answer, reasoning, sources, query_type, latency_ms}

        Raises:
            ParseError: If LLM output is missing ANSWER, REASONING, or SOURCES.
        """
        t0 = time.time()
        query_type = detect_query_type(question)

        chunks = self.rag.query(question, k=k, threshold=threshold)
        context_block = "\n\n".join(
            f"[{i+1}] §{r.section} {r.section_title}\n{r.content}"
            for i, r in enumerate(chunks)
        )

        system_prompt = (
            "You are an expert telecom engineer. "
            "Only use the provided context. Do not use external knowledge.\n"
            "Respond strictly in this format:\n"
            "ANSWER: <direct answer>\n"
            "REASONING: <step-by-step chain of thought>\n"
            "SOURCES: <§X.X.X | doc_name> for each chunk used"
        )
        prompt = f"{system_prompt}\n\nCONTEXT:\n{context_block}\n\nQUESTION: {question}"

        raw_response = llm_generate_fn(prompt)
        parsed = parse_structured_response(raw_response)

        latency_ms = int((time.time() - t0) * 1000)
        return {
            "answer":     parsed["answer"],
            "reasoning":  parsed["reasoning"],
            "sources":    parsed["sources"],
            "query_type": query_type,
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    import sys

    # Example usage
    rag = TelecomRAG()

    # Show stats
    stats = rag.get_stats()
    print(f"\n📊 Vector Store Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique sections: {stats['unique_sections']}")
    print(f"  Embedding model: {stats['embedding_model']}")

    # Example query
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is MIMO in 5G?"

    print(f"\n🔍 Query: {question}")
    results = rag.query(question, k=3)
    print(format_results(results))
