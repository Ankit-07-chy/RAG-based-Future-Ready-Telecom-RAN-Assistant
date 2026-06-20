"""
Cross-encoder re-ranker for refining top-k search results.
Improves precision by scoring (query, document) pairs jointly.
"""
from typing import List, Tuple, Optional
import logging
import numpy as np
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

# Small, fast cross-encoder model suitable for CPU inference
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-MiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    """
    Re-ranks search results using a cross-encoder model.
    Joint scoring of (query, document) pairs for better relevance.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        """
        Initialize cross-encoder re-ranker.

        Args:
            model_name: HuggingFace model ID for cross-encoder
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("✅ Cross-encoder ready")

    def rerank(
        self,
        query: str,
        documents: List,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Tuple]:
        """
        Re-rank documents using cross-encoder scores.

        Args:
            query: Search query
            documents: List of Document objects to rerank
            top_k: Number of top results to return
            min_score: Minimum score threshold (0-1)

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        # Prepare sentence pairs
        sentences = [
            (query, doc.page_content[:512])  # Limit to 512 chars for efficiency
            for doc in documents
        ]

        # Score all pairs
        logger.debug(f"Scoring {len(sentences)} document pairs...")
        scores = self.model.predict(sentences)

        # Combine documents and scores
        results = list(zip(documents, scores))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and top_k
        results = [
            (doc, float(score))
            for doc, score in results
            if score >= min_score
        ][:top_k]

        return results

    def batch_rerank(
        self,
        query: str,
        document_batches: List[List],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Tuple]:
        """
        Re-rank documents from multiple batches (for large document sets).

        Args:
            query: Search query
            document_batches: List of document batches
            top_k: Number of top results to return
            min_score: Minimum score threshold

        Returns:
            Top-k results across all batches
        """
        all_results = []

        for batch in document_batches:
            results = self.rerank(query, batch, top_k=len(batch), min_score=min_score)
            all_results.extend(results)

        # Deduplicate by document ID and sort
        seen = set()
        unique_results = []
        for doc, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            doc_id = id(doc)
            if doc_id not in seen:
                unique_results.append((doc, score))
                seen.add(doc_id)

        return unique_results[:top_k]


if __name__ == "__main__":
    # Example usage
    from langchain_core.documents import Document

    reranker = CrossEncoderReranker()

    # Sample documents
    docs = [
        Document(
            page_content="MIMO (Multiple-Input Multiple-Output) is a wireless technology that uses multiple antennas.",
            metadata={"source": "3gpp", "section": "5.3.1"}
        ),
        Document(
            page_content="PRACH (Physical Random Access Channel) is used for initial access.",
            metadata={"source": "3gpp", "section": "5.3.2"}
        ),
        Document(
            page_content="Handover procedures in 5G NR are optimized for low latency.",
            metadata={"source": "3gpp", "section": "5.4.1"}
        ),
    ]

    query = "What is MIMO?"
    results = reranker.rerank(query, docs, top_k=2)

    print(f"\n📊 Cross-encoder re-ranking results for: '{query}'")
    for i, (doc, score) in enumerate(results, 1):
        print(f"[{i}] Score: {score:.4f}")
        print(f"    {doc.page_content[:100]}...")
