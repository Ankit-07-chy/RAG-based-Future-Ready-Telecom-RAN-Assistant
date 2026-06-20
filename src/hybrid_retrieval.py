"""
Hybrid retrieval system combining semantic search + BM25 keyword search.
Implements Reciprocal Rank Fusion (RRF) to merge two ranking strategies.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Dual-stage retrieval: semantic search (FAISS) + keyword search (BM25).
    Combines rankings using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: Optional[FAISS] = None,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
        vector_store_path: Optional[Path] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Pre-loaded FAISS index
            embeddings: HuggingFace embeddings model
            vector_store_path: Path to load FAISS index from disk
        """
        self.vector_store = vector_store
        self.embeddings = embeddings

        # Load vector store if path provided
        if vector_store_path and vector_store is None:
            logger.info(f"Loading FAISS index from {vector_store_path}...")
            if embeddings is None:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "embed_vectorstore",
                    Path(__file__).parent / "3gpp_embed_vectorstore.py"
                )
                embed_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(embed_module)

                MODEL_NAME = embed_module.MODEL_NAME
                MODEL_DEVICE = embed_module.MODEL_DEVICE
                NORMALIZE_EMBEDDINGS = embed_module.NORMALIZE_EMBEDDINGS

                embeddings = HuggingFaceEmbeddings(
                    model_name=MODEL_NAME,
                    model_kwargs={"device": MODEL_DEVICE},
                    encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
                )
                self.embeddings = embeddings

            self.vector_store = FAISS.load_local(
                str(vector_store_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )

        # Build BM25 index from documents
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from vector store documents."""
        if self.vector_store is None:
            raise RuntimeError("Vector store not loaded. Cannot build BM25 index.")

        logger.info("Building BM25 index from documents...")

        # Extract documents and content
        self.doc_store = self.vector_store.docstore._dict
        self.docs_list = list(self.doc_store.values())

        # Tokenize for BM25
        tokenized_docs = [
            doc.page_content.lower().split() for doc in self.docs_list
        ]

        # Build BM25
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"BM25 index ready ({len(self.docs_list)} documents)")

    def semantic_search(self, query: str, k: int = 20) -> List[Tuple]:
        """
        FAISS semantic search.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")

        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        return docs_with_scores

    def keyword_search(self, query: str, k: int = 20) -> List[Tuple]:
        """
        BM25 keyword search.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if not hasattr(self, 'bm25'):
            raise RuntimeError("BM25 index not built")

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_idx = np.argsort(scores)[-k:][::-1]

        # Return docs with scores
        results = [
            (self.docs_list[idx], float(scores[idx]))
            for idx in top_k_idx
            if scores[idx] > 0
        ]

        return results

    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple],
        keyword_results: List[Tuple],
        k: float = 60.0,
    ) -> List[Tuple]:
        """
        Reciprocal Rank Fusion (RRF) to combine two ranked lists.
        Formula: RRF(d) = Σ 1 / (k + rank(d))

        Args:
            semantic_results: Ranked results from semantic search
            keyword_results: Ranked results from BM25 search
            k: RRF constant (higher = more balanced, lower = semantic-heavy)

        Returns:
            Fused ranked list (document, combined_score) tuples
        """
        fused_scores = {}

        # Score semantic results
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_id = id(doc)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)

        # Score keyword results
        for rank, (doc, score) in enumerate(keyword_results, 1):
            doc_id = id(doc)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)

        # Create combined result list
        combined_docs = {id(doc): doc for doc, _ in semantic_results + keyword_results}
        combined = [
            (combined_docs[doc_id], score)
            for doc_id, score in sorted(
                fused_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        return combined

    def hybrid_search(
        self,
        query: str,
        k: int = 20,
        semantic_weight: float = 0.7,
        top_k_candidates: int = 20,
    ) -> List[Tuple]:
        """
        Hybrid search combining semantic + keyword search.

        Args:
            query: Search query
            k: Final number of results to return
            semantic_weight: Weight for semantic search (0-1), keyword = 1 - semantic_weight
            top_k_candidates: How many candidates to get from each retriever before fusion

        Returns:
            Top-k results fused from both strategies
        """
        logger.debug(f"Hybrid search: '{query}'")

        # Get candidates from both retrievers
        semantic_results = self.semantic_search(query, k=top_k_candidates)
        keyword_results = self.keyword_search(query, k=top_k_candidates)

        # Fuse rankings
        fused = self.reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            k=60.0 / semantic_weight,  # Adjust k parameter based on weight preference
        )

        # Return top-k
        return fused[:k]

    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            "vector_store_size": len(self.docs_list) if hasattr(self, 'docs_list') else 0,
            "bm25_indexed": hasattr(self, 'bm25'),
            "retrieval_strategies": ["semantic", "keyword", "hybrid"]
        }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    vector_store_path = project_root / "data" / "vectorstore" / "faiss_index"

    retriever = HybridRetriever(vector_store_path=vector_store_path)

    # Test hybrid search
    query = "What is MIMO in 5G?"
    results = retriever.hybrid_search(query, k=3)

    print(f"\n🔍 Hybrid search results for: '{query}'")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"    Section: {doc.metadata.get('section', 'N/A')}")
        print(f"    Title: {doc.metadata.get('section_title', 'N/A')}")
        print(f"    Preview: {doc.page_content[:150]}...")
