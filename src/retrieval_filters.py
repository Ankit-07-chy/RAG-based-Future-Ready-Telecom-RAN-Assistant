"""
Retrieval filters: source diversity + MMR (Maximal Marginal Relevance).
Prevents results from being dominated by a single source or redundant chunks.
"""
from typing import List, Dict, Tuple, Optional, Set
import logging
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)


class SourceDiversityFilter:
    """
    Filters and reweights search results to ensure diversity across sources.
    """

    KNOWN_SOURCES = {
        "3gpp": 0.4,           # 3GPP specifications
        "teleqna": 0.3,        # TeleQnA Q&A dataset
        "oran": 0.2,           # O-RAN alarms/KPIs
        "simu5g": 0.1,         # SimU5G scenarios
    }

    def __init__(self, source_weights: Optional[Dict[str, float]] = None):
        """
        Initialize diversity filter.

        Args:
            source_weights: Custom weights for each source (sums to 1.0)
                           Default: {"3gpp": 0.4, "teleqna": 0.3, "oran": 0.2, "simu5g": 0.1}
        """
        if source_weights:
            # Normalize weights
            total = sum(source_weights.values())
            self.source_weights = {k: v / total for k, v in source_weights.items()}
        else:
            self.source_weights = self.KNOWN_SOURCES.copy()

        logger.info(f"Source weights: {self.source_weights}")

    def _get_source(self, doc) -> str:
        """Extract source from document metadata."""
        # Try different metadata keys for source
        for key in ["source", "source_type", "doc_type"]:
            if key in doc.metadata:
                return doc.metadata[key].lower()

        # Fallback: infer from document source path
        source_path = doc.metadata.get("source", "")
        if "3gpp" in source_path.lower():
            return "3gpp"
        elif "teleqna" in source_path.lower():
            return "teleqna"
        elif "oran" in source_path.lower():
            return "oran"
        elif "simu5g" in source_path.lower():
            return "simu5g"

        return "unknown"

    def filter_by_source_diversity(
        self,
        results: List[Tuple],
        k: int = 5,
        min_sources: int = 1,
    ) -> List[Tuple]:
        """
        Filter results to maximize source diversity.

        Args:
            results: List of (document, score) tuples from search
            k: Number of results to return
            min_sources: Minimum number of different sources to include

        Returns:
            Filtered results with diverse sources, maintaining top scores
        """
        if not results:
            return []

        # Group by source
        sources = defaultdict(list)
        for doc, score in results:
            source = self._get_source(doc)
            sources[source].append((doc, score))

        logger.debug(f"Results from sources: {list(sources.keys())}")

        # Select from each source proportionally
        selected = []
        source_counts = defaultdict(int)

        for doc, score in results:
            source = self._get_source(doc)

            # Check if we should include this source
            weight = self.source_weights.get(source, 0.1)
            target_count = max(1, int(k * weight))

            if source_counts[source] < target_count:
                selected.append((doc, score))
                source_counts[source] += 1

            if len(selected) >= k:
                break

        # Fill remaining slots from best-scoring docs
        if len(selected) < k:
            for doc, score in results:
                if len(selected) >= k:
                    break
                if not any(d == doc for d, _ in selected):
                    selected.append((doc, score))

        return selected[:k]

    def reweight_by_source(
        self,
        results: List[Tuple],
        source_boost: Optional[Dict[str, float]] = None,
    ) -> List[Tuple]:
        """
        Reweight results based on source, then re-sort.

        Args:
            results: List of (document, score) tuples
            source_boost: Per-source score multipliers
                         E.g., {"teleqna": 1.2} boosts TeleQnA results by 20%

        Returns:
            Re-weighted and re-sorted results
        """
        if source_boost is None:
            source_boost = {}

        reweighted = []
        for doc, score in results:
            source = self._get_source(doc)
            boost = source_boost.get(source, 1.0)
            new_score = score * boost
            reweighted.append((doc, new_score))

        # Sort by new score
        reweighted.sort(key=lambda x: x[1], reverse=True)

        return reweighted

    def get_source_distribution(self, results: List[Tuple]) -> Dict[str, int]:
        """Get count of results per source."""
        distribution = defaultdict(int)
        for doc, _ in results:
            source = self._get_source(doc)
            distribution[source] += 1
        return dict(distribution)


class MMRFilter:
    """
    Maximal Marginal Relevance (MMR) filter.
    Balances relevance to the query against redundancy among selected chunks.
    MMR(d) = λ * sim(d, query) - (1-λ) * max_sim(d, selected)
    """

    def __init__(self, lambda_param: float = 0.5):
        """
        Args:
            lambda_param: 1.0 = pure relevance, 0.0 = pure diversity.
                          0.5 balances both (recommended).
        """
        self.lambda_param = lambda_param

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a, b = np.array(v1), np.array(v2)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def filter(
        self,
        results: List[Tuple],
        query_embedding: Optional[List[float]] = None,
        k: int = 5,
        embeddings_fn=None,
        query: Optional[str] = None,
    ) -> List[Tuple]:
        """
        Apply MMR to select k maximally relevant and diverse documents.

        Args:
            results: List of (Document, score) tuples, ordered by relevance.
            query_embedding: Pre-computed query embedding vector.
            k: Number of documents to select.
            embeddings_fn: Callable(text) -> List[float]; used when
                           query_embedding is None.
            query: Query string (used with embeddings_fn if query_embedding absent).

        Returns:
            MMR-selected subset of results (up to k).
        """
        if not results:
            return []
        if len(results) <= k:
            return results

        if query_embedding is None and embeddings_fn is None:
            logger.warning("MMRFilter: no embeddings provided — falling back to score order.")
            return results[:k]

        doc_embeddings = []
        for doc, score in results:
            if embeddings_fn is not None:
                doc_embeddings.append(embeddings_fn(doc.page_content))
            else:
                doc_embeddings.append(None)

        if query_embedding is None and embeddings_fn is not None and query:
            query_embedding = embeddings_fn(query)

        if query_embedding is None:
            logger.warning("MMRFilter: query embedding unavailable — falling back to score order.")
            return results[:k]

        selected_indices: List[int] = []
        candidate_indices = list(range(len(results)))

        for _ in range(min(k, len(results))):
            best_idx = None
            best_score = float("-inf")

            for idx in candidate_indices:
                if idx in selected_indices:
                    continue

                emb = doc_embeddings[idx]
                relevance = (
                    self._cosine_similarity(emb, query_embedding)
                    if emb is not None
                    else results[idx][1]
                )

                if selected_indices:
                    max_sim = max(
                        self._cosine_similarity(emb, doc_embeddings[sel])
                        for sel in selected_indices
                        if doc_embeddings[sel] is not None
                    ) if emb is not None else 0.0
                else:
                    max_sim = 0.0

                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)

        return [results[i] for i in selected_indices]


if __name__ == "__main__":
    # Example usage
    from langchain_core.documents import Document

    filter = SourceDiversityFilter()

    # Sample results from different sources
    results = [
        (Document(page_content="3GPP spec", metadata={"source": "3gpp"}), 0.95),
        (Document(page_content="3GPP spec 2", metadata={"source": "3gpp"}), 0.93),
        (Document(page_content="3GPP spec 3", metadata={"source": "3gpp"}), 0.91),
        (Document(page_content="3GPP spec 4", metadata={"source": "3gpp"}), 0.89),
        (Document(page_content="TeleQnA answer", metadata={"source": "teleqna"}), 0.88),
        (Document(page_content="O-RAN alarm", metadata={"source": "oran"}), 0.85),
    ]

    # Without diversity filter
    print("Without diversity filter (top 5):")
    for i, (doc, score) in enumerate(results[:5], 1):
        source = filter._get_source(doc)
        print(f"  {i}. [{source}] {score:.3f}")

    # With diversity filter
    filtered = filter.filter_by_source_diversity(results, k=5)
    print("\nWith diversity filter (top 5):")
    for i, (doc, score) in enumerate(filtered, 1):
        source = filter._get_source(doc)
        print(f"  {i}. [{source}] {score:.3f}")

    # Source distribution
    print(f"\nSource distribution: {filter.get_source_distribution(filtered)}")
