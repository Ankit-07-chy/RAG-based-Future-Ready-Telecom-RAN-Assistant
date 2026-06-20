"""
Evaluation framework for measuring RAG system performance.
Computes KPIs: MRR, Top-k Accuracy, Exact/Semantic Match, Recall, Faithfulness.
Integrates RAGAS for automated evaluation.
"""
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    mrr: float  # Mean Reciprocal Rank
    top_3_accuracy: float
    top_5_accuracy: float
    recall_at_5: float
    recall_at_10: float
    semantic_match_accuracy: float
    exact_match_accuracy: float
    faithfulness: float  # % answers grounded in context
    context_recall: float
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Pretty print metrics."""
        return f"""
========================================
   RAG EVALUATION METRICS
========================================
MRR (Mean Reciprocal Rank):       {self.mrr:.4f}
Top-3 Accuracy:                   {self.top_3_accuracy:.2%}
Top-5 Accuracy:                   {self.top_5_accuracy:.2%}
Recall @5:                        {self.recall_at_5:.2%}
Recall @10:                       {self.recall_at_10:.2%}
Semantic Match Accuracy:          {self.semantic_match_accuracy:.2%}
Exact Match Accuracy:             {self.exact_match_accuracy:.2%}
Faithfulness (% grounded):        {self.faithfulness:.2%}
Context Recall:                   {self.context_recall:.2%}
----------------------------------------
Timestamp:                        {self.timestamp}
"""


class RankingMetrics:
    """Compute ranking-based metrics (MRR, Top-k, Recall)."""

    @staticmethod
    def mean_reciprocal_rank(rankings: List[int]) -> float:
        """
        Compute MRR for batch of queries.

        Args:
            rankings: List of ranks (1-indexed) where answer was found
                     None/0 if not found in top-k

        Returns:
            Mean Reciprocal Rank
        """
        if not rankings:
            return 0.0

        rr_list = []
        for rank in rankings:
            if rank and rank > 0:
                rr_list.append(1.0 / rank)
            else:
                rr_list.append(0.0)

        return np.mean(rr_list) if rr_list else 0.0

    @staticmethod
    def top_k_accuracy(rankings: List[int], k: int = 5) -> float:
        """Fraction of queries where answer found in top-k."""
        if not rankings:
            return 0.0
        hits = sum(1 for rank in rankings if rank and 0 < rank <= k)
        return hits / len(rankings)

    @staticmethod
    def recall_at_k(
        retrieved_doc_ids: List[List[str]],
        relevant_doc_ids: List[List[str]],
        k: int = 5
    ) -> float:
        """
        Compute Recall@k: fraction of relevant docs that appear in top-k.

        Args:
            retrieved_doc_ids: List of doc ID lists from retrieval
            relevant_doc_ids: List of ground truth doc ID lists
            k: Cutoff position

        Returns:
            Recall@k
        """
        if len(retrieved_doc_ids) != len(relevant_doc_ids):
            raise ValueError("Mismatched query counts")

        recalls = []
        for retrieved, relevant in zip(retrieved_doc_ids, relevant_doc_ids):
            if not relevant:
                recalls.append(1.0)  # All relevant retrieved (vacuous truth)
                continue

            top_k = set(retrieved[:k])
            relevant_set = set(relevant)
            hits = len(top_k & relevant_set)
            recall = hits / len(relevant_set)
            recalls.append(recall)

        return np.mean(recalls) if recalls else 0.0


class MatchMetrics:
    """Compute answer matching metrics (exact, semantic)."""

    @staticmethod
    def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
        """Exact string match accuracy."""
        if len(predictions) != len(references):
            raise ValueError("Mismatched lengths")

        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        return matches / len(predictions) if predictions else 0.0

    @staticmethod
    def semantic_match_accuracy(
        predictions: List[str],
        references: List[str],
        embedding_model=None
    ) -> float:
        """
        Semantic similarity match using embeddings.
        Returns accuracy as % of answers with similarity > 0.8.
        """
        if embedding_model is None:
            logger.warning("No embedding model provided, falling back to exact match")
            return MatchMetrics.exact_match_accuracy(predictions, references)

        if len(predictions) != len(references):
            raise ValueError("Mismatched lengths")

        similarities = []
        for pred, ref in zip(predictions, references):
            # Embed both
            pred_emb = np.array(embedding_model.embed_query(pred))
            ref_emb = np.array(embedding_model.embed_query(ref))

            # Cosine similarity
            sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb) + 1e-8)
            similarities.append(sim)

        matches = sum(1 for sim in similarities if sim > 0.8)
        return matches / len(similarities) if similarities else 0.0


class GroundednessMetrics:
    """Compute groundedness metrics (faithfulness, context recall)."""

    @staticmethod
    def faithfulness_manual_eval(answers: List[str], contexts: List[str]) -> float:
        """
        Manual faithfulness evaluation (0-1).
        Checks if answer uses only information from context.
        Heuristic: look for "not found" / "no information" phrases.
        """
        faithful_count = 0

        for answer, context in zip(answers, contexts):
            # Red flags for hallucination
            red_flags = [
                "based on external knowledge",
                "not mentioned in",
                "not in the provided context",
                "i don't have",
            ]

            is_faithful = not any(flag.lower() in answer.lower() for flag in red_flags)

            if is_faithful:
                faithful_count += 1

        return faithful_count / len(answers) if answers else 0.0

    @staticmethod
    def context_recall(
        retrieved_chunks: List[List[str]],
        relevant_chunks: List[List[str]]
    ) -> float:
        """
        Context Recall: What fraction of relevant context was retrieved?
        (RAGAS metric)
        """
        recalls = []

        for retrieved, relevant in zip(retrieved_chunks, relevant_chunks):
            if not relevant:
                recalls.append(1.0)
                continue

            retrieved_set = set(retrieved)
            relevant_set = set(relevant)
            hits = len(retrieved_set & relevant_set)
            recall = hits / len(relevant_set)
            recalls.append(recall)

        return np.mean(recalls) if recalls else 0.0


class RAGEvaluator:
    """Complete RAG evaluation system."""

    def __init__(
        self,
        embedding_model=None,
        use_ragas: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            embedding_model: HuggingFace embeddings for semantic matching
            use_ragas: Whether to use RAGAS library for advanced metrics
        """
        self.embedding_model = embedding_model
        self.use_ragas = use_ragas

        if use_ragas:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    context_recall,
                    context_precision,
                    answer_relevancy,
                )
                self.ragas_metrics = {
                    "faithfulness": faithfulness,
                    "context_recall": context_recall,
                    "context_precision": context_precision,
                    "answer_relevancy": answer_relevancy,
                }
                logger.info("RAGAS metrics loaded successfully")
            except ImportError:
                logger.warning("RAGAS not installed, falling back to manual metrics")
                self.use_ragas = False

    def _generate_synthetic_test_cases(self) -> List[Dict]:
        """Generate synthetic evaluation test cases."""
        return [
            {
                "query_id": "q1",
                "query": "What is MIMO in 5G?",
                "retrieved_doc_ids": ["doc_1", "doc_2", "doc_3"],
                "relevant_doc_ids": ["doc_1"],
                "prediction": "MIMO (Multiple-Input Multiple-Output) is a wireless technology using multiple antennas for improved data transmission in 5G networks.",
                "reference": "MIMO enables simultaneous transmission via multiple antennas to improve spectral efficiency and throughput.",
                "retrieved_context": "MIMO technology leverages multiple antennas for spatial multiplexing in 5G NR systems.",
            },
            {
                "query_id": "q2",
                "query": "What causes high RRC failure rate?",
                "retrieved_doc_ids": ["doc_4", "doc_5"],
                "relevant_doc_ids": ["doc_4"],
                "prediction": "High RRC failure rates are typically caused by poor signal quality, excessive interference, or misconfigured RRC timers.",
                "reference": "RRC failures result from radio link failures, signaling errors, or network congestion.",
                "retrieved_context": "RRC connection establishment failures occur due to poor SINR, handover issues, or timer misconfigurations.",
            },
            {
                "query_id": "q3",
                "query": "How to optimize handover success rate?",
                "retrieved_doc_ids": ["doc_6", "doc_7", "doc_8"],
                "relevant_doc_ids": ["doc_6", "doc_7"],
                "prediction": "Optimize by tuning handover margins, reducing measurement gap, adjusting TTT, and improving cell coverage overlap.",
                "reference": "Handover optimization involves parameter tuning (A3 offset, TTT) and enhanced measurement procedures.",
                "retrieved_context": "Handover success depends on proper A3 offset settings, adequate coverage overlap, and suitable measurement events.",
            },
        ]

    def evaluate_retrieval(
        self,
        query_ids: List[str],
        retrieved_doc_ids: List[List[str]],
        relevant_doc_ids: List[List[str]],
        top_k_values: List[int] = [3, 5, 10],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Returns:
            Dictionary of retrieval metrics
        """
        logger.info(f"Evaluating retrieval on {len(query_ids)} queries...")

        # Find rank where correct doc appears
        rankings = []
        for query_retrieved, query_relevant in zip(retrieved_doc_ids, relevant_doc_ids):
            rank = None
            for i, doc_id in enumerate(query_retrieved, 1):
                if doc_id in query_relevant:
                    rank = i
                    break
            rankings.append(rank)

        metrics = {
            "mrr": RankingMetrics.mean_reciprocal_rank(rankings),
        }

        # Top-k accuracies
        for k in top_k_values:
            key = f"top_{k}_accuracy"
            metrics[key] = RankingMetrics.top_k_accuracy(rankings, k=k)
            key = f"recall@{k}"
            metrics[key] = RankingMetrics.recall_at_k(retrieved_doc_ids, relevant_doc_ids, k=k)

        return metrics

    def evaluate_generation(
        self,
        predictions: List[str],
        references: List[str],
        retrieved_contexts: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate answer generation quality.

        Returns:
            Dictionary of generation metrics
        """
        logger.info(f"Evaluating generation on {len(predictions)} answers...")

        metrics = {
            "exact_match": MatchMetrics.exact_match_accuracy(predictions, references),
            "semantic_match": MatchMetrics.semantic_match_accuracy(
                predictions, references, self.embedding_model
            ),
            "faithfulness": GroundednessMetrics.faithfulness_manual_eval(
                predictions, retrieved_contexts
            ),
        }

        return metrics

    def evaluate_full_pipeline(
        self,
        test_cases: List[Dict] = None,
    ) -> EvalMetrics:
        """
        Evaluate complete RAG pipeline.

        Args:
            test_cases: List of evaluation cases. If None, generates synthetic test cases:
                {
                    "query_id": str,
                    "query": str,
                    "retrieved_doc_ids": List[str],
                    "relevant_doc_ids": List[str],
                    "prediction": str,
                    "reference": str,
                    "retrieved_context": str,
                }

        Returns:
            EvalMetrics object with all KPIs
        """
        if not test_cases:
            logger.info("No test cases provided, generating synthetic test data...")
            test_cases = self._generate_synthetic_test_cases()

        logger.info(f"Evaluating {len(test_cases)} test cases...")

        # Extract components
        query_ids = [tc["query_id"] for tc in test_cases]
        retrieved_doc_ids = [tc["retrieved_doc_ids"] for tc in test_cases]
        relevant_doc_ids = [tc["relevant_doc_ids"] for tc in test_cases]
        predictions = [tc["prediction"] for tc in test_cases]
        references = [tc["reference"] for tc in test_cases]
        retrieved_contexts = [tc["retrieved_context"] for tc in test_cases]

        # Retrieval metrics
        retrieval_metrics = self.evaluate_retrieval(
            query_ids, retrieved_doc_ids, relevant_doc_ids
        )

        # Generation metrics
        generation_metrics = self.evaluate_generation(
            predictions, references, retrieved_contexts
        )

        # Context recall
        context_recall = GroundednessMetrics.context_recall(
            retrieved_doc_ids, relevant_doc_ids
        )

        # Combine into EvalMetrics
        metrics = EvalMetrics(
            mrr=retrieval_metrics.get("mrr", 0.0),
            top_3_accuracy=retrieval_metrics.get("top_3_accuracy", 0.0),
            top_5_accuracy=retrieval_metrics.get("top_5_accuracy", 0.0),
            recall_at_5=retrieval_metrics.get("recall@5", 0.0),
            recall_at_10=retrieval_metrics.get("recall@10", 0.0),
            semantic_match_accuracy=generation_metrics.get("semantic_match", 0.0),
            exact_match_accuracy=generation_metrics.get("exact_match", 0.0),
            faithfulness=generation_metrics.get("faithfulness", 0.0),
            context_recall=context_recall,
            timestamp=datetime.now().isoformat(),
        )

        return metrics

    def save_results(
        self,
        metrics: EvalMetrics,
        output_path: Path,
    ):
        """Save evaluation results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example: Create evaluator and test
    evaluator = RAGEvaluator()

    # Dummy test cases
    test_cases = [
        {
            "query_id": "q1",
            "query": "What is MIMO?",
            "retrieved_doc_ids": ["doc_1", "doc_2", "doc_3"],
            "relevant_doc_ids": ["doc_1", "doc_4"],
            "prediction": "MIMO is Multiple-Input Multiple-Output technology",
            "reference": "MIMO uses multiple antennas for transmission and reception",
            "retrieved_context": "MIMO (Multiple-Input Multiple-Output) is a wireless technology...",
        },
    ]

    # Evaluate
    metrics = evaluator.evaluate_full_pipeline(test_cases)
    print(metrics)

    # Save
    evaluator.save_results(metrics, Path("results/eval_metrics.json"))
