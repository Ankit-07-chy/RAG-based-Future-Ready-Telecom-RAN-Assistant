"""
Evaluation framework for Telecom RAG.

Computes the five target KPIs on a held-out TeleQnA test split with per-question
CSV output. Methodology (documented so the numbers are defensible):

  Corpus           The retrieval corpus is built from 3GPP specs + O-RAN +
                   Simu5G + the TeleQnA TRAIN split. The TEST split is held out
                   and NOT indexed, so a test question cannot self-retrieve its
                   own answer (this is what made earlier runs trivially 1.0).

  Relevance        TeleQnA has no gold passage labels, so a retrieved chunk is
                   "relevant" (answer-bearing) if a high fraction of the
                   ground-truth answer tokens appear in it.

  MRR              Mean 1/rank of the first answer-bearing chunk in the reranked
                   top-k results.
  Top-k Accuracy   Fraction of questions with an answer-bearing chunk in the
                   reranked top-k.
  Recall           Fraction of questions with an answer-bearing chunk anywhere
                   in the top-N candidate pool (retriever recall, before rerank).
  Accuracy         True multiple-choice accuracy: the LLM selects an option
                   number from the retrieved context; scored against the
                   ground-truth option index.
  Faithfulness     LLM-as-judge grounding score (is the answer supported by the
                   retrieved context?), with a token-overlap fallback.
"""
import csv
import importlib.util
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    EVALS_DIR,
    EVAL_MAX_SAMPLES,
    EVAL_TEST_RATIO,
    RAW_TELEQNA_DIR,
    RESULTS_DIR,
    TOP_K_RERANK,
    TOP_K_RETRIEVE,
    VECTORSTORE_PATH,
)
from src.parse_teleqna import load_teleqna_eval_split
from src.rag_query import detect_query_type, parse_structured_response, ParseError

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = VECTORSTORE_PATH / "faiss_index"

# A chunk is "answer-bearing" if it contains at least this fraction of the
# ground-truth answer's content tokens.
ANSWER_BEARING_THRESHOLD = 0.5


def _load_embed_module():
    spec = importlib.util.spec_from_file_location(
        "embed_vectorstore",
        Path(__file__).parent / "3gpp_embed_vectorstore.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doc_identifier(doc) -> str:
    meta = doc.metadata or {}
    if meta.get("doc_id"):
        return str(meta["doc_id"])
    if meta.get("question_id"):
        return f"teleqna_{meta['question_id']}"
    section = meta.get("section", "na")
    source = Path(str(meta.get("source", "unknown"))).name
    chunk_index = meta.get("chunk_index", 0)
    return f"{source}_{section}_{chunk_index}"


_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on",
    "by", "with", "as", "that", "this", "be", "it", "from", "at", "which",
}


def _content_tokens(text: str) -> set:
    return {t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS and len(t) > 2}


def _answer_coverage(chunk_text: str, answer: str) -> float:
    """Fraction of the answer's content tokens present in the chunk."""
    ans_tokens = _content_tokens(answer)
    if not ans_tokens:
        return 0.0
    chunk_tokens = _content_tokens(chunk_text)
    return len(ans_tokens & chunk_tokens) / len(ans_tokens)


def _is_answer_bearing(chunk_text: str, answer: str) -> bool:
    return _answer_coverage(chunk_text, answer) >= ANSWER_BEARING_THRESHOLD


def _token_overlap_score(prediction: str, reference: str) -> float:
    pred_tokens = set(re.findall(r"\w+", prediction.lower()))
    ref_tokens = set(re.findall(r"\w+", reference.lower()))
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def _heuristic_faithfulness(answer: str, contexts: List[str]) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for ctx in contexts:
        context_tokens.update(re.findall(r"\w+", ctx.lower()))
    if not context_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    red_flags = [
        "based on external knowledge",
        "not mentioned in",
        "not in the provided context",
        "i don't have",
    ]
    if any(flag in answer.lower() for flag in red_flags):
        overlap *= 0.5
    return min(1.0, overlap)


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics."""
    mrr: float
    top_k_accuracy: float
    accuracy: float
    recall: float
    faithfulness: float
    num_samples: int
    avg_latency_ms: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        def flag(v, t):
            return "PASS" if v >= t else "FAIL"
        return f"""
============================================================
            RAG EVALUATION METRICS
============================================================
Metric              Value     Target    Status
------------------------------------------------------------
MRR                 {self.mrr:.2%}    > 75%     {flag(self.mrr, 0.75)}
Top-{TOP_K_RERANK} Accuracy       {self.top_k_accuracy:.2%}    > 85%     {flag(self.top_k_accuracy, 0.85)}
Accuracy            {self.accuracy:.2%}    > 80%     {flag(self.accuracy, 0.80)}
Recall              {self.recall:.2%}    > 85%     {flag(self.recall, 0.85)}
Faithfulness        {self.faithfulness:.2%}    > 90%     {flag(self.faithfulness, 0.90)}
------------------------------------------------------------
Samples evaluated:  {self.num_samples}
Avg latency:        {self.avg_latency_ms:.0f} ms
Timestamp:          {self.timestamp}
============================================================
"""


@dataclass
class QuestionEvalResult:
    """Per-question evaluation record."""
    query_id: str
    query: str
    reference_answer: str
    reference_option: int
    predicted_answer: str
    predicted_option: int
    reasoning: str
    sources: str
    query_type: str
    reciprocal_rank: float
    top_k_hit: bool
    accuracy: float
    recall: float
    faithfulness: float
    latency_ms: int
    retrieved_doc_ids: str


class RAGEvaluator:
    """End-to-end RAG evaluator using the TeleQnA held-out test set."""

    def __init__(
        self,
        vector_store_path: Path = FAISS_INDEX_PATH,
        embedding_model=None,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        llm_engine=None,
        judge_faithfulness: bool = True,
    ):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.llm_engine = llm_engine
        self.judge_faithfulness = judge_faithfulness
        self.hybrid_retriever = None
        self.reranker = None
        self._load_retrieval_stack()

    def _load_retrieval_stack(self):
        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Run `python main.py --mode pipeline` first."
            )

        from src.hybrid_retrieval import HybridRetriever

        self.hybrid_retriever = HybridRetriever(vector_store_path=self.vector_store_path)

        if self.use_reranker:
            from src.reranker import CrossEncoderReranker
            from src.config import RERANKER_MODEL

            self.reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)

        if self.embedding_model is None:
            embed_module = _load_embed_module()
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embed_module.MODEL_NAME,
                model_kwargs={"device": embed_module.MODEL_DEVICE},
                encode_kwargs={"normalize_embeddings": embed_module.NORMALIZE_EMBEDDINGS},
            )

        if self.llm_engine is None:
            try:
                from src.llm_engine import GroqLLMEngine
                from src.config import get_config

                cfg = get_config().llm
                self.llm_engine = GroqLLMEngine(
                    model_name=cfg.model_name,
                    api_key=cfg.api_key,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                )
            except Exception as exc:
                logger.warning(f"LLM unavailable for eval generation: {exc}")
                self.llm_engine = None

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVE):
        """Return (candidates, reranked) document/score lists.

        candidates = top-N from hybrid/semantic search (retriever recall pool)
        reranked   = top-TOP_K_RERANK after cross-encoder reranking
        """
        if self.use_hybrid:
            candidates = self.hybrid_retriever.hybrid_search(query, k=k)
        else:
            candidates = self.hybrid_retriever.semantic_search(query, k=k)

        docs = [doc for doc, _ in candidates]
        if self.reranker and docs:
            reranked = self.reranker.rerank(query, docs, top_k=TOP_K_RERANK)
        else:
            reranked = candidates[:TOP_K_RERANK]
        return candidates, reranked

    def _build_context_block(self, reranked) -> str:
        return "\n\n".join(
            f"[{i+1}] {doc.metadata.get('section', 'N/A')} | "
            f"{doc.metadata.get('section_title', doc.metadata.get('doc_type', ''))}\n"
            f"{doc.page_content}"
            for i, (doc, _) in enumerate(reranked)
        )

    def answer_mcq(self, record: Dict, reranked) -> Dict:
        """Have the LLM select an option for a multiple-choice question."""
        options = record.get("options") or []
        context_block = self._build_context_block(reranked)
        options_block = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, start=1))

        if self.llm_engine and options:
            system_prompt = (
                "You are an expert telecom (3GPP / O-RAN / 5G RAN) engineer. "
                "Use the retrieved context below together with telecom domain knowledge "
                "to choose the single best option for the multiple-choice question.\n"
                "Respond strictly in this format:\n"
                "ANSWER: <the number of the correct option>\n"
                "REASONING: <one or two sentences grounded in the context>\n"
                "SOURCES: <the [n] context blocks you relied on>"
            )
            prompt = (
                f"{system_prompt}\n\nCONTEXT:\n{context_block}\n\n"
                f"QUESTION: {record['question']}\n\nOPTIONS:\n{options_block}"
            )
            try:
                raw = self.llm_engine.generate(prompt, max_tokens=400, temperature=0.0)
                try:
                    parsed = parse_structured_response(raw)
                    answer_field, reasoning, sources = (
                        parsed["answer"], parsed["reasoning"], parsed["sources"],
                    )
                except ParseError:
                    answer_field, reasoning, sources = (
                        raw.strip(),
                        "LLM response did not follow the structured format.",
                        ", ".join(_doc_identifier(doc) for doc, _ in reranked),
                    )
                predicted_option = self._parse_option(answer_field, options)
                return {
                    "predicted_option": predicted_option,
                    "predicted_answer": (
                        options[predicted_option - 1] if 1 <= predicted_option <= len(options)
                        else answer_field
                    ),
                    "reasoning": reasoning,
                    "sources": sources,
                }
            except Exception as e:
                logger.warning(f"LLM MCQ answering failed, falling back to retrieval baseline: {e}")

        # No LLM / no options → fall back to nearest option by retrieval overlap.
        best_ctx = reranked[0][0].page_content if reranked else ""
        predicted_option = 0
        if options:
            scored = [(i + 1, _token_overlap_score(best_ctx, opt)) for i, opt in enumerate(options)]
            predicted_option = max(scored, key=lambda x: x[1])[0]
        return {
            "predicted_option": predicted_option,
            "predicted_answer": (
                options[predicted_option - 1] if 1 <= predicted_option <= len(options) else best_ctx[:200]
            ),
            "reasoning": "Retrieval-only baseline (no LLM): nearest option by context overlap.",
            "sources": ", ".join(_doc_identifier(doc) for doc, _ in reranked),
        }

    @staticmethod
    def _parse_option(answer_field: str, options: List[str]) -> int:
        """Extract a 1-based option index from the LLM answer."""
        m = re.search(r"\b(\d{1,2})\b", answer_field)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= len(options):
                return idx
        # Match by option text if no number was given.
        for i, opt in enumerate(options, start=1):
            if opt.lower().strip() and opt.lower().strip() in answer_field.lower():
                return i
        return 0

    def judge(self, answer: str, contexts: List[str]) -> float:
        """LLM-as-judge faithfulness (0/1), with token-overlap fallback."""
        if not self.judge_faithfulness or not self.llm_engine or not answer.strip():
            return _heuristic_faithfulness(answer, contexts)
        context_block = "\n\n".join(contexts)
        prompt = (
            "You are a strict fact-checker. Decide whether the ANSWER is supported "
            "by the CONTEXT (no claims that contradict or go beyond it).\n"
            "Reply with exactly one word: YES or NO.\n\n"
            f"CONTEXT:\n{context_block}\n\nANSWER:\n{answer}\n\nSupported?"
        )
        try:
            verdict = self.llm_engine.generate(prompt, max_tokens=5, temperature=0.0)
            return 1.0 if verdict.strip().upper().startswith("Y") else 0.0
        except Exception as exc:
            logger.warning(f"Faithfulness judge failed, using heuristic: {exc}")
            return _heuristic_faithfulness(answer, contexts)

    def evaluate_question(self, record: Dict, top_k: int = TOP_K_RERANK) -> QuestionEvalResult:
        query = record["question"]
        reference = record["answer"]
        reference_option = int(record.get("answer_index", 0) or 0)
        t0 = time.time()

        candidates, reranked = self.retrieve(query, k=TOP_K_RETRIEVE)
        reranked_docs = [doc for doc, _ in reranked]
        candidate_docs = [doc for doc, _ in candidates]
        retrieved_ids = [_doc_identifier(d) for d in reranked_docs]

        # Answer-bearing retrieval metrics (no gold passage labels available).
        rank = 0
        for idx, doc in enumerate(reranked_docs, start=1):
            if _is_answer_bearing(doc.page_content, reference):
                rank = idx
                break
        reciprocal_rank = 1.0 / rank if rank else 0.0
        top_k_hit = 0 < rank <= top_k
        recall = 1.0 if any(_is_answer_bearing(d.page_content, reference) for d in candidate_docs) else 0.0

        # MCQ answer + accuracy
        mcq = self.answer_mcq(record, reranked)
        predicted_option = mcq["predicted_option"]
        if reference_option > 0:
            accuracy = 1.0 if predicted_option == reference_option else 0.0
        else:
            # No ground-truth index → semantic match fallback.
            accuracy = _token_overlap_score(mcq["predicted_answer"], reference)

        contexts = [d.page_content for d in reranked_docs]
        faithfulness = self.judge(mcq["predicted_answer"] + "\n" + mcq["reasoning"], contexts)
        latency_ms = int((time.time() - t0) * 1000)

        return QuestionEvalResult(
            query_id=str(record["question_id"]),
            query=query,
            reference_answer=reference,
            reference_option=reference_option,
            predicted_answer=mcq["predicted_answer"],
            predicted_option=predicted_option,
            reasoning=mcq["reasoning"],
            sources=mcq["sources"],
            query_type=detect_query_type(query),
            reciprocal_rank=reciprocal_rank,
            top_k_hit=top_k_hit,
            accuracy=accuracy,
            recall=recall,
            faithfulness=faithfulness,
            latency_ms=latency_ms,
            retrieved_doc_ids=" | ".join(retrieved_ids),
        )

    def run_evaluation(
        self,
        max_samples: Optional[int] = EVAL_MAX_SAMPLES,
        test_ratio: float = EVAL_TEST_RATIO,
        teleqna_dir: Path = RAW_TELEQNA_DIR,
    ) -> Tuple[EvalMetrics, List[QuestionEvalResult]]:
        _, test_records = load_teleqna_eval_split(teleqna_dir, test_ratio=test_ratio)
        if max_samples and max_samples < len(test_records):
            test_records = test_records[:max_samples]

        logger.info(f"Evaluating {len(test_records)} held-out TeleQnA questions...")
        per_question: List[QuestionEvalResult] = []

        for idx, record in enumerate(test_records, start=1):
            result = self.evaluate_question(record)
            per_question.append(result)
            if idx % 10 == 0 or idx == len(test_records):
                running_acc = float(np.mean([r.accuracy for r in per_question]))
                logger.info(
                    f"  Progress {idx}/{len(test_records)} | "
                    f"running acc={running_acc:.2%} | "
                    f"latest rr={result.reciprocal_rank:.2f}"
                )

        n = len(per_question)
        metrics = EvalMetrics(
            mrr=float(np.mean([r.reciprocal_rank for r in per_question])) if n else 0.0,
            top_k_accuracy=float(np.mean([1.0 if r.top_k_hit else 0.0 for r in per_question])) if n else 0.0,
            accuracy=float(np.mean([r.accuracy for r in per_question])) if n else 0.0,
            recall=float(np.mean([r.recall for r in per_question])) if n else 0.0,
            faithfulness=float(np.mean([r.faithfulness for r in per_question])) if n else 0.0,
            num_samples=n,
            avg_latency_ms=float(np.mean([r.latency_ms for r in per_question])) if n else 0.0,
            timestamp=datetime.now().isoformat(),
        )
        return metrics, per_question

    def save_results(
        self,
        metrics: EvalMetrics,
        per_question: List[QuestionEvalResult],
        output_dir: Path = EVALS_DIR,
    ) -> Tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"evaluation_results_{stamp}.json"
        csv_path = output_dir / f"evaluation_per_question_{stamp}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        # Also keep a stable "latest" copy in results/ for the README/demo.
        with open(RESULTS_DIR / "latest_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        fieldnames = list(asdict(per_question[0]).keys()) if per_question else []
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_question:
                writer.writerow(asdict(row))

        logger.info(f"Saved aggregate metrics → {json_path}")
        logger.info(f"Saved per-question results → {csv_path}")
        return json_path, csv_path


def evaluate_single_response(
    query: str,
    answer: str,
    reasoning: str,
    retrieved_docs: List,
    reference_answer: Optional[str] = None,
    embedding_model=None,
) -> Dict[str, float]:
    """Score one RAG response for faithfulness and optional answer accuracy.

    Used by the Streamlit UI for ad-hoc, single-response evaluation.
    """
    contexts = [doc.page_content for doc in retrieved_docs]
    metrics = {
        "faithfulness": _heuristic_faithfulness(answer, contexts),
        "retrieved_chunks": float(len(retrieved_docs)),
    }
    if reference_answer:
        metrics["accuracy"] = _token_overlap_score(answer, reference_answer)
        if embedding_model:
            pred_emb = np.array(embedding_model.embed_query(answer))
            ref_emb = np.array(embedding_model.embed_query(reference_answer))
            cosine = float(
                np.dot(pred_emb, ref_emb)
                / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb) + 1e-8)
            )
            metrics["accuracy"] = max(metrics["accuracy"], cosine)
    return metrics


def run_full_evaluation(
    max_samples: Optional[int] = EVAL_MAX_SAMPLES,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    judge_faithfulness: bool = True,
) -> EvalMetrics:
    """Convenience entry point used by main.py --mode eval."""
    evaluator = RAGEvaluator(
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        judge_faithfulness=judge_faithfulness,
    )
    metrics, per_question = evaluator.run_evaluation(max_samples=max_samples)
    evaluator.save_results(metrics, per_question)
    print(metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    run_full_evaluation(max_samples=20)
