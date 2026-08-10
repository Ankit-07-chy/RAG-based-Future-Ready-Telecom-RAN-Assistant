import logging # used for logging in the RAGChain class and other components
from typing import List, Dict, Optional, Tuple # used for type annotations in the RAGChain and related classes
from enum import Enum 
from dataclasses import dataclass 
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import get_config, TOP_K_RETRIEVE, TOP_K_RERANK, VECTORSTORE_PATH, RERANKER_MODEL

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification types."""
    SPEC_QUESTION = "spec_question"        # 3GPP specification Q&A
    RCA_ANALYSIS = "rca_analysis"          # Root cause analysis
    ANOMALY_DETECTION = "anomaly_detection"
    OPTIMIZATION = "optimization"
    GENERAL = "general"


@dataclass
class RAGResponse:
    """Structured RAG response."""
    answer: str
    reasoning: str
    sources: List[Dict[str, str]]
    query_type: QueryType
    confidence: float
    retrieved_chunks: int


class ResponseFormatter:
    """Formats LLM outputs into structured ANSWER/REASONING/SOURCES format."""

    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """Extract ANSWER, REASONING, SOURCES from LLM output."""
        sections = {
            "answer": "",
            "reasoning": "",
            "sources": ""
        }

        # Try to find explicit sections
        import re

        # Extract ANSWER section
        answer_match = re.search(
            r"(?:ANSWER|Answer|answer)[:\s]+([\s\S]*?)(?=(?:REASONING|Reasoning|reasoning)|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if answer_match:
            sections["answer"] = answer_match.group(1).strip()

        # Extract REASONING section
        reasoning_match = re.search(
            r"(?:REASONING|Reasoning|reasoning)[:\s]+([\s\S]*?)(?=(?:SOURCES|Sources|sources)|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            sections["reasoning"] = reasoning_match.group(1).strip()

        # Extract SOURCES section
        sources_match = re.search(
            r"(?:SOURCES|Sources|sources)[:\s]+(.+?)$",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if sources_match:
            sections["sources"] = sources_match.group(1).strip()

        # Fallback: if no sections found, use whole text as answer
        if not sections["answer"]:
            sections["answer"] = text[:500]

        return sections

    @staticmethod
    def format_sources(documents: List[Document]) -> List[Dict[str, str]]:
        """Format documents as citations."""
        sources = []
        for doc in documents:
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "section": doc.metadata.get("section", "N/A"),
                "title": doc.metadata.get("section_title", "N/A"),
                "preview": doc.page_content[:100] + "..."
            })
        return sources

    @staticmethod
    def format_response(
        answer: str,
        reasoning: str,
        sources: List[Dict[str, str]],
        query_type: QueryType,
        confidence: float = 0.85
    ) -> RAGResponse:
        """Create structured RAG response."""
        return RAGResponse(
            answer=answer,
            reasoning=reasoning,
            sources=sources,
            query_type=query_type,
            confidence=confidence,
            retrieved_chunks=len(sources)
        )


class QueryRouter:
    """Routes queries to appropriate handlers based on detected type."""

    KEYWORDS = {
        QueryType.SPEC_QUESTION: [
            "what is", "explain", "define", "describe", "how",
            "specification", "3gpp", "ts", "tr", "technical"
        ],
        QueryType.RCA_ANALYSIS: [
            "why", "root cause", "failure", "problem", "error",
            "issue", "debug", "troubleshoot", "rca", "alarm"
        ],
        QueryType.ANOMALY_DETECTION: [
            "anomaly", "normal", "abnormal", "pattern", "unusual",
            "kpi", "metric", "trend", "outlier", "unexpected"
        ],
        QueryType.OPTIMIZATION: [
            "optimize", "improve", "enhance", "performance",
            "handover", "throughput", "latency", "efficiency",
            "best practice", "recommendation", "configuration"
        ],
    }

    @staticmethod
    def detect_query_type(query: str) -> QueryType:
        """Detect query type from keywords."""
        query_lower = query.lower()

        for qtype, keywords in QueryRouter.KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return qtype

        return QueryType.GENERAL

    @staticmethod
    def get_system_prompt(query_type: QueryType) -> str:
        """Get query-specific system prompt."""
        base_prompt = """You are an expert telecom engineer assisting with 5G/NR network questions.
You have access to 3GPP specifications, O-RAN documentation, and network simulation data.

IMPORTANT RULES:
1. ONLY use information from the provided context - DO NOT use external knowledge
2. If the answer is not in the context, say "I don't have information about this in the knowledge base"
3. Always cite your sources with document section numbers
4. Be precise and technical, suitable for network engineers

Respond in this exact format:
ANSWER: [Direct answer to the question]
REASONING: [Step-by-step explanation]
SOURCES: [List document sections used]"""

        if query_type == QueryType.RCA_ANALYSIS:
            base_prompt += """

For Root Cause Analysis:
1. List potential causes in order of likelihood
2. Explain how each cause could lead to the symptom
3. Suggest troubleshooting steps
4. Reference relevant 3GPP sections or alarm definitions"""

        elif query_type == QueryType.OPTIMIZATION:
            base_prompt += """

For Optimization Questions:
1. Explain the current mechanism
2. Identify optimization points
3. Provide specific configuration recommendations
4. Reference best practices from specs and case studies"""

        elif query_type == QueryType.ANOMALY_DETECTION:
            base_prompt += """

For Anomaly Detection:
1. Compare against normal patterns
2. Identify deviations and severity
3. Suggest root causes
4. Recommend monitoring thresholds"""

        return base_prompt


def format_fallback_content(page_content: str, query: str, exception: Optional[Exception] = None) -> Tuple[str, str]:
    """Format retrieved document chunks cleanly when LLM generation falls back."""
    import re
    if page_content.startswith("Question:"):
        q_match = re.search(r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?:\s*Explanation:\s*(.*))?$", page_content, re.DOTALL | re.IGNORECASE)
        if q_match:
            question_text = q_match.group(1).strip()
            answer_text = q_match.group(2).strip()
            explanation_text = q_match.group(3).strip() if q_match.group(3) else ""
            
            answer = f"According to the retrieved reference question ('{question_text}'), the correct answer is: '{answer_text}'."
            if explanation_text:
                answer += f" Explanation: {explanation_text}"
            
            reasoning = "Retrieved direct Q&A match from TeleQnA dataset."
            if exception:
                reasoning += f" (LLM generation fallback due to: {exception})"
            return answer, reasoning

    answer = page_content
    reasoning = "Retrieved matching reference passage from knowledge base."
    if exception:
        reasoning += f" (LLM generation fallback due to: {exception})"
    return answer, reasoning


class RAGChain:
    """
    Unified RAG chain combining retrieval, context formatting, and LLM generation.
    """

    def __init__(
        self,
        vector_store_path: Optional[Path] = None,
        llm_engine=None,
        reranker=None,
        diversity_filter=None,
    ):
        """
        Initialize RAG chain.

        Args:
            vector_store_path: Path to FAISS index
            llm_engine: LLM engine for generation
            reranker: Cross-encoder reranker
            diversity_filter: Source diversity filter
        """
        self.vector_store_path = vector_store_path or (VECTORSTORE_PATH / "faiss_index")
        self.llm_engine = llm_engine
        self.reranker = reranker
        self.diversity_filter = diversity_filter
        self.hybrid_retriever = None

        # If no LLM engine provided, attempt to instantiate one from config
        if self.llm_engine is None:
            try:
                cfg = get_config()
                llm_cfg = cfg.llm
                from src.llm_engine import GroqLLMEngine

                logger.info("Initializing default Groq LLM engine from configuration...")
                self.llm_engine = GroqLLMEngine(
                    model_name=llm_cfg.model_name,
                    api_key=llm_cfg.api_key,
                    max_tokens=llm_cfg.max_tokens,
                    temperature=llm_cfg.temperature,
                )
                logger.info("LLM engine initialized successfully")
            except Exception as e:
                logger.warning(f"LLM engine not available: {e}. Continuing without LLM.")
                self.llm_engine = None

        # Load vector store
        self._load_vector_store()
        self._init_retrieval_components()

    def _init_retrieval_components(self):
        """Initialize hybrid retrieval, reranker, and diversity filter."""
        cfg = get_config()
        try:
            from src.hybrid_retrieval import HybridRetriever

            self.hybrid_retriever = HybridRetriever(vector_store=self.vector_store)
            logger.info("Hybrid retriever initialized")
        except Exception as exc:
            logger.warning(f"Hybrid retriever unavailable: {exc}")

        if self.reranker is None and cfg.retriever.use_reranking:
            try:
                from src.reranker import CrossEncoderReranker

                self.reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
            except Exception as exc:
                logger.warning(f"Reranker unavailable: {exc}")

        if self.diversity_filter is None and cfg.retriever.use_source_diversity:
            try:
                from src.retrieval_filters import SourceDiversityFilter

                self.diversity_filter = SourceDiversityFilter()
            except Exception as exc:
                logger.warning(f"Diversity filter unavailable: {exc}")

    def _load_vector_store(self):
        """Load FAISS vector store."""
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

        self.vector_store = FAISS.load_local(
            str(self.vector_store_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Loaded vector store with {len(self.vector_store.docstore._dict)} documents")

    def retrieve(
        self,
        query: str,
        k: int = TOP_K_RERANK,
        use_hybrid: bool = True,
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search, reranking, and diversity filters.
        """
        candidate_k = max(TOP_K_RETRIEVE, k)

        if use_hybrid and self.hybrid_retriever is not None:
            results = self.hybrid_retriever.hybrid_search(query, k=candidate_k)
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]
        else:
            results = self.vector_store.similarity_search_with_score(query, k=candidate_k)
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]

        cfg = get_config()
        paired = list(zip(docs, scores))

        if self.reranker and docs:
            reranked = self.reranker.rerank(query, docs, top_k=candidate_k, min_score=cfg.reranker.min_score)
            paired = reranked

        if self.diversity_filter and len(paired) > k:
            paired = self.diversity_filter.filter_by_source_diversity(paired, k=k)
        else:
            paired = paired[:k]

        return [doc for doc, _ in paired]

    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context prompt."""
        if not documents:
            return "No relevant documents found in knowledge base."

        context_parts = ["=== RETRIEVED CONTEXT ===\n"]

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("section", "N/A")
            title = doc.metadata.get("section_title", "N/A")

            context_parts.append(f"[{i}] Source: {source} | Section: §{section} | Title: {title}")
            context_parts.append(f"Content: {doc.page_content}")
            context_parts.append("")

        return "\n".join(context_parts)

    def generate(
        self,
        query: str,
        retrieved_docs: List[Document],
        query_type: QueryType,
    ) -> str:
        """Generate LLM response with retrieved context."""
        if not self.llm_engine:
            raise RuntimeError("LLM engine not configured")

        system_prompt = QueryRouter.get_system_prompt(query_type)
        context = self.format_context(retrieved_docs)

        full_prompt = f"""{system_prompt}

{context}

Question: {query}

Respond in the required format (ANSWER / REASONING / SOURCES):"""

        response = self.llm_engine.generate(full_prompt, max_tokens=512, temperature=0.3)
        return response

    def process_query(
        self,
        query: str,
        k: int = 5,
        use_hybrid: bool = False,
        use_llm: bool = True,
    ) -> RAGResponse:
        """
        Process complete RAG query.

        Args:
            query: User query
            k: Number of retrieved documents
            use_hybrid: Use hybrid retrieval
            use_llm: Generate LLM response (vs just retrieve)

        Returns:
            Structured RAG response
        """
        # Detect query type
        query_type = QueryRouter.detect_query_type(query)
        logger.info(f"Query type detected: {query_type.value}")

        # Retrieve documents
        docs = self.retrieve(query, k=k, use_hybrid=use_hybrid)
        logger.info(f"Retrieved {len(docs)} documents")

        if not docs:
            return RAGResponse(
                answer="No relevant information found in knowledge base.",
                reasoning="The query did not match any indexed documents.",
                sources=[],
                query_type=query_type,
                confidence=0.0,
                retrieved_chunks=0
            )

        # Generate response if LLM available
        if use_llm and self.llm_engine:
            try:
                llm_response = self.generate(query, docs, query_type)
                sections = ResponseFormatter.extract_sections(llm_response)
                answer = sections.get("answer", "")
                reasoning = sections.get("reasoning", "")
            except Exception as e:
                logger.warning(f"LLM generation failed, falling back to retrieval: {e}")
                answer, reasoning = format_fallback_content(docs[0].page_content, query, e)
        else:
            answer, reasoning = format_fallback_content(docs[0].page_content, query, None)

        # Format sources
        sources = ResponseFormatter.format_sources(docs)

        # Calculate confidence based on relevance and document count
        confidence = min(1.0, len(docs) / k)

        return RAGResponse(
            answer=answer,
            reasoning=reasoning,
            sources=sources,
            query_type=query_type,
            confidence=confidence,
            retrieved_chunks=len(docs)
        )

    def stream_response(
        self,
        query: str,
        k: int = 5,
    ):
        """Stream response token-by-token."""
        query_type = QueryRouter.detect_query_type(query)
        docs = self.retrieve(query, k=k)

        system_prompt = QueryRouter.get_system_prompt(query_type)
        context = self.format_context(docs)

        full_prompt = f"""{system_prompt}

{context}

Question: {query}

Respond in the required format (ANSWER / REASONING / SOURCES):"""

        if self.llm_engine and hasattr(self.llm_engine, 'generate_streaming'):
            for token in self.llm_engine.generate_streaming(full_prompt):
                yield token


if __name__ == "__main__":
    # Example usage
    chain = RAGChain()

    # Test queries
    test_queries = [
        "What is MIMO in 5G?",
        "Why is my cell experiencing high RRC failure rate?",
        "Is this KPI pattern normal?",
        "How can I optimize handover success rate?"
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")

        try:
            response = chain.process_query(query, k=3, use_llm=False)
            print(f"Type: {response.query_type.value}")
            print(f"Confidence: {response.confidence:.2%}")
            print(f"Retrieved chunks: {response.retrieved_chunks}")
            print(f"\nAnswer: {response.answer[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
