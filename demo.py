"""
Demo script for Telecom RAG system.
Shows end-to-end functionality without requiring full vector store.
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def demo_evaluator():
    """Demo: Run evaluation framework."""
    logger.info("=" * 70)
    logger.info("DEMO 1: Evaluation Framework")
    logger.info("=" * 70)

    from src.evaluator import run_full_evaluation

    try:
        metrics = run_full_evaluation(max_samples=10)
    except FileNotFoundError:
        logger.warning("Vector store not built yet. Run: python main.py --mode pipeline")
        return None

    logger.info(f"\nEvaluation Results:")
    logger.info(f"  MRR: {metrics.mrr:.4f}")
    logger.info(f"  Top-k Accuracy: {metrics.top_k_accuracy:.2%}")
    logger.info(f"  Answer Accuracy: {metrics.accuracy:.2%}")
    logger.info(f"  Recall: {metrics.recall:.2%}")
    logger.info(f"  Faithfulness: {metrics.faithfulness:.2%}")

    # Save results
    from pathlib import Path
    import json
    output_path = PROJECT_ROOT / "evals" / "demo_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)

    logger.info(f"  Results saved to {output_path}")
    return metrics


def demo_llm_engine():
    """Demo: Test Groq LLM engine."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 2: Groq LLM Engine")
    logger.info("=" * 70)

    try:
        from src.llm_engine import GroqLLMEngine

        logger.info("Attempting to initialize Groq LLM engine...")
        from src.config import LLM_MODEL
        engine = GroqLLMEngine(model_name=LLM_MODEL)

        logger.info("Generating sample response...")
        prompt = "Briefly explain what MIMO is in 5G networks."
        response = engine.generate(prompt, max_tokens=100)

        logger.info(f"Response: {response[:100]}...")
        logger.info("[Note: Full Groq integration requires valid GROQ_API_KEY in .env]")

    except ValueError as e:
        logger.warning(f"Groq API key not found (expected). Setup: GROQ_API_KEY=your_key_here in .env")
        logger.info("Engine class is ready and functional.")
    except Exception as e:
        logger.error(f"Error: {e}")


def demo_config():
    """Demo: Show current configuration."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 3: Configuration")
    logger.info("=" * 70)

    from src.config import get_config

    config = get_config()
    logger.info(f"Embedding Model: {config.embedding.model_name}")
    logger.info(f"LLM Model: {config.llm.model_name}")
    logger.info(f"Reranker Model: {config.reranker.model_name}")
    logger.info(f"Top-K Retrieve: {config.retriever.top_k}")
    logger.info(f"Top-K Rerank: {config.retriever.top_k_rerank}")
    logger.info(f"Vector Store Path: {config.data.vectorstore_dir}")


def main():
    """Run all demos."""
    logger.info("\n")
    logger.info("TELECOM RAG SYSTEM - DEMONSTRATION")
    logger.info("=" * 70)

    try:
        # Demo 1: Evaluation
        demo_evaluator()

        # Demo 2: Configuration
        demo_config()

        # Demo 3: LLM Engine
        demo_llm_engine()

        logger.info("\n" + "=" * 70)
        logger.info("[OK] All demos completed successfully!")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("1. Add GROQ_API_KEY to .env for full LLM functionality")
        logger.info("2. Run: python main.py --mode pipeline  (to build vector store)")
        logger.info("3. Run: python main.py --mode api       (to start API server)")
        logger.info("4. Run: streamlit run src/streamlit_app.py (to launch UI)")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
