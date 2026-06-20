"""
Telecom RAG System - Main CLI Entry Point for Streamlit, FastAPI

Pipeline, API, evaluation, and demo modes for the Telecom RAG System.
"""

# Importing Libraries
import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Base Directory
PROJECT_ROOT = Path(__file__).resolve().parent


# main function to run the Telecom RAG System
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Telecom RAG System - Retrieval-Augmented Generation for 3GPP Specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python main.py --mode pipeline            # Build vector store
            python main.py --mode api --port 8000     # Start API server
            python main.py --mode streamlit            # Launch Streamlit query UI
            python main.py --mode query --query "What is MIMO in 5G?"
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["pipeline", "api", "eval", "demo", "query", "streamlit"],
        default="demo",
        help="Execution mode (default: demo)"
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query string (for --mode query)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (for --mode api, default: 8000)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API host (for --mode api, default: 0.0.0.0)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of API workers (default: 4)"
    )

    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite existing vector store (for --mode pipeline)"
    )

    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Dry-run: parse only, skip embedding (for --mode pipeline)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Debug mode enabled")

    logger.info("=" * 80)
    logger.info("TELECOM RAG SYSTEM - Main CLI")
    logger.info("=" * 80)

    try:
        if args.mode == "pipeline":
            logger.info(f"Mode: PIPELINE (parse → embed → index)")
            from src.pipeline import run_pipeline
            result = run_pipeline(
                merge_existing=not args.no_merge,
                skip_embedding=args.skip_embedding,
            )
            logger.info(f"Pipeline completed: {result}")

        elif args.mode == "api":
            logger.info(f"Mode: API SERVER (host={args.host}, port={args.port}, workers={args.workers})")
            import uvicorn
            from src.api import app
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                workers=args.workers
            )

        elif args.mode == "eval":
            logger.info(f"Mode: EVALUATION (computing KPIs)")
            from src.evaluator import RAGEvaluator
            evaluator = RAGEvaluator()
            metrics = evaluator.evaluate_full_pipeline()
            logger.info(f"Evaluation complete:\n{metrics}")

        elif args.mode == "query":
            logger.info(f"Mode: QUERY")
            if not args.query:
                print("Error: --query parameter required for query mode")
                sys.exit(1)
            from src.rag_query import TelecomRAG
            rag = TelecomRAG()
            results = rag.query(args.query, k=5)
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] §{result.section} — {result.section_title}")
                print(f"    Score: {result.relevance_score:.2%}")
                print(f"    {result.content[:150]}...")

        elif args.mode == "demo":
            logger.info(f"Mode: DEMO (end-to-end validation)")
            import demo
            demo.main()

        elif args.mode == "streamlit":
            logger.info("Mode: STREAMLIT UI")
            import subprocess
            subprocess.run(["streamlit", "run", "src/streamlit_app.py"], check=False)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.debug)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("[OK] Completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
