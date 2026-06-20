"""
Telecom RAG Pipeline 
steps: parse all sources → embed → build FAISS index → save.
Sources: 3GPP .docx, TeleQnA, O-RAN alarms/KPIs, Simu5G scenarios.
"""

import importlib.util # for dynamic import of embed_vectorstore.py
import json # for saving chunk stats
import logging # for logging to both console and file
import sys # for dynamic path manipulation
from datetime import datetime # for timestamped logs
from pathlib import Path # for file path handling
from typing import List, Optional # for type hints
from langchain_core.documents import Document # for Document objects used in embedding and vector store

#------------------------------Dynamic imports and path setup--------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MIN_CHUNK_WORDS, BATCH_SIZE, VECTORSTORE_PATH, RAW_DIR
from src.parse_3gpp import load_and_chunk_3gpp_docs, validate_chunks
from src.parse_teleqna import load_teleqna_documents
from src.parse_oran import parse_oran_data
from src.parse_simu5g import parse_simu5g_data


# Dynamic import due to dash in module name
_embed_spec = importlib.util.spec_from_file_location(
    "embed_vectorstore",
    Path(__file__).parent / "3gpp_embed_vectorstore.py",
)
_embed_module = importlib.util.module_from_spec(_embed_spec)
_embed_spec.loader.exec_module(_embed_module)
embed_and_store = _embed_module.embed_and_store

# Logging — both stderr and logs/ file
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _log_chunk_stats(source: str, docs: List[Document]) -> None:
    if not docs:
        logger.info(f"  [{source}] 0 chunks loaded.")
        return
    wc = [len(d.page_content.split()) for d in docs]
    logger.info(
        f"  [{source}] {len(docs)} chunks | "
        f"avg {sum(wc)//len(wc)} words | min {min(wc)} | max {max(wc)}"
    )


def run_pipeline(
    vector_store_path: Optional[Path] = None,
    merge_existing: bool = True,
    skip_embedding: bool = False,
) -> dict:
    """
    Full RAG pipeline:
      1. Parse 3GPP .docx documents
      2. Parse TeleQnA Q&A pairs
      3. Parse O-RAN alarm/KPI logs
      4. Parse Simu5G failure scenarios
      5. Embed all docs + build/merge FAISS index
      6. Save to data/vectorstore/
      7. Log chunk counts to logs/

    Returns dict with status, per-source chunk counts, and vector_store_path.
    """
    if vector_store_path is None:
        vector_store_path = VECTORSTORE_PATH / "faiss_index"

    logger.info("=" * 70)
    logger.info("TELECOM RAG PIPELINE — MULTI-SOURCE")
    logger.info("=" * 70)
    logger.info(f"Vector store : {vector_store_path}")
    logger.info(f"MIN_CHUNK_WORDS: {MIN_CHUNK_WORDS}  |  BATCH_SIZE: {BATCH_SIZE}")

    all_docs: List[Document] = []
    counts: dict = {}

    # Step 1 — 3GPP
    logger.info("\n[1/4] Parsing 3GPP Release 16/18 .docx documents...")
    try:
        gpp_docs = load_and_chunk_3gpp_docs(RAW_DIR / "3gpp_docs", min_chunk_size=MIN_CHUNK_WORDS)
        validate_chunks(gpp_docs, sample_size=min(5, len(gpp_docs)))
    except Exception as e:
        logger.warning(f"3GPP parsing failed: {e}")
        gpp_docs = []
    _log_chunk_stats("3gpp", gpp_docs)
    counts["3gpp"] = len(gpp_docs)
    all_docs.extend(gpp_docs)

    # Step 2 — TeleQnA
    logger.info("\n[2/4] Parsing TeleQnA Q&A dataset...")
    try:
        teleqna_docs = load_teleqna_documents(RAW_DIR / "teleqna_dataset")
    except Exception as e:
        logger.warning(f"TeleQnA parsing failed: {e}")
        teleqna_docs = []
    _log_chunk_stats("teleqna", teleqna_docs)
    counts["teleqna"] = len(teleqna_docs)
    all_docs.extend(teleqna_docs)

    # Step 3 — O-RAN
    logger.info("\n[3/4] Parsing O-RAN alarm/KPI logs...")
    try:
        oran_docs = parse_oran_data(RAW_DIR / "oran_datasets")
    except Exception as e:
        logger.warning(f"O-RAN parsing failed: {e}")
        oran_docs = []
    _log_chunk_stats("oran", oran_docs)
    counts["oran"] = len(oran_docs)
    all_docs.extend(oran_docs)

    # Step 4 — Simu5G
    logger.info("\n[4/4] Parsing Simu5G failure scenarios...")
    try:
        simu5g_docs = parse_simu5g_data(RAW_DIR / "simu5g")
    except Exception as e:
        logger.warning(f"Simu5G parsing failed: {e}")
        simu5g_docs = []
    _log_chunk_stats("simu5g", simu5g_docs)
    counts["simu5g"] = len(simu5g_docs)
    all_docs.extend(simu5g_docs)

    logger.info(f"\n  TOTAL docs to embed: {len(all_docs)}")
    for src, cnt in counts.items():
        logger.info(f"    {src:12s}: {cnt}")

    if not all_docs:
        logger.error("No documents loaded from any source. Aborting.")
        return {"status": "failed", "reason": "no_documents", "counts": counts}

    # Save chunk stats to logs/
    stats_path = LOGS_DIR / f"chunk_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    stats_path.write_text(json.dumps({**counts, "total": len(all_docs)}, indent=2))
    logger.info(f"  Chunk stats saved → {stats_path}")

    if skip_embedding:
        logger.info("\n[EMBED] Skipping embedding (skip_embedding=True)")
        return {"status": "success", "counts": counts, "total": len(all_docs), "skipped_embedding": True}

    logger.info("\n[EMBED] Building FAISS index...")
    vector_store = embed_and_store(
        all_docs,
        vector_store_path=vector_store_path,
        merge_existing=merge_existing,
    )

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Documents : {len(all_docs)} | Store: {vector_store_path}")
    logger.info("=" * 70)

    return {
        "status": "success",
        "counts": counts,
        "total": len(all_docs),
        "vector_store_path": str(vector_store_path),
        "vector_store": vector_store,
    }


def main():
    result = run_pipeline(
        vector_store_path=None,        # Uses default: data/vectorstore/faiss_index
        merge_existing=True,           # Merges with existing vector store
        skip_embedding=False,          # Performs embedding
    )
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
