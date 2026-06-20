#---------------------------------------------------------------------------------
from pathlib import Path # For file path manipulations
from typing import Optional, List, Generator, Tuple # For type annotations
import logging # For logging
import time # For timing operations
import os # For file system operations
import gc # For garbage collection
import psutil # For monitoring memory usage
from tqdm import tqdm # For progress bars
import numpy as np # For numerical operations
from langchain_huggingface import HuggingFaceEmbeddings # For embedding model
from langchain_community.vectorstores import FAISS # For vector store
from langchain_core.documents import Document # For document representation
from langchain_core.embeddings import Embeddings # For embedding interface


#------------------------------------CONFIG-----------------------------------------
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vectorstore"
FAISS_INDEX_PATH  = VECTOR_STORE_DIR / "faiss_index"

# Model config
MODEL_NAME          = "BAAI/bge-base-en-v1.5"  # 768-dim. Matches spec requirement.
MODEL_DEVICE        = "cpu"                    # Use "cuda" if GPU available/ not available in my device
NORMALIZE_EMBEDDINGS= True                     # Recommended for cosine similarity.
BATCH_SIZE          = 32                       # Spec-mandated batch size.

# Thresholds
MIN_EMBEDDING_DIM   = 100                  # Sanity check for model output.
MAX_MEMORY_USAGE    = 0.85                 # Halt if RAM exceeds 85%.
EMBEDDING_TIMEOUT   = 300                  # 5 min timeout per batch.

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

#---------------------------------------------------------------------------------------------


def check_memory_usage() -> bool:
    """Return True if memory usage is below MAX_MEMORY_USAGE."""
    return psutil.virtual_memory().percent < MAX_MEMORY_USAGE * 100

def validate_embeddings(embeddings: List[List[float]]) -> bool:
    if not embeddings:
        return False
    dim = len(embeddings[0])
    if dim < MIN_EMBEDDING_DIM:
        logger.error(f"Embedding dimension {dim} < {MIN_EMBEDDING_DIM} — invalid model output.")
        return False
    for emb in embeddings:
        if len(emb) != dim:
            logger.error("Inconsistent embedding dimensions in batch.")
            return False
        if np.all(emb == 0):  # Zero vector
            logger.warning("Zero-vector embedding detected — may indicate model failure.")
    return True

def get_embedding_model() -> Embeddings:
    start = time.time()
    logger.info(f"Loading embedding model: {MODEL_NAME} (device={MODEL_DEVICE})...")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": MODEL_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
    )

    # Test with a dummy text
    test_emb = embeddings.embed_query("test")
    if not validate_embeddings([test_emb]):
        raise RuntimeError("Embedding model validation failed.")

    logger.info(f"Model loaded in {time.time() - start:.2f}s.")
    return embeddings


def embed_documents(
    docs: List[Document],
    embeddings: Embeddings,
    batch_size: int = BATCH_SIZE,
) -> List[List[float]]:
    
    all_embeddings = []
    for i in tqdm(
        range(0, len(docs), batch_size),
        desc="Embedding batches",
        unit="batch",
    ):
        batch = docs[i : i + batch_size]
        if not check_memory_usage():
            logger.warning("Memory usage high — pausing embedding to prevent OOM.")
            import gc
            gc.collect()
            time.sleep(5)

        try:
            batch_embeddings = embeddings.embed_documents(
                [d.page_content for d in batch]
            )
            if not validate_embeddings(batch_embeddings):
                raise ValueError("Invalid embeddings in batch.")
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Failed to embed batch {i//batch_size}: {e}")
            raise
    return all_embeddings

def build_vector_store(
    docs: List[Document],
    embeddings: Embeddings,
    save_path: Path,
    merge_existing: bool = True,
) -> FAISS:
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if merge_existing and save_path.exists():
        logger.info(f"Loading existing index from {save_path} for incremental update...")
        existing_store = FAISS.load_local(
            str(save_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Existing index has {len(existing_store.docstore._dict)} documents.")
    else:
        existing_store = None

    # Embed all new documents
    logger.info(f"Embedding {len(docs)} new documents...")
    new_embeddings = embed_documents(docs, embeddings)

    # Create new FAISS index for the batch
    new_index = FAISS.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    new_index.index = new_index.index  # Ensure index is built

    if existing_store:
        # Merge indices (FAISS does not natively support incremental adds with metadata)
        # Workaround: Rebuild combined index
        combined_docs = existing_store.docstore._dict.values()
        combined_docs = list(combined_docs) + docs
        combined_embeddings = [
            existing_store.embedding_function.embed_query(doc.page_content)
            for doc in combined_docs
        ]
        vector_store = FAISS.from_embeddings(
            embeddings=combined_embeddings,
            documents=combined_docs,
            embedding=embeddings,
        )
    else:
        vector_store = new_index

    # Save
    vector_store.save_local(str(save_path))
    logger.info(f"Vector store saved to {save_path} (total docs: {len(vector_store.docstore._dict)}).")
    return vector_store


def validate_vector_store(store: FAISS, sample_size: int = 5) -> None:
    """Spot-check the vector store for integrity."""
    if len(store.docstore._dict) == 0:
        logger.error("Vector store is empty!")
        return

    # Check embedding dimensions
    sample_docs = list(store.docstore._dict.values())[:sample_size]
    for doc in sample_docs:
        emb = store.embedding_function.embed_query(doc.page_content)
        if len(emb) < MIN_EMBEDDING_DIM:
            logger.error(f"Invalid embedding dimension for doc: {doc.metadata.get('section')}")

    # Check metadata
    missing_meta = [d for d in sample_docs if "section" not in d.metadata]
    if missing_meta:
        logger.warning(f"{len(missing_meta)} docs missing 'section' metadata in sample.")

    logger.info(f"Vector store validation passed (sampled {sample_size} docs).")


def embed_and_store_streaming(
    doc_generator,
    vector_store_path: Path = FAISS_INDEX_PATH,
    batch_save_interval: int = 1,
) -> FAISS:
    """
    Memory-efficient streaming pipeline.

    Processes documents one at a time from a generator:
    1. Load and chunk a single document
    2. Embed its chunks
    3. Merge into vector store
    4. Clear the document from memory
    5. Move to next document

    Args:
        doc_generator: Generator yielding (doc_name, chunks_list) tuples
        vector_store_path: Where to save/load the FAISS index
        batch_save_interval: Save vector store after N documents (1 = save after each doc)

    Returns:
        Final FAISS vector store
    """
    vector_store_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()

    load_existing = vector_store_path.exists()
    vector_store = None
    doc_count = 0
    total_chunks = 0

    for doc_name, chunks in doc_generator:
        if not chunks:
            continue

        logger.info(f"Processing document: {doc_name} ({len(chunks)} chunks)")

        if check_memory_usage() is False:
            logger.warning("Memory pressure detected — pausing to collect garbage.")
            gc.collect()
            time.sleep(2)

        try:
            if vector_store is None:
                if load_existing:
                    logger.info(f"Loading existing vector store from {vector_store_path}...")
                    vector_store = FAISS.load_local(
                        str(vector_store_path),
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info(f"Loaded {len(vector_store.docstore._dict)} existing documents.")
                    load_existing = False

            doc_embeddings = embed_documents(chunks, embeddings)

            if not validate_embeddings(doc_embeddings):
                logger.error(f"Invalid embeddings for {doc_name} — skipping.")
                del chunks, doc_embeddings
                continue

            if vector_store is None:
                vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                )
                logger.info(f"Created initial vector store with {len(chunks)} documents.")
            else:
                new_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                )
                vector_store.merge_from(new_store)
                logger.info(f"Merged {len(chunks)} documents. Total: {len(vector_store.docstore._dict)}")

            total_chunks += len(chunks)
            doc_count += 1

            if doc_count % batch_save_interval == 0:
                vector_store.save_local(str(vector_store_path))
                logger.info(f"[Checkpoint] Saved vector store ({len(vector_store.docstore._dict)} docs total)")

            del chunks, doc_embeddings
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to process {doc_name}: {e}")
            raise

    if vector_store is not None:
        vector_store.save_local(str(vector_store_path))
        logger.info(f"Final vector store saved: {len(vector_store.docstore._dict)} documents, {total_chunks} chunks")
        validate_vector_store(vector_store)
    else:
        logger.error("No documents were successfully processed!")

    return vector_store


def embed_and_store(
    all_docs: List[Document],
    vector_store_path: Path = FAISS_INDEX_PATH,
    merge_existing: bool = True,
) -> FAISS:
    """
    Legacy function for backward compatibility.
    Processes all documents at once (high memory usage).
    For streaming, use embed_and_store_streaming().
    """
    if not all_docs:
        logger.warning("No documents to embed — skipping.")
        return None

    embeddings = get_embedding_model()
    vector_store = build_vector_store(
        all_docs,
        embeddings,
        vector_store_path,
        merge_existing=merge_existing,
    )
    validate_vector_store(vector_store)
    return vector_store



#-----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from parse_3gpp import load_and_chunk_3gpp_docs_streaming

    three_gpp_dir = PROJECT_ROOT / "data" / "raw" / "3gpp_docs"

    # Use streaming pipeline for memory efficiency
    doc_generator = load_and_chunk_3gpp_docs_streaming(three_gpp_dir, glob_pattern="*.docx") # returns generator of (doc_name, chunks_list) tuples
    vector_store = embed_and_store_streaming(
        doc_generator,
        vector_store_path=FAISS_INDEX_PATH,
        batch_save_interval=1,  # Save after each document
    )

    if vector_store:
        logger.info(f"Pipeline complete. Vector store ready at: {FAISS_INDEX_PATH}")