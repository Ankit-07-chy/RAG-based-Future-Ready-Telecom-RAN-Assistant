"""
TeleQnA parser — produces two outputs:
  1. Retrieval Documents (LangChain Document objects with full metadata)
  2. Instruction-response pairs for QLoRA fine-tuning
"""
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TELEQNA_DIR = PROJECT_ROOT / "data" / "raw" / "teleqna_dataset"
ALT_TELEQNA_DIR = PROJECT_ROOT / "data" / "raw" / "teleqna"


def _iter_qa_file(qa_path: Path) -> Iterator[Tuple[Dict, str, str]]:
    """Yield raw Q&A records from a single JSON/JSONL/TXT file."""
    suffix = qa_path.suffix.lower()

    if suffix in {".json", ".txt"}:
        with open(qa_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                if suffix == ".txt":
                    f.seek(0)
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line), str(qa_path), f"{qa_path.stem}_{line_idx}"
                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed to parse line {line_idx} in {qa_path} as JSON.")
                            raise
                    return
                raise

        if isinstance(data, list):
            for idx, item in enumerate(data):
                yield item, str(qa_path), f"{qa_path.stem}_{idx}"
        elif isinstance(data, dict):
            for key, item in data.items():
                yield item, str(qa_path), str(key)
        else:
            logger.warning(f"Unsupported JSON structure in {qa_path}: {type(data).__name__}")

    elif suffix == ".jsonl":
        with open(qa_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line), str(qa_path), f"{qa_path.stem}_{idx}"


def _iter_qa_records(teleqna_dir: Path) -> Iterator[Tuple[Dict, str, str]]:
    """Yield raw Q&A dicts from all JSON/JSONL/TXT files in teleqna_dir."""
    if teleqna_dir.is_file():
        yield from _iter_qa_file(teleqna_dir)
        return

    found = False
    if teleqna_dir.exists():
        for qa_file in teleqna_dir.rglob("*"):
            if qa_file.suffix.lower() in {".json", ".jsonl", ".txt"}:
                found = True
                yield from _iter_qa_file(qa_file)

    if not found and teleqna_dir != ALT_TELEQNA_DIR and ALT_TELEQNA_DIR.exists():
        logger.info(
            f"No TeleQnA records found in {teleqna_dir}; falling back to {ALT_TELEQNA_DIR}.")
        yield from _iter_qa_records(ALT_TELEQNA_DIR)
        return

    if not found:
        logger.warning(
            f"No JSON/JSONL/TXT files found in {teleqna_dir}. Returning empty.")


def load_teleqna_documents(
    teleqna_dir: Path = DEFAULT_TELEQNA_DIR,
) -> List[Document]:
    """
    Parse TeleQnA files into LangChain Document objects for retrieval.

    Each document's page_content = "Question: ...\nAnswer: ..."
    Metadata keys: question_id, category, source_doc, doc_type, word_count.

    Returns:
        List of Document objects.
    """
    docs: List[Document] = []
    for idx, (item, src_file, record_id) in enumerate(_iter_qa_records(teleqna_dir)):
        question   = str(item.get("question", "")).strip()
        answer     = str(item.get("answer", item.get("correct_option", ""))).strip()
        category   = str(item.get("category", item.get("topic", "general"))).strip()
        source_doc = str(item.get("source", item.get("reference", ""))).strip()
        question_id = str(item.get("id", item.get("question_id", record_id or idx)))

        if not question or not answer:
            continue

        page_content = f"Question: {question}\nAnswer: {answer}"
        word_count = len(page_content.split())

        docs.append(Document(
            page_content=page_content,
            metadata={
                "question_id": question_id,
                "category":    category,
                "source_doc":  source_doc,
                "doc_type":    "teleqna",
                "source":      src_file,
                "word_count":  word_count,
            },
        ))

    logger.info(f"Loaded {len(docs)} TeleQnA retrieval documents from {teleqna_dir}")
    return docs


def load_teleqna_finetune_pairs(
    teleqna_dir: Path = DEFAULT_TELEQNA_DIR,
) -> List[Dict[str, str]]:
    """
    Parse TeleQnA files into instruction-response pairs for QLoRA fine-tuning.

    Returns:
        List of {"instruction": str, "response": str} dicts.
    """
    pairs: List[Dict[str, str]] = []
    for _idx, (item, _src, _record_id) in enumerate(_iter_qa_records(teleqna_dir)):
        question = str(item.get("question", "")).strip()
        answer   = str(item.get("answer", item.get("correct_option", ""))).strip()
        if not question or not answer:
            continue
        pairs.append({"instruction": question, "response": answer})

    logger.info(f"Loaded {len(pairs)} TeleQnA fine-tuning pairs from {teleqna_dir}")
    return pairs


def parse_teleqna(
    teleqna_dir: Path = DEFAULT_TELEQNA_DIR,
) -> Tuple[List[Document], List[Dict[str, str]]]:
    """
    Convenience wrapper returning both outputs at once.

    Returns:
        (retrieval_docs, finetune_pairs)
    """
    return load_teleqna_documents(teleqna_dir), load_teleqna_finetune_pairs(teleqna_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs, pairs = parse_teleqna()
    print(f"Retrieval docs : {len(docs)}")
    print(f"Fine-tune pairs: {len(pairs)}")
    if docs:
        d = docs[0]
        print(f"\nSample doc metadata : {d.metadata}")
        print(f"Sample doc content  : {d.page_content[:200]}")
    if pairs:
        print(f"\nSample pair: {pairs[0]}")
