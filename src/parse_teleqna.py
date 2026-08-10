"""
TeleQnA parser — produces:
  1. Retrieval Documents (LangChain Document objects with full metadata)
  2. Instruction-response pairs for QLoRA fine-tuning
  3. A deterministic train/test split for honest evaluation

TeleQnA records are multiple-choice. We preserve the option list and the
correct-option index so the evaluator can score true MCQ accuracy, and we
expose a train/test split so held-out test questions can be EXCLUDED from the
retrieval corpus (preventing trivial self-retrieval during evaluation).
"""
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from src.config import (
    EVAL_SEED,
    EVAL_TEST_RATIO,
    RAW_TELEQNA_DIR,
)

logger = logging.getLogger(__name__)

ALT_TELEQNA_DIR = RAW_TELEQNA_DIR.parent / "teleqna"


def _extract_options(raw: Dict) -> List[str]:
    """Collect 'option 1'..'option N' values in numeric order."""
    numbered: List[Tuple[int, str]] = []
    for key, value in raw.items():
        m = re.match(r"^option[_ ]?(\d+)$", str(key), re.IGNORECASE)
        if m and str(value).strip():
            numbered.append((int(m.group(1)), str(value).strip()))
    numbered.sort(key=lambda x: x[0])
    return [text for _, text in numbered]


def _normalize_record(raw: Dict, record_id: str) -> Dict:
    """Normalize a TeleQnA record across JSON variants.

    Returns keys: question_id, question, answer (correct option text),
    answer_index (1-based, 0 if unknown), options (list), explanation,
    category, source_doc.
    """
    question = str(raw.get("question", "")).strip()
    category = str(raw.get("category", raw.get("topic", "general"))).strip()
    source_doc = str(raw.get("source", raw.get("reference", ""))).strip()
    question_id = str(raw.get("id", raw.get("question_id", record_id))).strip()
    explanation = str(raw.get("explanation", "")).strip()

    options = _extract_options(raw)

    answer_raw = str(raw.get("answer", raw.get("correct_option", ""))).strip()

    # Parse "option N: <text>" → index + text
    answer_index = 0
    answer_text = answer_raw
    m = re.match(r"^option[_ ]?(\d+)\s*[:.\-]?\s*(.*)$", answer_raw, re.IGNORECASE | re.DOTALL)
    if m:
        answer_index = int(m.group(1))
        answer_text = m.group(2).strip()
        # If the answer field only carried the index, recover text from options.
        if not answer_text and 1 <= answer_index <= len(options):
            answer_text = options[answer_index - 1]

    # If we have the text but not the index, locate it among the options.
    if answer_index == 0 and answer_text and options:
        for i, opt in enumerate(options, start=1):
            if opt.strip() == answer_text.strip():
                answer_index = i
                break

    return {
        "question_id": question_id,
        "question": question,
        "answer": answer_text,
        "answer_index": answer_index,
        "options": options,
        "explanation": explanation,
        "category": category,
        "source_doc": source_doc,
    }


def _iter_qa_file(qa_path: Path) -> Iterator[Tuple[Dict, str, str]]:
    """Yield raw Q&A records from a single JSON/JSONL/TXT file."""
    suffix = qa_path.suffix.lower()

    if suffix in {".json", ".txt"}:
        with open(qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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
        for qa_file in sorted(teleqna_dir.rglob("*")):
            if qa_file.suffix.lower() in {".json", ".jsonl", ".txt"}:
                found = True
                yield from _iter_qa_file(qa_file)

    if not found and teleqna_dir != ALT_TELEQNA_DIR and ALT_TELEQNA_DIR.exists():
        logger.info(
            f"No TeleQnA records found in {teleqna_dir}; falling back to {ALT_TELEQNA_DIR}."
        )
        yield from _iter_qa_records(ALT_TELEQNA_DIR)


def _load_normalized_records(teleqna_dir: Path = RAW_TELEQNA_DIR) -> List[Dict]:
    """Load and normalize all valid TeleQnA records (question + answer present)."""
    records: List[Dict] = []
    for idx, (item, src_file, record_id) in enumerate(_iter_qa_records(teleqna_dir)):
        record = _normalize_record(item, record_id or str(idx))
        if not record["question"] or not record["answer"]:
            continue
        record["doc_id"] = f"teleqna_{record['question_id']}"
        record["source"] = src_file
        records.append(record)
    return records


def _deterministic_split(
    records: List[Dict],
    test_ratio: float = EVAL_TEST_RATIO,
    seed: int = EVAL_SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle with a fixed seed and split into (train, test)."""
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def _record_to_document(record: Dict) -> Document:
    """Build a retrieval Document from a normalized record."""
    answer = record["answer"]
    if record["explanation"] and record["explanation"] not in answer:
        answer = f"{answer}\nExplanation: {record['explanation']}"
    page_content = f"Question: {record['question']}\nAnswer: {answer}"
    return Document(
        page_content=page_content,
        metadata={
            "question_id": record["question_id"],
            "category": record["category"],
            "source_doc": record["source_doc"],
            "doc_type": "teleqna",
            "doc_id": record["doc_id"],
            "source": record["source"],
            "word_count": len(page_content.split()),
            "section": record["question_id"],
            "section_title": record["category"],
        },
    )


def load_teleqna_documents(
    teleqna_dir: Path = RAW_TELEQNA_DIR,
    split: Optional[str] = None,
    test_ratio: float = EVAL_TEST_RATIO,
    seed: int = EVAL_SEED,
) -> List[Document]:
    """
    Parse TeleQnA files into LangChain Document objects for retrieval.

    Args:
        split: None → all records; "train" → training split only (used to build
               the retrieval corpus so test questions are not self-retrievable);
               "test" → held-out split only.
    """
    records = _load_normalized_records(teleqna_dir)
    if split in {"train", "test"}:
        train, test = _deterministic_split(records, test_ratio, seed)
        records = train if split == "train" else test

    docs = [_record_to_document(r) for r in records]
    logger.info(
        f"Loaded {len(docs)} TeleQnA retrieval documents from {teleqna_dir} "
        f"(split={split or 'all'})"
    )
    return docs


def load_teleqna_finetune_pairs(
    teleqna_dir: Path = RAW_TELEQNA_DIR,
    split: Optional[str] = "train",
) -> List[Dict[str, str]]:
    """Parse TeleQnA into instruction-response pairs for QLoRA fine-tuning.

    Defaults to the train split to avoid leaking held-out eval questions.
    """
    records = _load_normalized_records(teleqna_dir)
    if split in {"train", "test"}:
        train, test = _deterministic_split(records, EVAL_TEST_RATIO, EVAL_SEED)
        records = train if split == "train" else test

    pairs: List[Dict[str, str]] = []
    for r in records:
        instruction = r["question"]
        if r["options"]:
            opts = "\n".join(f"{i}. {opt}" for i, opt in enumerate(r["options"], start=1))
            instruction = f"{r['question']}\n\nOptions:\n{opts}"
        response = r["answer"]
        if r["explanation"]:
            response = f"{response}\n\nExplanation: {r['explanation']}"
        pairs.append({"instruction": instruction, "response": response})

    logger.info(f"Loaded {len(pairs)} TeleQnA fine-tuning pairs (split={split or 'all'})")
    return pairs


def load_teleqna_eval_split(
    teleqna_dir: Path = RAW_TELEQNA_DIR,
    test_ratio: float = EVAL_TEST_RATIO,
    seed: int = EVAL_SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split TeleQnA into train/test records for evaluation.

    Each record carries: question_id, question, answer, answer_index, options,
    explanation, category, doc_id, reference_content, source.
    """
    records = _load_normalized_records(teleqna_dir)
    for r in records:
        r["reference_content"] = f"Question: {r['question']}\nAnswer: {r['answer']}"

    train_records, test_records = _deterministic_split(records, test_ratio, seed)
    logger.info(
        f"TeleQnA split: {len(train_records)} train / {len(test_records)} test "
        f"(ratio={test_ratio}, seed={seed})"
    )
    return train_records, test_records


def parse_teleqna(
    teleqna_dir: Path = RAW_TELEQNA_DIR,
) -> Tuple[List[Document], List[Dict[str, str]]]:
    """Convenience wrapper returning both retrieval docs and fine-tuning pairs."""
    return load_teleqna_documents(teleqna_dir), load_teleqna_finetune_pairs(teleqna_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs, pairs = parse_teleqna()
    train, test = load_teleqna_eval_split()
    print(f"Retrieval docs : {len(docs)}")
    print(f"Fine-tune pairs: {len(pairs)}")
    print(f"Eval split     : {len(train)} train / {len(test)} test")
    if test:
        r = test[0]
        print(f"\nSample test record:")
        print(f"  question     : {r['question'][:80]}")
        print(f"  options      : {len(r['options'])}")
        print(f"  answer_index : {r['answer_index']}")
        print(f"  answer       : {r['answer'][:80]}")
