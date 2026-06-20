#!/usr/bin/env python3
"""
Extract and process TeleQnA dataset.
Handles protected ZIP extraction and JSON parsing.
"""
# The code is Provided BY EnnovateX Samsung Hackathon

import json
import zipfile
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import jsonlines

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_TELEQNA_DIR = DATA_DIR / "raw" / "teleqna"
PROCESSED_DIR = DATA_DIR / "processed"

TELEQNA_ZIP = RAW_TELEQNA_DIR / "TeleQnA.zip"
TELEQNA_PASSWORD = "teleqnadataset"


def extract_teleqna_zip(zip_path: Path, password: str, extract_to: Path) -> Path:
    """
    Extract password-protected TeleQnA ZIP file.

    Args:
        zip_path: Path to .zip file
        password: ZIP password
        extract_to: Directory to extract to

    Returns:
        Path to extracted directory
    """
    if not zip_path.exists():
        logger.error(f"ZIP file not found: {zip_path}")
        raise FileNotFoundError(f"TeleQnA ZIP not found at {zip_path}")

    logger.info(f"Extracting {zip_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to, pwd=password.encode())
        logger.info(f"✅ Extracted to {extract_to}")
        return extract_to
    except zipfile.BadZipFile:
        logger.error(f"Invalid or corrupted ZIP file: {zip_path}")
        raise
    except RuntimeError as e:
        logger.error(f"Failed to extract (wrong password?): {e}")
        raise


def parse_teleqna_json(json_path: Path) -> List[Dict]:
    """
    Parse TeleQnA JSON file.
    Expected format: {
        "question_id": {
            "question": "...",
            "option_1": "...",
            "option_2": "...",
            "option_3": "...",
            "option_4": "...",
            "answer": "option_3: ...",
            "explanation": "...",
            "category": "Standards specifications"
        }
    }
    """
    logger.info(f"Parsing {json_path.name}...")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    for q_id, q_data in data.items():
        questions.append({
            "q_id": q_id,
            "question": q_data.get("question", ""),
            "options": [
                q_data.get(f"option_{i}", "") for i in range(1, 5)
            ],
            "answer": q_data.get("answer", ""),
            "explanation": q_data.get("explanation", ""),
            "category": q_data.get("category", ""),
        })

    logger.info(f"Parsed {len(questions)} questions from {json_path.name}")
    return questions


def save_teleqna_splits(
    questions: List[Dict],
    train_output: Path,
    test_output: Path,
    train_ratio: float = 0.8
):
    """
    Split questions into train and test sets, save as JSONL.
    """
    # Shuffle and split
    import random
    random.shuffle(questions)

    split_idx = int(len(questions) * train_ratio)
    train_questions = questions[:split_idx]
    test_questions = questions[split_idx:]

    logger.info(f"Splitting: {len(train_questions)} train, {len(test_questions)} test")

    # Save train set
    with jsonlines.open(train_output, 'w') as writer:
        for q in tqdm(train_questions, desc="Writing train set"):
            writer.write(q)
    logger.info(f"✅ Saved training set: {train_output} ({len(train_questions)} Q&As)")

    # Save test set
    with jsonlines.open(test_output, 'w') as writer:
        for q in tqdm(test_questions, desc="Writing test set"):
            writer.write(q)
    logger.info(f"✅ Saved test set: {test_output} ({len(test_questions)} Q&As)")


def create_demo_teleqna():
    """
    Create a small demo TeleQnA dataset for testing.
    Useful when actual ZIP is not available.
    """
    demo_data = [
        {
            "q_id": "demo_1",
            "question": "What is MIMO in 5G NR?",
            "options": [
                "Multiple-Input Multiple-Output",
                "Multiple Integrated Mobile Operations",
                "Multi-Iteration Multiple Operations",
                "None of the above"
            ],
            "answer": "option_1: Multiple-Input Multiple-Output",
            "explanation": "MIMO refers to Multiple-Input Multiple-Output antenna systems used in 5G to enhance spectral efficiency.",
            "category": "Standards specifications"
        },
        {
            "q_id": "demo_2",
            "question": "What does PRACH stand for?",
            "options": [
                "Physical Random Access Channel",
                "Protocol Random Access Code",
                "Physical Routing Access Channel",
                "Procedure Routing Access Code"
            ],
            "answer": "option_1: Physical Random Access Channel",
            "explanation": "PRACH is the Physical Random Access Channel used by UEs to request access to the network.",
            "category": "Standards specifications"
        },
    ]

    demo_output = PROCESSED_DIR / "teleqna_demo.jsonl"
    with jsonlines.open(demo_output, 'w') as writer:
        for q in demo_data:
            writer.write(q)

    logger.info(f"✅ Created demo dataset: {demo_output}")
    return demo_output


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("TELEQNA DATASET EXTRACTION & PROCESSING")
    logger.info("="*70)

    # Try to extract real ZIP
    if TELEQNA_ZIP.exists():
        try:
            extract_dir = extract_teleqna_zip(TELEQNA_ZIP, TELEQNA_PASSWORD, PROCESSED_DIR / "teleqna_extracted")

            # Find JSON files in extracted directory
            json_files = list(extract_dir.glob("**/*.json"))
            if json_files:
                logger.info(f"Found {len(json_files)} JSON files")

                all_questions = []
                for jf in json_files:
                    questions = parse_teleqna_json(jf)
                    all_questions.extend(questions)

                logger.info(f"Total: {len(all_questions)} questions from all files")

                # Save splits
                save_teleqna_splits(
                    all_questions,
                    PROCESSED_DIR / "teleqna_train.jsonl",
                    PROCESSED_DIR / "teleqna_test.jsonl"
                )
            else:
                logger.warning("No JSON files found in extracted ZIP")
                create_demo_teleqna()
        except Exception as e:
            logger.error(f"Failed to extract ZIP: {e}")
            logger.info("Creating demo dataset instead...")
            create_demo_teleqna()
    else:
        logger.warning(f"TeleQnA ZIP not found at {TELEQNA_ZIP}")
        logger.info("Creating demo dataset for testing...")
        create_demo_teleqna()

    logger.info("\n" + "="*70)
    logger.info("✅ TeleQnA extraction complete")
    logger.info("="*70)


if __name__ == "__main__":
    main()
