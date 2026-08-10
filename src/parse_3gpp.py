"""
parse_3gpp.py 
"""
#--------------------Import-------------------------------------------------------------
from pathlib import Path # used for file path manipulations
import os
import re # for regex-based cleaning and parsing
from tqdm import tqdm # for progress bars
import argparse # for CLI argument parsing
import logging # for better error reporting
from typing import List, Dict, Any, Tuple # for type annotations
#---------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

#---------------------Used Imports---------------------------------------------
from langchain_docling import DoclingLoader # for loading .docx files (if available)
from langchain_core.documents import Document # for representing chunks as Documents
from langchain_huggingface import HuggingFaceEmbeddings # for embedding the chunks 
from langchain_community.vectorstores import FAISS # for storing the chunks in a vector database
#---------------------------------------------------------------------------------------------

#---------------------Constants & Configurations---------------------------------------------
from src.config import (
    MIN_CHUNK_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
    RAW_3GPP_DIR,
)

DEFAULT_CHUNK_SIZE   = CHUNK_SIZE_WORDS
DEFAULT_CHUNK_OVERLAP = CHUNK_OVERLAP_WORDS
#---------------------------------------------------------------------------------------------

#  STEP 1 — CLEANING
# -----------------
def remove_preface(text: str) -> str:
    """
    Remove boilerplate before 'Definitions, Symbols and Abbreviations'.
    Looks for the SECOND occurrence because the first is in the Table of Contents.
    """
    pattern = (
        r"(?:3\s+Definitions,\s*symbols\s*and\s*abbreviations"
        r"|Definitions,\s*symbols\s*and\s*abbreviations)"
    )
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if len(matches) >= 2:
        return text[matches[1].start():]
    elif len(matches) == 1:
        return text[matches[0].start():]
    return text


def remove_void_sections(text: str) -> str:
    """
    Remove 'Void' placeholder sections that 3GPP uses when a clause was deleted.
    These add noise to the vector store.
    """
    return re.sub(
        r'^\d+(?:\.\d+)*\s+Void\s*$',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )


def normalise_text(text: str) -> str:
    """
    Standard whitespace normalisation.
    """
    text = text.replace('\t', ' ')
    text = re.sub(r' +', ' ', text)                                # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                         # max 2 blank lines
    text = "\n".join(line.strip() for line in text.splitlines())   # strip each line
    return text.strip()


def clean_document(text: str) -> str:
    """
    Master cleaning function — run all cleaning steps in order.
    """
    text = remove_preface(text)
    text = remove_void_sections(text)
    text = normalise_text(text)
    return text
#---------------------------------------------------------------------------------------------

#  STEP 2 — SECTION PARSING

SECTION_PATTERN = re.compile(
    r'^(\d+(?:\.\d+)*)\s{1,4}([A-Za-z].{2,}?)$',
    re.MULTILINE
)


def is_valid_section_title(title: str) -> bool:
    """
    Extra guard to reject false-positive section matches.
    """
    title = title.strip()
    if len(title) < 3:
        return False
    if title.lower() in ('void', 'n/a', 'reserved', 'tbd', 'ffs'):
        return False
    # Reject lines that are mostly numbers/symbols (likely table rows)
    alpha_ratio = sum(c.isalpha() for c in title) / max(len(title), 1)
    if alpha_ratio < 0.4:
        return False
    return True


def find_sections(text: str) -> List[Dict[str, Any]]:
    """
    Find all section headers in the text.
    Returns list of dicts:
        { 'num': '5.3.1', 'title': 'UE behaviour', 'start': <char_pos>, 'end': <header_end> }
    """
    sections = []
    for match in SECTION_PATTERN.finditer(text):
        num = match.group(1).strip()
        title = match.group(2).strip()
        if is_valid_section_title(title):
            sections.append({
                'num':   num,
                'title': title,
                'start': match.start(),
                'end':   match.end(),
            })
    return sections
#---------------------------------------------------------------------------------------------

# STEP 3 — LENGTH-BASED CHUNKING


def _section_at_char_pos(
    sections: List[Dict[str, Any]],
    char_pos: int,
) -> Dict[str, str]:
    """Return the section header active at a character position in the document."""
    current = {"num": "0", "title": "Full Document"}
    for section in sections:
        if section["start"] <= char_pos:
            current = {"num": section["num"], "title": section["title"]}
        else:
            break
    return current


def _word_char_spans(text: str) -> List[Tuple[int, int]]:
    """Map each whitespace-delimited word to (start, end) char offsets in text."""
    spans: List[Tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        spans.append((match.start(), match.end()))
    return spans


def _section_spans(full_text: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return document spans split by section headers, including preface if present."""
    if not sections:
        return [{
            "num": "0",
            "title": "Full Document",
            "start": 0,
            "end": len(full_text),
        }]

    spans: List[Dict[str, Any]] = []
    first = sections[0]
    if first["start"] > 0 and full_text[: first["start"]].strip():
        spans.append({
            "num": "0",
            "title": "Preamble",
            "start": 0,
            "end": first["start"],
        })

    for idx, section in enumerate(sections):
        end = sections[idx + 1]["start"] if idx + 1 < len(sections) else len(full_text)
        spans.append({
            "num": section["num"],
            "title": section["title"],
            "start": section["start"],
            "end": end,
        })

    return spans


def _chunk_section(
    section_text: str,
    section_meta: Dict[str, Any],
    chunk_size_words: int,
    overlap_words: int,
    min_chunk_words: int,
    start_index: int,
) -> List[Document]:
    """Chunk text within a section span, keeping section metadata intact."""
    word_spans = _word_char_spans(section_text)
    total_words = len(word_spans)
    if total_words < min_chunk_words:
        return []

    if total_words <= chunk_size_words:
        return [Document(
            page_content=section_text.strip(),
            metadata={
                "section": section_meta["num"],
                "section_title": section_meta["title"],
                "word_count": total_words,
                "chunk_type": "section_aware",
                "chunk_index": start_index,
            },
        )]

    step = max(1, chunk_size_words - overlap_words)
    chunks: List[Dict[str, Any]] = []
    chunk_index = start_index
    start_word = 0

    while start_word < total_words:
        end_word = min(start_word + chunk_size_words, total_words)
        char_start = word_spans[start_word][0]
        char_end = word_spans[end_word - 1][1]
        chunk_text = section_text[char_start:char_end].strip()
        word_count = len(chunk_text.split())

        if end_word == total_words:
            if word_count < min_chunk_words and chunks:
                # Merge the small tail into the previous chunk.
                prev = chunks[-1]
                prev_text = prev["page_content"]
                merged_text = f"{prev_text}\n\n{chunk_text}".strip()
                prev["page_content"] = merged_text
                prev["word_count"] = len(merged_text.split())
            else:
                chunks.append({
                    "page_content": chunk_text,
                    "word_count": word_count,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
            break

        if word_count < min_chunk_words:
            if chunks:
                prev = chunks[-1]
                prev_text = prev["page_content"]
                merged_text = f"{prev_text}\n\n{chunk_text}".strip()
                prev["page_content"] = merged_text
                prev["word_count"] = len(merged_text.split())
            break

        chunks.append({
            "page_content": chunk_text,
            "word_count": word_count,
            "chunk_index": chunk_index,
        })
        chunk_index += 1
        start_word += step

    return [Document(
        page_content=chunk["page_content"],
        metadata={
            "section": section_meta["num"],
            "section_title": section_meta["title"],
            "word_count": chunk["word_count"],
            "chunk_type": "section_aware",
            "chunk_index": chunk["chunk_index"],
        },
    ) for chunk in chunks]


def chunk_by_length(
    full_text: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> List[Document]:
    """
    Split cleaned document text into section-aware chunks.
    Each section is chunked independently, preserving section boundaries
    and merging short tail fragments into the previous chunk.
    """
    if not full_text.strip():
        return []

    sections = find_sections(full_text)
    section_spans = _section_spans(full_text, sections)
    final_docs: List[Document] = []
    chunk_index = 0

    for section in section_spans:
        section_text = full_text[section["start"]:section["end"]].strip()
        if not section_text:
            continue

        section_chunks = _chunk_section(
            section_text,
            section,
            chunk_size_words,
            overlap_words,
            min_chunk_words,
            chunk_index,
        )

        if not section_chunks:
            # Keep short sections by merging them into the previous chunk when possible.
            if final_docs:
                prev = final_docs[-1]
                merged_text = f"{prev.page_content}\n\n{section_text}".strip()
                prev.metadata["word_count"] = len(merged_text.split())
                prev.page_content = merged_text
            else:
                final_docs.append(Document(
                    page_content=section_text,
                    metadata={
                        "section": section["num"],
                        "section_title": section["title"],
                        "word_count": len(section_text.split()),
                        "chunk_type": "section_aware",
                        "chunk_index": chunk_index,
                    },
                ))
                chunk_index += 1
        else:
            for chunk in section_chunks:
                final_docs.append(chunk)
                chunk_index += 1

    logger.info(
        "  -> %s section-aware chunks (size=%s, overlap=%s)",
        len(final_docs), chunk_size_words, overlap_words,
    )
    return final_docs
#---------------------------------------------------------------------------------------------

#  STEP 4 — DOCUMENT LOADING & ORCHESTRATION

#-------------------------Load & chunk documents one at a time (streaming)----------------
def load_and_chunk_3gpp_docs_streaming(
    three_gpp_dir: Path,
    glob_pattern: str = "*.docx",
    chunk_size_words: int = DEFAULT_CHUNK_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
):
    
    doc_files = sorted(list(three_gpp_dir.glob(glob_pattern)))
    if not doc_files:
        logger.warning("[WARNING] No files found in %s matching '%s'", three_gpp_dir, glob_pattern)
        return

    total_chunks = 0
    for doc_idx, doc_file in enumerate(doc_files, 1):
        loader = DoclingLoader(str(doc_file))
        raw_pages = list(loader.load())
        
        if not raw_pages:
            logger.warning("[SKIP] %s - loader returned no content", doc_file.name)
            continue
        
        # Verify we have content in each page
        pages_with_content = [p for p in raw_pages if p.page_content]
        pages_without_content = len(raw_pages) - len(pages_with_content)
        
        if pages_without_content > 0:
            logger.warning("%s pages have no content", pages_without_content)

        page_contents = []
        for page_num, page in enumerate(raw_pages, 1):
            if page.page_content:
                page_contents.append(page.page_content)
        
        full_text = "\n\n".join(page_contents)
        
        logger.info("  Full document length: %s characters", f"{len(full_text):,}")
        logger.info("  Pages with content: %s/%s", len(page_contents), len(raw_pages))
        
        full_text = clean_document(full_text)
        logger.info("After cleaning: %s chars", f"{len(full_text):,}")

        chunks = chunk_by_length(
            full_text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        for chunk in chunks:
            chunk_index = chunk.metadata.get("chunk_index", 0)
            section = chunk.metadata.get("section", "0")
            chunk.metadata.update({
                'source':   str(doc_file),
                'doc_name': doc_file.name,
                'doc_type': '3gpp_spec',
                'doc_id':   f"{doc_file.stem}_{section}_{chunk_index}",
            })

        logger.info("    Created %s chunks", len(chunks))
        total_chunks += len(chunks)

        del full_text, raw_pages
        yield doc_file.name, chunks


    logger.info("Total 3GPP chunks generated: %s", total_chunks)


# ------------------------Load all the docs and chunk them at once (legacy)----------------
def load_and_chunk_3gpp_docs(
    three_gpp_dir: Path,
    glob_pattern: str = "*.docx",
    chunk_size_words: int = DEFAULT_CHUNK_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    
    all_docs: List[Document] = []
    for doc_name, chunks in load_and_chunk_3gpp_docs_streaming(
        three_gpp_dir, glob_pattern, chunk_size_words, overlap_words
    ):
        all_docs.extend(chunks)
    return all_docs
#---------------------------------------------------------------------------

#  STEP 5 — QUICK VALIDATION

def validate_chunks(docs: List[Document], sample_size: int = 10) -> None:
    if not docs:
        logger.info("[VALIDATE] No documents to validate.")
        return

    word_counts = [d.metadata.get('word_count', len(d.page_content.split())) for d in docs]
    chunk_types = {}
    for d in docs:
        ct = d.metadata.get('chunk_type', 'unknown')
        chunk_types[ct] = chunk_types.get(ct, 0) + 1

    logger.info("Chunk Validation Report")
    logger.info("Total chunks      : %s", len(docs))
    logger.info("Min words/chunk   : %s", min(word_counts))
    logger.info("Max words/chunk   : %s", max(word_counts))
    logger.info("Avg words/chunk   : %.0f", sum(word_counts) / len(word_counts))
    for ct, count in sorted(chunk_types.items(), key=lambda x: -x[1]):
        logger.info("  %s: %s", ct, count)

    tiny = [d for d in docs if len(d.page_content.split()) < MIN_CHUNK_WORDS]
    if tiny:
        logger.warning("%s chunks below %s words", len(tiny), MIN_CHUNK_WORDS)

    missing_section = [d for d in docs if 'section' not in d.metadata]
    if missing_section:
        logger.warning("%s chunks missing 'section' metadata", len(missing_section))

    import random
    for d in random.sample(docs, min(sample_size, len(docs))):
        preview = d.page_content[:200].replace('\n', ' ')
        logger.info(
            "Sample chunk section=%s words=%s type=%s preview=%s...",
            d.metadata.get('section', 'N/A'),
            d.metadata.get('word_count', '?'),
            d.metadata.get('chunk_type', '?'),
            preview,
        )




# ----------------------------- Main Fxn & CLI ------------------------------------------------
def main() -> int:
    three_gpp_dir = RAW_3GPP_DIR
    if not three_gpp_dir.exists():
        print(f"[ERROR] Directory not found: {three_gpp_dir}")
        return 2

    docs = load_and_chunk_3gpp_docs(three_gpp_dir, glob_pattern="*.docx")
    validate_chunks(docs)

    return 0

#-------------Main Guard ------------------------------------------------------------
if __name__ == '__main__':
    raise SystemExit(main())
#----------------------------------------------------------------------------