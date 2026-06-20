"""
parse_3gpp.py 
"""
#--------------------Import-------------------------------------------------------------
from pathlib import Path # used for file path manipulations
import re # for regex-based cleaning and parsing
from tqdm import tqdm # for progress bars
import argparse # for CLI argument parsing
import logging # for better error reporting
from types import SimpleNamespace  # used for dot notation for simple objects
from dataclasses import dataclass, field # used for autogeneration of few methods inside class
from typing import List, Dict, Any # for type annotations
#---------------------------------------------------------------------------------------------


#---------------------Used Imports---------------------------------------------
from langchain_docling import DoclingLoader # for loading .docx files (if available)
from langchain_core.documents import Document # for representing chunks as Documents
from langchain_huggingface import HuggingFaceEmbeddings # for embedding the chunks 
from langchain_community.vectorstores import FAISS # for storing the chunks in a vector database
#---------------------------------------------------------------------------------------------

#---------------------Constants & Configurations---------------------------------------------
MIN_CHUNK_WORDS      = 50    # chunks below this are discarded as noise
DEFAULT_CHUNK_SIZE   = 200   # target min words per chunk
PARENT_CONTEXT_LINES = 3     # how many lines of parent section to prepend to child chunks
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
        { 'num': '5.3.1', 'title': 'UE behaviour', 'start': <char_pos> }
    """
    sections = []
    for match in SECTION_PATTERN.finditer(text):
        num   = match.group(1).strip()
        title = match.group(2).strip()
        if is_valid_section_title(title):
            sections.append({
                'num':   num,
                'title': title,
                'start': match.start()
            })
    return sections
#---------------------------------------------------------------------------------------------

# STEP 3 — HIERARCHICAL CHUNKING


def get_parent_context(section_num: str, all_sections: List[Dict[str, Any]], full_text: str) -> str:
    parts = section_num.split('.')
    if len(parts) <= 1:
        return ''  # top-level section has no parent

    parent_num = '.'.join(parts[:-1])

    parent = next((s for s in all_sections if s['num'] == parent_num), None)
    if not parent:
        return ''

    parent_idx = all_sections.index(parent)
    parent_start = parent['start']
    if parent_idx + 1 < len(all_sections):
        parent_end = all_sections[parent_idx + 1]['start']
    else:
        parent_end = len(full_text)

    parent_content = full_text[parent_start:parent_end].strip()
    parent_lines   = [l for l in parent_content.splitlines() if l.strip()]

    if not parent_lines:
        return ''

    context_lines = parent_lines[:PARENT_CONTEXT_LINES]
    return f"[Parent: §{parent_num} — {parent['title']}\n]".replace('\n]', '\n') + "\n".join(context_lines)


def split_into_section_chunks(full_text: str) -> List[Dict[str, Any]]:
    sections = find_sections(full_text)
    if not sections:
        return [{'num': '0', 'title': 'Full Document', 'content': full_text}]

    raw_chunks = []
    for i, section in enumerate(sections):
        start = section['start']
        end   = sections[i + 1]['start'] if i + 1 < len(sections) else len(full_text)

        content = full_text[start:end].strip()
        if content:
            raw_chunks.append({
                'num':     section['num'],
                'title':   section['title'],
                'content': content
            })

    return raw_chunks


def get_section_depth(section_num: str) -> int:
    return len(section_num.split('.'))


def assemble_chunk(parent_context: str, body: str) -> str:
    if parent_context:
        return f"{parent_context}\n\n{body}"
    return body


def chunk_large_section_by_paragraphs(
    content: str,
    section_num: str,
    section_title: str,
    parent_context: str,
    min_chunk_size: int
) -> List[Document]:
    paragraphs    = [p.strip() for p in content.split('\n\n') if p.strip()]
    chunks        = []
    current_paras = []
    current_words = 0
    part_num      = 1

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > min_chunk_size and current_paras:
            chunk_text = assemble_chunk(
                parent_context,
                '\n\n'.join(current_paras)
            )
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    'section':       section_num,
                    'section_title': section_title,
                    'part':          part_num,
                    'word_count':    len(chunk_text.split()),
                    'chunk_type':    'paragraph_split'
                }
            ))
            current_paras = [para]
            current_words = para_words
            part_num     += 1
        else:
            current_paras.append(para)
            current_words += para_words

    if current_paras:
        chunk_text = assemble_chunk(parent_context, '\n\n'.join(current_paras))
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                'section':       section_num,
                'section_title': section_title,
                'part':          part_num,
                'word_count':    len(chunk_text.split()),
                'chunk_type':    'paragraph_split'
            }
        ))

    return chunks


def chunk_with_hierarchy(
    full_text: str,
    min_chunk_size: int = DEFAULT_CHUNK_SIZE
) -> List[Document]:
    all_sections = find_sections(full_text)
    raw_chunks   = split_into_section_chunks(full_text)
    final_docs   = []

    for raw in raw_chunks:
        section_num   = raw['num']
        section_title = raw['title']
        content       = raw['content']
        word_count    = len(content.split())

        parent_context = get_parent_context(section_num, all_sections, full_text)

        if word_count <= min_chunk_size:
            chunk_text = assemble_chunk(parent_context, content)
            if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
                final_docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        'section':       section_num,
                        'section_title': section_title,
                        'word_count':    len(chunk_text.split()),
                        'chunk_type':    'full_section'
                    }
                ))
            continue

        current_depth    = get_section_depth(section_num)
        subsec_pattern   = re.compile(
            rf'^({re.escape(section_num)}\.\d+)\s{{1,4}}([A-Za-z].{{2,}}?)$',
            re.MULTILINE
        )
        subsec_matches   = list(subsec_pattern.finditer(content))

        if subsec_matches:
            split_positions = [m.start() for m in subsec_matches] + [len(content)]

            intro = content[:split_positions[0]].strip()
            if intro and len(intro.split()) >= MIN_CHUNK_WORDS:
                chunk_text = assemble_chunk(parent_context, intro)
                final_docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        'section':       section_num,
                        'section_title': section_title,
                        'subsection':    'intro',
                        'word_count':    len(chunk_text.split()),
                        'chunk_type':    'subsection_intro'
                    }
                ))

            for i, match in enumerate(subsec_matches):
                sub_start = split_positions[i]
                sub_end   = split_positions[i + 1]
                sub_text  = content[sub_start:sub_end].strip()

                real_sub_num   = match.group(1).strip()
                real_sub_title = match.group(2).strip()

                if not sub_text or len(sub_text.split()) < MIN_CHUNK_WORDS:
                    continue

                sub_parent_ctx = get_parent_context(real_sub_num, all_sections, full_text)
                sub_word_count = len(sub_text.split())

                if sub_word_count <= min_chunk_size:
                    chunk_text = assemble_chunk(sub_parent_ctx, sub_text)
                    final_docs.append(Document(
                        page_content=chunk_text,
                        metadata={
                            'section':       real_sub_num,
                            'section_title': real_sub_title,
                            'word_count':    len(chunk_text.split()),
                            'chunk_type':    'subsection'
                        }
                    ))
                else:
                    para_chunks = chunk_large_section_by_paragraphs(
                        sub_text, real_sub_num, real_sub_title,
                        sub_parent_ctx, min_chunk_size
                    )
                    final_docs.extend(para_chunks)

        else:
            para_chunks = chunk_large_section_by_paragraphs(
                content, section_num, section_title,
                parent_context, min_chunk_size
            )
            final_docs.extend(para_chunks)

    print(f"  → {len(final_docs)} chunks after filtering (min {MIN_CHUNK_WORDS} words)")
    return final_docs
#---------------------------------------------------------------------------------------------

#  STEP 4 — DOCUMENT LOADING & ORCHESTRATION

#-------------------------Load & chunk documents one at a time (streaming)----------------
def load_and_chunk_3gpp_docs_streaming(
    three_gpp_dir: Path,
    glob_pattern: str = "*.docx",
    min_chunk_size: int = DEFAULT_CHUNK_SIZE
):
    
    doc_files = sorted(list(three_gpp_dir.glob(glob_pattern)))
    if not doc_files:
        print(f"[WARNING] No files found in {three_gpp_dir} matching '{glob_pattern}'")
        return

    total_chunks = 0
    for doc_idx, doc_file in enumerate(doc_files, 1):
        loader = DoclingLoader(str(doc_file))
        raw_pages = list(loader.load())
        
        if not raw_pages:
            print(f"  [SKIP] {doc_file.name} — loader returned no content")
            continue
        
        # Verify we have content in each page
        pages_with_content = [p for p in raw_pages if p.page_content]
        pages_without_content = len(raw_pages) - len(pages_with_content)
        
        if pages_without_content > 0:
            print(f"  [WARNING] {pages_without_content} pages have no content")

        page_contents = []
        for page_num, page in enumerate(raw_pages, 1):
            if page.page_content:
                page_contents.append(page.page_content)
        
        full_text = "\n\n".join(page_contents)
        
        print(f"  Full document length: {len(full_text):,} characters")
        print(f"  Pages with content: {len(page_contents)}/{len(raw_pages)}")
        
        full_text = clean_document(full_text)
        print(f"After cleaning: {len(full_text):,} chars")

        chunks = chunk_with_hierarchy(full_text, min_chunk_size=min_chunk_size)

        for chunk in chunks:
            chunk.metadata.update({
                'source':   str(doc_file),
                'doc_name': doc_file.name,
                'doc_type': '3gpp_spec'
            })

        print(f"    Created {len(chunks)} chunks")
        total_chunks += len(chunks)

        del full_text, raw_pages
        yield doc_file.name, chunks


    print(f"\n{'─'*60}")
    print(f"Total 3GPP chunks generated: {total_chunks}")
    print(f"{'─'*60}")


# ------------------------Load all the docs and chunk them at once (legacy)----------------
def load_and_chunk_3gpp_docs(
    three_gpp_dir: Path,
    glob_pattern: str = "*.docx",
    min_chunk_size: int = DEFAULT_CHUNK_SIZE
) -> List[Document]:
    
    all_docs: List[Document] = []
    for doc_name, chunks in load_and_chunk_3gpp_docs_streaming(three_gpp_dir, glob_pattern, min_chunk_size):
        all_docs.extend(chunks)
    return all_docs
#---------------------------------------------------------------------------

#  STEP 5 — QUICK VALIDATION

def validate_chunks(docs: List[Document], sample_size: int = 10) -> None:
    if not docs:
        print("[VALIDATE] No documents to validate.")
        return

    word_counts = [d.metadata.get('word_count', len(d.page_content.split())) for d in docs]
    chunk_types = {}
    for d in docs:
        ct = d.metadata.get('chunk_type', 'unknown')
        chunk_types[ct] = chunk_types.get(ct, 0) + 1

    print("\n── Chunk Validation Report ──────────────────────────────────")
    print(f"Total chunks      : {len(docs)}")
    print(f"Min words/chunk   : {min(word_counts)}")
    print(f"Max words/chunk   : {max(word_counts)}")
    print(f"Avg words/chunk   : {sum(word_counts) / len(word_counts):.0f}")
    print(f"Chunk type breakdown:")
    for ct, count in sorted(chunk_types.items(), key=lambda x: -x[1]):
        print(f"  {ct:<25} {count}")

    tiny = [d for d in docs if len(d.page_content.split()) < MIN_CHUNK_WORDS]
    if tiny:
        print(f"\n[WARNING] {len(tiny)} chunks below {MIN_CHUNK_WORDS} words — review these:")
        for t in tiny[:5]:
            print(f"  §{t.metadata.get('section')} — {len(t.page_content.split())} words")

    missing_section = [d for d in docs if 'section' not in d.metadata]
    if missing_section:
        print(f"\n[WARNING] {len(missing_section)} chunks missing 'section' metadata")

    import random
    print(f"\n── Sample Chunks (random {sample_size}) ──────────────────────────")
    for d in random.sample(docs, min(sample_size, len(docs))):
        print(f"\n  §{d.metadata.get('section', 'N/A')} | {d.metadata.get('section_title', '')} "
              f"| {d.metadata.get('word_count', '?')} words | {d.metadata.get('chunk_type', '?')}")
        preview = d.page_content[:200].replace('\n', ' ')
        print(f"  Preview: {preview}...")
    print("─────────────────────────────────────────────────────────────\n")




# ----------------------------- Main Fxn & CLI ------------------------------------------------
def main() -> int:
    # Determine project root relative to this file
    project_root = Path(__file__).resolve().parent.parent
    default_dir = project_root / 'data' / 'raw' / '3gpp_docs'

    three_gpp_dir = default_dir
    if not three_gpp_dir.exists():
        print(f"[ERROR] Directory not found: {three_gpp_dir}")
        return 2

    docs = load_and_chunk_3gpp_docs(three_gpp_dir, glob_pattern="*.docx", min_chunk_size=DEFAULT_CHUNK_SIZE)
    validate_chunks(docs)

    return 0

#-------------Main Guard ------------------------------------------------------------
if __name__ == '__main__':
    raise SystemExit(main())
#----------------------------------------------------------------------------