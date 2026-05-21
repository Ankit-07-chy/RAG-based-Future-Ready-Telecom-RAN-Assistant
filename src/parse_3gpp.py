# Importing
from  pathlib import Path
from langchain_docling import DoclingLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
import re
from tqdm import tqdm 

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MIN_CHUNK_WORDS      = 50    # chunks below this are discarded as noise
DEFAULT_CHUNK_SIZE   = 200   # target min words per chunk
PARENT_CONTEXT_LINES = 3     # how many lines of parent section to prepend to child chunks


# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================



# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — CLEANING
# ─────────────────────────────────────────────────────────────────────────────

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
        # Only one match found — still better to start from there than keep boilerplate
        return text[matches[0].start():]
    return text


def remove_void_sections(text: str) -> str:
    """
    Remove 'Void' placeholder sections that 3GPP uses when a clause was deleted.
    These add noise to the vector store.
    Example:  '5.3.2  Void'
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


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — SECTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# FIX: Loosened regex compared to original.
# Original required title to start with capital and contain only letters/spaces/hyphens.
# This missed headers like:
#   "5.3.4 UE behaviour in RRC_IDLE"  (underscore)
#   "6.1.2 NR and E-UTRA"             (hyphen mid-word)
#   "7.2 TS 38.331 reference"         (spec number in title)
#
# New pattern:
#   - Section number: one or more digit groups separated by dots  e.g. 5, 5.3, 5.3.4
#   - At least one space
#   - Title: anything that starts with a letter (upper or lower), min 3 chars
#   - Anchored to start of line (MULTILINE)

SECTION_PATTERN = re.compile(
    r'^(\d+(?:\.\d+)*)\s{1,4}([A-Za-z].{2,}?)$',
    re.MULTILINE
)


def is_valid_section_title(title: str) -> bool:
    """
    Extra guard to reject false-positive section matches.
    Filters out things like pure number lines, very short noise lines,
    and common false positives from table formatting.
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


def find_sections(text: str) -> list[dict]:
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


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def get_parent_context(section_num: str, all_sections: list[dict], full_text: str) -> str:
    """
    FIX: Original code had no parent context injection.
    
    For a child section like '5.3.2', find its parent '5.3' and extract
    the first PARENT_CONTEXT_LINES lines as a context header to prepend.
    
    This makes every chunk self-contained — the LLM knows what the parent
    clause is about even when only seeing the child chunk.
    """
    parts = section_num.split('.')
    if len(parts) <= 1:
        return ''  # top-level section has no parent

    parent_num = '.'.join(parts[:-1])

    # Find parent section in the list
    parent = next((s for s in all_sections if s['num'] == parent_num), None)
    if not parent:
        return ''

    # Find where the parent section's content starts and ends
    parent_idx = all_sections.index(parent)
    parent_start = parent['start']
    # Parent ends where the next section starts (or end of text)
    if parent_idx + 1 < len(all_sections):
        parent_end = all_sections[parent_idx + 1]['start']
    else:
        parent_end = len(full_text)

    parent_content = full_text[parent_start:parent_end].strip()
    parent_lines   = [l for l in parent_content.splitlines() if l.strip()]

    if not parent_lines:
        return ''

    # Take heading + first few content lines as context
    context_lines = parent_lines[:PARENT_CONTEXT_LINES]
    return f"[Parent: §{parent_num} — {parent['title']}]\n" + "\n".join(context_lines)


def split_into_section_chunks(full_text: str) -> list[dict]:
    """
    Split the full document text into per-section raw chunks.
    Each chunk contains the complete text of one section.
    
    Returns list of:
        { 'num': '5.3', 'title': 'Header text', 'content': '...' }
    """
    sections = find_sections(full_text)
    if not sections:
        # Fallback: treat entire document as one chunk
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
    """Return depth of section: '5' → 1, '5.3' → 2, '5.3.1' → 3"""
    return len(section_num.split('.'))


def chunk_large_section_by_paragraphs(
    content: str,
    section_num: str,
    section_title: str,
    parent_context: str,
    min_chunk_size: int
) -> list[Document]:
    """
    Last-resort splitter: when a section has no subsections and is too large,
    split by paragraphs and group until we reach min_chunk_size.
    """
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

    # Remaining paragraphs
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


def assemble_chunk(parent_context: str, body: str) -> str:
    """
    Combine optional parent context header with the chunk body.
    """
    if parent_context:
        return f"{parent_context}\n\n{body}"
    return body


def chunk_with_hierarchy(
    full_text: str,
    min_chunk_size: int = DEFAULT_CHUNK_SIZE
) -> list[Document]:
    """
    Main chunking function.
    
    Strategy:
    1. Split text into per-section raw blocks using section headers.
    2. For each block:
       a. If small enough → emit as one chunk with parent context prepended.
       b. If large and has subsections → split at subsections (preserving their headers).
       c. If large and no subsections → split by paragraphs.
    3. Filter out any chunk below MIN_CHUNK_WORDS.
    
    FIX vs original:
    - Section headers are NO LONGER lost when splitting (was a bug in original).
    - Subsection metadata uses REAL section numbers, not enumerate index.
    - Parent context is injected into every child chunk.
    - Minimum word count filter removes noise chunks.
    """
    all_sections = find_sections(full_text)
    raw_chunks   = split_into_section_chunks(full_text)
    final_docs   = []

    for raw in raw_chunks:
        section_num   = raw['num']
        section_title = raw['title']
        content       = raw['content']
        word_count    = len(content.split())

        parent_context = get_parent_context(section_num, all_sections, full_text)

        # ── Case A: Small enough — emit as single chunk ──────────────────
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

        # ── Case B: Large — try splitting by direct subsections ──────────
        # We look for subsections that are exactly ONE level deeper.
        # e.g., for section '5.3', we look for '5.3.X' but NOT '5.3.X.Y'
        current_depth    = get_section_depth(section_num)
        subsec_pattern   = re.compile(
            rf'^({re.escape(section_num)}\.\d+)\s{{1,4}}([A-Za-z].{{2,}}?)$',
            re.MULTILINE
        )
        subsec_matches   = list(subsec_pattern.finditer(content))

        if subsec_matches:
            # FIX: Use finditer + manual slicing to PRESERVE subsection headers.
            # Original used re.split() which consumed (discarded) the matched header text.
            split_positions = [m.start() for m in subsec_matches] + [len(content)]

            # Text before the first subsection (intro paragraph of parent)
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

            # Each subsection — slice from its start to next subsection start
            for i, match in enumerate(subsec_matches):
                sub_start = split_positions[i]
                sub_end   = split_positions[i + 1]
                sub_text  = content[sub_start:sub_end].strip()

                # FIX: Use the REAL section number from the regex match,
                # not an enumerate index like the original did.
                real_sub_num   = match.group(1).strip()   # e.g. '5.3.2'
                real_sub_title = match.group(2).strip()   # e.g. 'UE behaviour'

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
                    # Subsection itself is large → paragraph-split it
                    para_chunks = chunk_large_section_by_paragraphs(
                        sub_text, real_sub_num, real_sub_title,
                        sub_parent_ctx, min_chunk_size
                    )
                    final_docs.extend(para_chunks)

        else:
            # ── Case C: No subsections — split by paragraphs ─────────────
            para_chunks = chunk_large_section_by_paragraphs(
                content, section_num, section_title,
                parent_context, min_chunk_size
            )
            final_docs.extend(para_chunks)

    print(f"  → {len(final_docs)} chunks after filtering (min {MIN_CHUNK_WORDS} words)")
    return final_docs


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — DOCUMENT LOADING & ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_chunk_3gpp_docs(
    three_gpp_dir: Path,
    glob_pattern: str = "*.docx",       # FIX: was hardcoded to one file
    min_chunk_size: int = DEFAULT_CHUNK_SIZE
) -> list[Document]:
    """
    Load all 3GPP documents from a directory, clean, chunk, and return
    a flat list of LangChain Documents with full metadata.

    Args:
        three_gpp_dir:  Path to folder containing 3GPP .docx files.
        glob_pattern:   File pattern to match. Default '*.docx' processes all.
                        Use e.g. '37340-h30.docx' for a single file during testing.
        min_chunk_size: Target minimum words per chunk.
    """
    doc_files = list(three_gpp_dir.glob(glob_pattern))
    if not doc_files:
        print(f"[WARNING] No files found in {three_gpp_dir} matching '{glob_pattern}'")
        return []

    all_docs = []

    for doc_file in tqdm(doc_files, desc="Processing 3GPP documents"):
        try:
            loader = DoclingLoader(str(doc_file))
            raw_pages = list(loader.lazy_load())

            if not raw_pages:
                print(f"  [SKIP] {doc_file.name} — DoclingLoader returned no content")
                continue

            # Merge all pages into one text block
            # (DoclingLoader may split a single docx across multiple Document objects)
            full_text = "\n\n".join(p.page_content for p in raw_pages if p.page_content)

            print(f"\n[INFO] {doc_file.name} — raw length: {len(full_text):,} chars")

            # Clean
            full_text = clean_document(full_text)
            print(f"       After cleaning: {len(full_text):,} chars")

            # Chunk
            chunks = chunk_with_hierarchy(full_text, min_chunk_size=min_chunk_size)

            # Attach document-level metadata to every chunk
            for chunk in chunks:
                chunk.metadata.update({
                    'source':   str(doc_file),
                    'doc_name': doc_file.name,
                    'doc_type': '3gpp_spec'
                })

            all_docs.extend(chunks)
            print(f"       Created {len(chunks)} chunks")

        except Exception as e:
            print(f"  [ERROR] Failed to process {doc_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'─'*60}")
    print(f"Total 3GPP chunks ready for embedding: {len(all_docs)}")
    print(f"{'─'*60}")
    return all_docs


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — QUICK VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_chunks(docs: list[Document], sample_size: int = 10) -> None:
    """
    Spot-check the output chunks.
    Prints statistics and a sample of chunks to review manually.
    Run this after loading to catch any remaining issues before embedding.
    """
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

    # Check for chunks that slipped through with suspiciously low word count
    tiny = [d for d in docs if len(d.page_content.split()) < MIN_CHUNK_WORDS]
    if tiny:
        print(f"\n[WARNING] {len(tiny)} chunks below {MIN_CHUNK_WORDS} words — review these:")
        for t in tiny[:5]:
            print(f"  §{t.metadata.get('section')} — {len(t.page_content.split())} words")

    # Check metadata completeness
    missing_section = [d for d in docs if 'section' not in d.metadata]
    if missing_section:
        print(f"\n[WARNING] {len(missing_section)} chunks missing 'section' metadata")

    # Sample output
    import random
    print(f"\n── Sample Chunks (random {sample_size}) ──────────────────────────")
    for d in random.sample(docs, min(sample_size, len(docs))):
        print(f"\n  §{d.metadata.get('section', 'N/A')} | {d.metadata.get('section_title', '')} "
              f"| {d.metadata.get('word_count', '?')} words | {d.metadata.get('chunk_type', '?')}")
        preview = d.page_content[:200].replace('\n', ' ')
        print(f"  Preview: {preview}...")
    print("─────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    PROJECT_ROOT  = Path.cwd().parent
    three_gpp_dir = PROJECT_ROOT / "data" / "raw" / "3gpp_docs"

    # ── For testing: process only one file ──
    # all_docs = load_and_chunk_3gpp_docs(three_gpp_dir, glob_pattern="37340-h30.docx")

    
    all_docs = load_and_chunk_3gpp_docs(
        three_gpp_dir,
        glob_pattern="*.docx",
        min_chunk_size=DEFAULT_CHUNK_SIZE
    )

    # Validate output before passing to embedding
    validate_chunks(all_docs, sample_size=10)

    # ── Next step: pass all_docs to your embedding + FAISS pipeline ──
    # from embedding_pipeline import embed_and_store
    # embed_and_store(all_docs)
   