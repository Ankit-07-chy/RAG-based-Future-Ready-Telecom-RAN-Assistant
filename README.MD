# Telecom RAG (Retrieval-Augmented Generation) by TeloMate.

A complete production-ready RAG system for 3GPP telecommunications specifications.

## Features of TeloMate

- ✅ **3GPP Document Processing**: Loads .docx files, cleans, and chunks with hierarchy awareness
- ✅ **Smart Chunking**: Preserves section hierarchy and parent context
- ✅ **Embedding Pipeline**: Batch embedding with memory management and validation
- ✅ **FAISS Vector Store**: Fast similarity search with incremental updates
- ✅ **RAG Query Interface**: Natural language queries over 3GPP specs
- ✅ **CLI & Python API**: Use from command line or Python code

## Project Structure

``` # This needs some final changes
telecom-rag/
├── data/
│   ├── raw/
│   │   ├── 3gpp_docs/         # 3GPP .docx specification files
│   │   └── teleqna/           # TeleQnA dataset
│   └── vectorstore/           # FAISS index (generated)
├── src/
│   ├── parse_3gpp.py          # Document parsing & chunking
│   ├── 3gpp_embed_vectorstore.py  # Embedding & vector store
│   ├── pipeline.py            # Orchestration pipeline
│   ├── rag_query.py           # Query interface
│   └── parse_teleqna.py       # TeleQnA dataset handling
└── experiments/
    └── exp.ipynb              # Experimentation notebook
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. Add 3GPP Documents

Place 3GPP .docx specification files in:
```
data/raw/3gpp_docs/
```

Example files (already in repo):
- `38211-h30.docx` (NR Physical Channels)
- `38212-h30.docx` (NR Multiplexing)
- etc.

### 3. Build Vector Store

```bash
python src/pipeline.py
```

Options:
```bash
python src/pipeline.py --help
python src/pipeline.py --chunk-size 150      # Smaller chunks as our query is Sparse based
python src/pipeline.py --no-merge            # Overwrite existing store
python src/pipeline.py --skip-embedding      # Test without embedding
```

### 4. Query the System

**From CLI:**
```bash
python src/rag_query.py "What is MIMO in 5G?"
python src/rag_query.py "How does HARQ work?"
```

**From Python:**
```python
from src.rag_query import TelecomRAG, format_results

rag = TelecomRAG()

# Simple query
results = rag.query("What is CSI feedback?", k=5)
print(format_results(results))

# Get stats
stats = rag.get_stats()
print(f"Total documents: {stats['total_chunks']}")

# Retrieve specific section
section_docs = rag.retrieve_by_section("5.3.1")
```

## Architecture

### 1. Document Parsing (`parse_3gpp.py`)

- **Step 1**: Load .docx files using Docling (with python-docx fallback)
- **Step 2**: Clean text (remove preface, void sections, normalize whitespace)
- **Step 3**: Detect section hierarchy (§5, §5.3, §5.3.1, etc.)
- **Step 4**: Chunk intelligently:
  - Full sections if < 200 words
  - Split by subsections if possible
  - Split by paragraphs for very large sections
  - Preserve parent context (3 lines from parent section)
- **Step 5**: Filter chunks below 50 words

**Output**: LangChain `Document` objects with rich metadata

### 2. Embedding Pipeline (`3gpp_embed_vectorstore.py`)

- **Model**: `BAAI/bge-small-en` (384-dim, lightweight)
- **Batch Size**: 128 documents
- **Memory Management**: Auto-pause if RAM > 85%
- **Validation**: Embedding dimension checks, zero-vector detection
- **Storage**: FAISS index with serialized metadata

### 3. RAG Query (`rag_query.py`)

- Load FAISS index + embeddings
- Embed query using same model
- Similarity search with threshold filtering
- Return top-k results with relevance scores

## Configuration

Edit these constants in `src/3gpp_embed_vectorstore.py`:

```python
MODEL_NAME          = "BAAI/bge-small-en"  # Embedding model
BATCH_SIZE          = 128                  # Docs per batch
MAX_MEMORY_USAGE    = 0.85                 # Halt if RAM > 85%
NORMALIZE_EMBEDDINGS= True                 # For cosine similarity
```

## Output Examples

### Chunking Report
```
[INFO] 38211-h30.docx — raw length: 1,234,567 chars
       After cleaning: 1,100,000 chars
       → 445 chunks after filtering (min 50 words)
```

### Query Results
```
🔍 Query: What is MIMO in 5G?

[1] §5.3.1 — Precoding for MIMO
    Relevance: 87.5% | Words: 156 | Source: data/raw/3gpp_docs/38211-h30.docx
    Preview: MIMO (Multiple-Input Multiple-Output) techniques in NR are used to enhance...

[2] §5.3.2 — Spatial Multiplexing
    Relevance: 82.1% | Words: 189 | Source: data/raw/3gpp_docs/38211-h30.docx
    Preview: Spatial multiplexing allows independent data streams on different spatial...
```

## Advanced Usage

### Incremental Updates

By default, new documents are merged with existing index:
```python
from src.pipeline import run_pipeline

result = run_pipeline(merge_existing=True)  # Default: merge
# or
result = run_pipeline(merge_existing=False) # Rebuild from scratch
```

### Custom Chunk Sizes

```bash
python src/pipeline.py --chunk-size 100    # Smaller chunks
python src/pipeline.py --chunk-size 300    # Larger chunks
```

### Retrieve by Section

```python
rag = TelecomRAG()
docs = rag.retrieve_by_section("5.3.1")
```

### Batch Processing

```python
from src.pipeline import run_pipeline

# Process multiple documents
result = run_pipeline()

if result["status"] == "success":
    print(f"Processed {result['docs_count']} documents")
    print(f"Store: {result['vector_store_path']}")
```

## Data Sources

- **3GPP Specs**: Technical standards (3GPP TS 38.211, 38.212, etc.)
- **TeleQnA**: 10,000 telecom QA pairs (available in `data/raw/teleqna/`)

## Performance

On typical machine:
- **Chunking**: ~100 docs/min
- **Embedding**: ~5,000 chunks/min (single GPU: 50,000/min)
- **Query**: <100ms for similarity search

## Troubleshooting

### "No documents loaded"
→ Check `data/raw/3gpp_docs/` contains .docx files

### "Vector store not found"
→ Run `python src/pipeline.py` to build it

### Memory errors during embedding
→ Reduce `BATCH_SIZE` in `3gpp_embed_vectorstore.py`

### Slow queries
→ Use smaller embedding model or reduce chunk count

## Testing

```bash
# Test without embedding
python src/pipeline.py --skip-embedding

# Test parsing only
python src/parse_3gpp.py --dir data/raw/3gpp_docs

# Test RAG query
python src/rag_query.py "test question"
```

## Future Enhancements

- [ ] Web API (FastAPI/Flask)
- [ ] Multi-modal embeddings (text + diagrams)
- [ ] Hybrid search (keyword + semantic)
- [ ] Query refinement with LLM
- [ ] Automated answer generation
- [ ] Integration with LangChain agents

## References

- [3GPP Specifications](https://www.3gpp.org/specifications)
- [TeleQnA Dataset](https://arxiv.org/abs/2310.15051)
- [BAAI BGE Embeddings](https://huggingface.co/BAAI/bge-small-en)
- [FAISS Documentation](https://faiss.ai/)

## License

MIT License

## Citation

If you use this RAG system, please cite:

```bibtex
@misc{telecom_rag,
  title={Telecom RAG: Retrieval-Augmented Generation for 3GPP Specifications},
  author={Your Name},
  year={2025}
}
```
