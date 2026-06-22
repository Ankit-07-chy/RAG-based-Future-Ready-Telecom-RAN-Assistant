# Telecom RAN Assistant — Domain-Specific RAG

A Retrieval-Augmented Generation system for Telecom Radio Access Networks. It
answers 3GPP specification questions and supports root-cause analysis, anomaly
detection, and optimization over public telecom datasets (TeleQnA, O-RAN,
Simu5G, 3GPP Release 16/18), with **source-cited, explainable** answers.

Built for the "Future-Ready Telecom RAN Assistant" problem statement and
evaluated against five target KPIs (MRR, Top-k Accuracy, Accuracy, Recall,
Faithfulness).

---

## Architecture

```
Raw data  ─ 3GPP .docx ─┐
          ─ TeleQnA ─────┤  parse_*  →  chunk + metadata
          ─ O-RAN JSON ──┤
          ─ Simu5G JSON ─┘
                          │
                  BGE bge-base-en-v1.5 (768-d) → FAISS index
                          │
        User query ──► Hybrid retrieval (BM25 + dense, RRF, top-20)
                     ─► Cross-encoder rerank (top-5)
                     ─► Source-diversity / MMR filter
                     ─► RAG chain: context + persona prompt → Groq LLM
                     ─► Structured output: ANSWER / REASONING / SOURCES
```

- **Retrieval, embedding, reranking run fully locally.** Only the final
  generation step calls the Groq API (query + retrieved public context only).
- **Generation model:** `llama-3.3-70b-versatile` via Groq (fast, low-latency).
- **Fine-tuning (optional):** QLoRA adapter trained on Colab — see
  [Fine-tuning](#fine-tuning-qlora).

---

## Setup

```bash
# 1. Install dependencies (Python 3.11–3.13)
pip install -r requirements.txt
# or: uv sync

# 2. Configure secrets
cp .env.example .env          # then add your GROQ_API_KEY

# 3. Build the vector store (parse → embed → FAISS index)
python main.py --mode pipeline
```

Data already lives under `data/raw/` (3GPP `.docx`, `teleqna_dataset/TeleQnA.txt`,
O-RAN and Simu5G JSON). The pipeline indexes 3GPP + O-RAN + Simu5G + the
**TeleQnA train split** (the held-out test split is deliberately excluded so
evaluation cannot self-retrieve answers).

---

## Usage

```bash
# Ask a question (full answer + reasoning + sources)
python main.py --mode query --query "What is the purpose of the RACH procedure in 5G NR?"

# Run the KPI evaluation on the held-out TeleQnA test set
python main.py --mode eval --max-eval-samples 200

# Start the REST API  (POST /query, POST /retrieval, GET /health, GET /stats)
python main.py --mode api --port 8000

# Launch the Streamlit demo UI
streamlit run src/streamlit_app.py
```

### API example

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Why would a cell show high RRC connection failures?", "use_llm": true, "use_hybrid": true}'
```

Returns `{answer, reasoning, sources[], query_type, confidence, retrieved_chunks}`.

---

## Evaluation & KPIs

Evaluation runs on a held-out 20% TeleQnA test split (deterministic seed). The
methodology is documented in `src/evaluator.py`:

- **Corpus excludes the test split** — no trivial self-retrieval.
- **Accuracy** = true multiple-choice option selection by the LLM from retrieved
  context, scored against the ground-truth option index.
- **MRR / Top-k / Recall** use *answer-bearing* relevance (a chunk is relevant
  if it contains the ground-truth answer's content tokens), since TeleQnA has no
  gold passage labels.
- **Faithfulness** = LLM-as-judge grounding score (answer supported by context).

| Metric | Target | Result |
|---|---|---|
| MRR | > 75% | _see `results/latest_metrics.json`_ |
| Top-5 Accuracy | > 85% | _see `results/latest_metrics.json`_ |
| Accuracy | > 80% | _see `results/latest_metrics.json`_ |
| Recall | > 85% | _see `results/latest_metrics.json`_ |
| Faithfulness | > 90% | _see `results/latest_metrics.json`_ |

Per-question results are written to `evals/evaluation_per_question_*.csv`.

---

## Explainability & Security

- **Explainability:** every answer is returned in `ANSWER / REASONING / SOURCES`
  form; sources cite the document section / `doc_type` / source file behind each
  claim. The prompt instructs the model to use only the retrieved context.
- **Security/privacy** (`src/security.py`):
  - Retrieval/embedding/reranking are local; only generation leaves the machine.
  - Input sanitization bounds query size, strips control characters, and rejects
    basic prompt-injection patterns (`POST /query` returns 400).
  - `redact()` masks emails / IPs / long identifier strings for safe display.
  - `.env` (API key) is gitignored; use `.env.example` as the template.

---

## Fine-tuning (QLoRA)

Generation uses the Groq API by default (for the latency KPI). A QLoRA adapter
can be trained for domain adaptation / on-prem inference:

```bash
# 1. Export TeleQnA train pairs (held-out test excluded)
python scripts/export_finetune_data.py

# 2. On a Colab GPU runtime:
pip install -q transformers peft bitsandbytes accelerate datasets trl
python scripts/finetune_qlora_colab.py --train_file teleqna_finetune_train.jsonl
# → download telecom_qlora_adapter/ into models/

# 3. Load locally via src.llm_engine.LocalLLMEngine(adapter_path="models/telecom_qlora_adapter")
```

Base model defaults to the ungated `Qwen/Qwen2.5-3B-Instruct`; LoRA config
(rank 16, alpha 32, targets `q_proj`,`v_proj`) mirrors `src/config.py`.

---

## Project layout

| Path | Purpose |
|---|---|
| `src/config.py` | Single source of truth: paths, models, hyperparameters |
| `src/parse_*.py` | Source parsers (3GPP, TeleQnA, O-RAN, Simu5G) → Documents |
| `src/3gpp_embed_vectorstore.py` | BGE embeddings + FAISS index build/load |
| `src/hybrid_retrieval.py` | BM25 + dense search fused with RRF |
| `src/reranker.py` | Cross-encoder re-ranking (top-20 → top-5) |
| `src/retrieval_filters.py` | Source-diversity + MMR filters |
| `src/llm_engine.py` | Groq engine + local QLoRA (`LocalLLMEngine`) |
| `src/rag_chain.py` / `src/rag_query.py` | Context assembly, prompting, parsing |
| `src/pipeline.py` | Offline orchestrator: parse → embed → index |
| `src/evaluator.py` | Computes the five KPIs on the TeleQnA hold-out |
| `src/api.py` | FastAPI service | `src/streamlit_app.py` | Demo UI |
| `scripts/` | TeleQnA extraction, fine-tune export, QLoRA training |

