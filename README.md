# chat-with-my-notes-v2

## Project Overview
A standalone RAG assistant and evaluation pipeline for a personal knowledge base built from Notion notes, self-sent emails, and chat exports. Existing repository work reflects Sprint 1 delivery, while planned enhancements are scoped for Sprint 2. This repo is part of AI Shipping Labs Sprint 2 and provides a production-ready FastAPI application with comprehensive search and RAG evaluation capabilities.

The system uses **Bilingual Retrieval Fusion (BRF)** for mixed Italian/English content, hybrid lexical-semantic search, and LLM-as-judge evaluation on a golden set. All retrievals achieve perfect Hit Rate@10 with MRR=1.00.

## Architecture
- **BRF (Bilingual Retrieval Fusion)**: Italian query → translation via Ollama judge → dual TF-IDF search (IT + EN) → RRF fusion with k=60
- **Semantic search**: Ollama embeddings (`nomic-embed-text`) with `minsearch.VectorSearch`
- **Hybrid fusion**: RRF over BRF + semantic results (recommended for production)
- **FastAPI server**: `POST /chat` (advanced options) and `POST /query` (simple interface)
- **pydantic-ai Agent**: Single `search_vault` tool with graceful degradation
- **LLM-as-Judge**: Async scoring on accuracy, completeness, hallucination, relevance (1–5 scale)
- **Golden set**: 5 manually curated queries with SHA1-verified chunk IDs

## Stack & Dependencies
- Python 3.11+ (project requires `>=3.11`)
- `pydantic-ai>=0.0.14`
- `httpx>=0.27.0`
- `minsearch>=0.0.11`
- `Ollama` for generator and judge endpoints
- Additional dependencies from `pyproject.toml`: `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `python-dotenv`, `pyyaml`

## Setup
1. Clone the repository.
2. Create a development environment and install dependencies with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Do not use `uv sync` for dependency installation, because `uv.lock` is intentionally not versioned in this repository to keep the project flexible for other users.

3. Copy the example environment file and configure endpoints:

```bash
cp .env.example .env
```

4. Set the following environment variables in `.env`:
- `OLLAMA_GENERATION_URL` — RAG generation model
- `OLLAMA_JUDGE_URL` — Translation, query expansion, and LLM-as-judge
- `OLLAMA_EMBED_URL` — Embedding model for semantic search

> Note: `OLLAMA_JUDGE_URL` serves multiple roles (translation, expansion, judge) using the same endpoint.

## Usage

### Evaluation
Run the evaluation pipeline from `run_eval.py`:

```bash
# Search-only evaluation (all modes)
python run_eval.py --mode all --top_k 10

# Search-only evaluation (specific mode)
python run_eval.py --mode brf --top_k 10
python run_eval.py --mode semantic --top_k 10
python run_eval.py --mode fusion --top_k 10

# Full RAG evaluation (retrieval + generation + judge)
python run_eval.py --mode rag_eval --top_k 10
```

### FastAPI Server
Start the production API server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### API Endpoints
- `POST /query` — Simple RAG interface: `{ "question": "..." }`
- `POST /chat` — Advanced interface with `top_k`, `filters`, `expand`
- `GET /index/info` — Index metadata
- `POST /ingest` — Trigger ingestion from configured sources

## Sprint 1 Evaluation Results

### Search Evaluation (Retrieval)
| Mode | Hit Rate@10 | MRR |
|---|---|---|
| BRF | 1.00 | 1.00 |
| Semantic | 0.600 | 0.192 |
| Fusion | 1.00 | 1.00 |

All retrieval modes tested on the 5-query golden set and reported in `evaluation/rag_eval_fusion_k10_v11.json`.

### RAG Evaluation (End-to-End)
- **Generator**: `gemma4-8k:latest` (8192 token context)
- **Judge**: `gemma4:e4b`
- **Results**: 5/5 queries completed successfully
- **Report**: `evaluation/rag_eval_fusion_k10_v11.json`
- **Run ID**: `search_eval_rag_eval_20260617T081433Z`
- **Mode**: `rag_eval`
- **top_k**: `10`
- **Date**: `2026-06-17T08:14:33.418731Z`
- **No timeouts** - full evaluation completed successfully

### Execution timings
Per-query timings from the v11 RAG evaluation run:
- QQ1: retrieval=6.1s generation=33.2s judge=26.6s total=65.9s
- QQ2: retrieval=6.4s generation=20.6s judge=24.3s total=51.2s
- QQ3: retrieval=8.2s generation=16.9s judge=17.3s total=42.3s
- QQ4: retrieval=5.3s generation=43.7s judge=27.1s total=76.1s
- QQ5: retrieval=8.4s generation=54.5s judge=28.2s total=91.1s

This timing breakdown reflects the full RAG pipeline with retrieval, answer generation, and judge scoring for each golden query.

## Output Files
Evaluation results are saved to `evaluation/` with timestamped filenames:
- `search_eval_*.json` — Search-only results (Hit Rate@k, MRR)
- `rag_eval_*.json` — Full pipeline results with judge scores
- `failure_log.json` — Detailed failure tracking (deprecated in v2, replaced by structured eval outputs)

## Test Suite Status
The project has 23 automated tests covering core functionality:

| Category | Tests | Status |
|---|---|---|
| API smoke tests (offline) | 2 | ✅ All pass |
| API smoke tests (integration) | 2 | ✅ All pass |
| Configuration | 2 | ✅ All pass |
| Golden set parsing | 3 | ✅ All pass |
| Judge score extraction | 4 | ✅ All pass |
| Notion adapter | 5 | ✅ All pass |
| Evaluation utilities | 5 | ✅ All pass |

**Run tests**:
```bash
cd chat-with-my-notes-v2
pytest tests/ -v
```

Note: Integration tests require the FastAPI server running (`uvicorn main:app --reload`) and Ollama models available.

## Known Limitations
| Item | Status |
|---|---|
| DERP relay latency | ✅ Resolved — dedicated Ollama instances with stable context |
| Long-context queries | ✅ Resolved — `gemma4-8k:latest` with 8192 token context |
| `num_predict` mapping | ✅ Resolved — updated Modelfile to `num_predict=4096` |
| Judge rubric calibration | 📋 Planned — Sprint 2 recall@k metric improvement |
| Golden set size | 📋 Planned — expand from 5 to 50–200 queries |

## pydantic-ai Integration
The `pydantic-ai` Agent in `agent/rag_agent.py` is **active** and powers the production FastAPI endpoints (`POST /chat`, `POST /query`). It uses a single `search_vault` tool for hybrid BRF + semantic retrieval with graceful degradation.

`run_eval.py` provides direct evaluation modes for benchmarking independent of the agent layer.

## Sprint 2 Roadmap
| Priority | Item | Status |
|---|---|---|
| High | Stable H3-based `chunk_id` | 🔄 Partial (SHA1 in use for eval) |
| High | Golden set expansion (50–200 queries) | 📋 Planned for Sprint 2 |
| Medium | Gmail adapter (self-sent emails) | 📋 Planned for Sprint 2 |
| Medium | Chat export adapter | 📋 Planned for Sprint 2 |
| Medium | Incremental sync endpoint (`POST /sync`) | 📋 Planned for Sprint 2 |
| Medium | Recall@k metric (not binary) | 📋 Planned for Sprint 2 |
| Low | Highlighting support in API | 📋 Planned for Sprint 2 |
| Low | Unit tests for search modules (`brf.py`, `semantic.py`, `fusion.py`) | 📋 Planned for Sprint 2 |
| Low | Unit tests for `ingest/orchestrator.py` | 📋 Planned for Sprint 2 |
| Low | Integration tests for `/chat` and `GET /index/info` endpoints | 📋 Planned for Sprint 2 |

> Sprint 2 note: the current production pipeline is not yet fully aligned with the experimental local pipeline. The local `run_eval.py` path uses direct Ollama HTTP calls with `num_predict=4096` and a controlled local context build, while production currently uses `pydantic-ai` with `max_tokens` semantics and a different agent/tool workflow. This will be realigned in the future to ensure fidelity of the production process with respect to local experimentation.
