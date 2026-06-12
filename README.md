# chat-with-my-notes-v2

## Project Overview
A standalone RAG evaluation pipeline for a personal knowledge base built from a Notion vault. This repo is part of AI Shipping Labs Sprint 1 and was developed outside Hermes as an isolated lab for quantitative benchmarking. It validates retrieval and generation behavior on a 5-query golden set.

## Architecture
- BRF (Bilingual Retrieval Fusion): Italian query → translation via `gemma4:e4b` → dual search over Italian and English indexes → RRF fusion with `k=60`.
- Generator: direct `httpx` call to Ollama at `gemma4-8k:latest` deployed on Minisforum (`100.110.155.109`), using `num_predict=512`.
- LLM-as-Judge: `gemma4:e4b` on Asus F15 (`100.127.163.44`), rubric-based scoring for correctness, completeness, and hallucination.
- Golden set: 5 manually curated queries over the Notion vault.

## Stack & Dependencies
- Python 3.11+ (project requires `>=3.11`)
- `pydantic-ai>=0.0.14`
- `httpx>=0.27.0`
- `minsearch>=0.0.11`
- `Ollama` for generator and judge endpoints
- Additional dependencies from `pyproject.toml`: `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `python-dotenv`, `pyyaml`

## Setup
1. Clone the repository.
2. Create and activate a virtual environment in `.venv`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

3. Copy the example environment file and configure endpoints.

```bash
cp .env.example .env
```

4. Set the following environment variables in `.env`:
- `OLLAMA_GENERATION_URL`
- `OLLAMA_JUDGE_URL`

> Note: `OLLAMA_JUDGE_URL` is also used for Italian-to-English translation and judge roles. It is the same endpoint with two operational roles.

## Usage
Run the evaluation pipeline from `run_eval.py`.

- Search-only evaluation (retrieval only, Hit Rate @k):

```bash
python run_eval.py --mode search_eval --top_k 10
```

- Full RAG evaluation (retrieval + generation + judge):

```bash
python run_eval.py --mode rag_eval --top_k 10
```

## Sprint 1 Results
| Metric | Result | Notes |
|---|---|---|
| Hit Rate @10 | 5/5 = 100% | retrieval layer validated on the 5-query golden set |
| Pass RAG eval | 0/5 | generation layer blocked by infrastructure issue |
| Hallucination score | 5/5 | no hallucinations detected |
| Q1, Q4, Q5 | empty responses | DERP relay latency + `num_predict=512` insufficient for long-context queries |
| Q2, Q3 | factually correct responses | judged fail due to completeness-focused rubric |

Diagnosis: retrieval is sound; failures are infrastructure-bound or rubric calibration issues, not indicative of poor RAG retrieval quality.

## Output Naming Convention
| Pattern | When used |
|---|---|
| `rag_eval_fusion_k10_vN.json` | `httpx` direct path (current) |
| `rag_eval_agent_k10_vN.json` | reserved for `pydantic-ai` production path (Sprint 2) |

## Known Issues
- DERP relay latency on Tailscale causing generation timeouts
- `num_predict=512` (Ollama) is insufficient for long-context queries
- `num_predict` (Ollama) vs `max_tokens` (pydantic-ai `OpenAIChatModel`) mapping not yet resolved
- Judge rubric does not distinguish empty responses from correct but incomplete responses — calibration planned for Sprint 2

## pydantic-ai Note
The `pydantic-ai` Agent is present in `agent/rag_agent.py` but is NOT active in the current evaluation path. `run_eval.py` uses direct `httpx` calls in `generate_answer()` instead. `pydantic-ai` is reserved for the Sprint 2 production path (`FastAPI → agent/rag_agent.py → Telegram`).

## Sprint 2 Roadmap
- FastAPI integration
- `pydantic-ai` Agent active in production path
- Judge rubric calibration (completeness threshold)
- Golden set expansion with synthetic query generation
- Fix DERP relay / increase `num_predict` for long-context queries
