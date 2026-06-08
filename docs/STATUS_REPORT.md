# Sprint Notes

### Sprint 1 (hermes-vault, before v2)
- Built BRF search (vault_search.py) with IT→EN translation via Ollama
- Query expansion with 3 LLM-generated variants
- LLM-as-judge evaluation (master_run_eval.py)
- Hardcoded paths, urllib.request, sync code
- Source: `sprint1-hermes-backup` repository

### Sprint 1 — Week 3 (May 2026): v2 scaffolding + migration
- New repo `chat-with-my-notes-v2` with pydantic-ai + FastAPI
- Migrated: `brf.py` (vault_search.py), `orchestrator.py` (vault_ingest.py),
  `adapters/notion.py`, `evaluation/judge.py`, `evaluation/golden_set.py`
- All paths via `config.py` (pydantic-settings)
- All HTTP via httpx (async)
- `pyproject.toml` fixed: `hatchling` requires explicit `packages` list when
  project name has hyphens; `pydantic-ai[openai]` extra does not exist in v1

### Sprint 1 — Week 3 (May 2026): semantic search + incremental index
- Added `src/search/semantic.py`: Ollama embeddings + `minsearch.VectorSearch`
- Added `embed_documents()` and `append_documents()` to semantic.py
- Migrated `minsearch.Index` → `minsearch.AppendableIndex` in orchestrator + brf
- Added `brf.append_documents()` for in-place incremental BRF updates
- Config extended: `ollama_embed_url`, `embed_model`, `semantic_index_path`
- Commit: `a567ee2`

### Sprint 1 — Week 4 (May–June 2026): RAG runtime + API
- Added `src/search/fusion.py`: `rrf_fuse(brf_results, semantic_results)`
- Rewrote `agent/rag_agent.py`: hybrid search via `asyncio.gather`, graceful
  degradation with `return_exceptions=True`, `OpenAIChatModel` + `OpenAIProvider`
- Added `POST /query` endpoint (simplified interface)
- Added `QueryRequest` / `QueryResponse` to `api/schemas.py`
- Fixed pydantic-ai v1 API: `OpenAIModel` → `OpenAIChatModel`,
  `result.data` → `result.output`
- Commit: `b51967e`

### Sprint 1 — Week 4 (May 28 to June 8, 2026): recent progress
- 28 May: scaffolded Phase 3 v2 architecture with pydantic-ai + FastAPI
- 29 May: fixed `pyproject.toml` for `uv sync` and added CLI support to orchestrator
- 30 May: added incremental index append support via `AppendableIndex` and `VectorSearch.append_batch`
- 31 May: completed RAG runtime with fusion layer, hybrid `search_vault`, and `/query` endpoint; fixed pydantic-ai v1 compatibility in `rag_agent`
- 1 June: added `CONCEPT_DOC.md` architecture doc and golden-set placeholder; fixed sprint numbering; completed semantic index pipeline for hybrid search
- 3 June: stabilized API execution by removing route-level `asyncio.wait_for`; added pytest integration config and query smoke tests; added offline smoke fixtures for judge extraction and Notion adapter; introduced Phase 3 golden queries and judge-only ground truth loader; loaded golden-set from `CONCEPT_DOC.md` with parser coverage
- 4 June: improved Notion ingest chunking with sentence-aware secondary splits and stable chunk IDs; added reusable `evaluation/chunk_stats.py` for chunk distribution analysis
- 5 June: switched generation calls from `max_tokens` to `num_predict`; stabilized golden expected chunk IDs for Q2/Q4 and extended Q4 with a second DVC chunk; added Golden Questions characteristics documentation; fixed ITA→ENG comparison logic for v2 vs Hermes
- 6 June: aligned fusion, evaluation, and golden-set mapping on canonical `chunk_id`; completed Search Eval pipeline with BRF 1.0/0.9 and Fusion 1.0/1.0 at `k=10`
- 8 June: added `rag_eval` support and configuration env loading fix; added config env path loader test; recorded RAG evaluation output artifacts

### Sprint 2 planning notes
- `Recall@k` metric update: move from binary hit/miss to ratio-based scoring for multi-chunk queries
- `evaluation/generated/` folder: separate exploratory runs from baseline artifacts
- Expand golden set to 50–200 queries via stratified LLM generation
- Gmail/chat export adapters are still placeholders; source coverage incomplete
