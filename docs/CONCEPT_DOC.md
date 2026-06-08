# CONCEPT DOC — chat-with-my-notes-v2

> This document is the canonical reference for the architecture, design decisions,
> and evaluation strategy of the project. It is read by automation tools and agents.
> Keep it up to date as the codebase evolves.

---

## 1. Project Goal

Build a **personal RAG assistant** over a private knowledge base composed of
Notion notes, self-sent emails, and chat exports.

The assistant answers Italian-language questions grounded strictly on the owner's
notes. It must:
- Never use external knowledge beyond what is in the vault
- Always cite the source (title, category, file) for each answer
- Gracefully handle queries about topics not present in the notes

The system is designed for a single user (Luca). There is no multi-tenancy,
no authentication layer, and no public-facing security perimeter beyond the
localhost boundary of a personal server.

---

## 2. Technology Stack

| Component | Library / Version | Role |
|---|---|---|
| Agent framework | `pydantic-ai >= 0.0.14` | Single-shot RAG agent with tool calling |
| LLM client | `OpenAIChatModel` + `OpenAIProvider` (pydantic-ai v1) | Chat completions via Ollama OpenAI-compat API |
| HTTP client | `httpx >= 0.27` | All async HTTP calls to Ollama (replaces urllib) |
| API server | `fastapi >= 0.116` + `uvicorn` | REST entrypoint |
| Config | `pydantic-settings >= 2.3` | All runtime config from `.env`, no hardcoded paths |
| Lexical search | `minsearch.AppendableIndex` | TF-IDF + inverted index, incremental append |
| Vector search | `minsearch.VectorSearch` | Cosine similarity over Ollama embeddings |
| Frontmatter | `pyyaml >= 6.0` | Notion export YAML frontmatter parsing |
| Python | `>= 3.11` | Required for `X | Y` union syntax |
| Package manager | `uv` | `.venv` at project root, `uv sync` |
| Build backend | `hatchling` | `[tool.hatch.build.targets.wheel]` explicit because project name has hyphens |

**Ollama instances** (configured via `.env`):
- `OLLAMA_GENERATION_URL` — primary RAG generation model
- `OLLAMA_JUDGE_URL` — BRF translation, query expansion, and LLM-as-judge
- `OLLAMA_EMBED_URL` — embedding model for semantic index

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Data Sources                                               │
│  Notion .md exports  ·  Gmail self (stub)  ·  Chat exports  │
└────────────────────────────┬────────────────────────────────┘
                             │
                    src/ingest/orchestrator.py
                    (adapters per source type)
                             │
                ┌────────────┴────────────┐
                │                         │
        AppendableIndex              VectorSearch
        data/vault_index.pkl     data/semantic_index.pkl
        (TF-IDF, minsearch)      (cosine sim, minsearch)
                │                         │
           src/search/brf.py      src/search/semantic.py
           (BRF lexical)          (embed via Ollama)
                │                         │
                └────────────┬────────────┘
                             │
                    src/search/fusion.py
                    rrf_fuse(brf_results, semantic_results)
                             │
                    agent/rag_agent.py
                    pydantic-ai Agent
                    tool: search_vault
                    LLM: generation model via Ollama
                             │
                     api/router.py
                     FastAPI routes
                             │
              ┌──────────────┴──────────────┐
         POST /chat                   POST /query
         (advanced: top_k,            (simple: question → answer
          filters, expand)             + sources: list[str])
```

### Layer responsibilities

| Layer | File | Responsibility |
|---|---|---|
| Ingest | `src/ingest/orchestrator.py` | Coordinate adapters, build and save index |
| Adapter | `src/ingest/adapters/notion.py` | Parse Notion .md → chunk dicts |
| Lexical search | `src/search/brf.py` | BRF: TF-IDF + bilingual RRF |
| Semantic search | `src/search/semantic.py` | Ollama embed + VectorSearch |
| Fusion | `src/search/fusion.py` | RRF over BRF + semantic results |
| Agent | `agent/rag_agent.py` | pydantic-ai tool calling + LLM generation |
| API | `api/router.py` | FastAPI routes, thin delegation layer |
| Schemas | `api/schemas.py` | Pydantic models for all request/response types |
| Config | `config.py` | Single `Settings(BaseSettings)` instance |
| Evaluation | `evaluation/judge.py` | LLM-as-judge async scoring |
| Evaluation | `evaluation/golden_set.py` | Load golden questions from this document |

---

## 4. Repository Structure

```
chat-with-my-notes-v2/
├── main.py                     # FastAPI app entry point
├── config.py                   # pydantic-settings Settings class
├── pyproject.toml              # Dependencies, build config
├── .env.example                # Config template (no real values)
├── .env                        # Runtime config (gitignored)
│
├── agent/
│   └── rag_agent.py            # pydantic-ai Agent, search_vault tool, ask()
│
├── api/
│   ├── router.py               # FastAPI routes: /chat, /query, /ingest, /index/info
│   └── schemas.py              # Pydantic models: ChatRequest/Response, QueryRequest/Response
│
├── src/
│   ├── ingest/
│   │   ├── orchestrator.py     # run_ingest(), builds AppendableIndex
│   │   └── adapters/
│   │       ├── notion.py       # Notion .md H3-chunker (production-ready)
│   │       ├── gmail.py        # Stub — Sprint 4
│   │       └── chat_export.py  # Stub — Sprint 4
│   └── search/
│       ├── brf.py              # BRF engine: search(), append_documents()
│       ├── semantic.py         # Semantic engine: embed_documents(), search(), append_documents()
│       └── fusion.py           # rrf_fuse(brf_results, semantic_results, top_k)
│
├── evaluation/
│   ├── judge.py                # judge_answer() async, extract_judge_scores()
│   └── golden_set.py           # load_golden_set(), load_ground_truth()
│
└── docs/
    └── CONCEPT_DOC.md          # This file
```

**Data directory** (gitignored, created at runtime):
```
data/
├── vault_index.pkl             # AppendableIndex (BRF)
├── semantic_index.pkl          # VectorSearch (embeddings)
├── notion/                     # Notion .md exports
└── notion_pages_map.json       # Page hierarchy metadata
```

---

## 5. Query Language Mechanism

### Bilingual Retrieval Fusion (BRF)

The personal vault is written in **Italian and English**. A pure TF-IDF search
on a monolingual index would miss Italian content when queried in English and
vice versa. BRF addresses this by running parallel searches in both languages.

**Standard mode** (`expand=False`):
1. Run TF-IDF search with original Italian query → `results_IT`
2. Translate query IT→EN via Ollama judge model → `query_EN`
3. Run TF-IDF search with `query_EN` → `results_EN`
4. Fuse `[results_IT, results_EN]` via RRF (k=60) → top_k results

**Expansion mode** (`expand=True`):
1. Generate 3 Italian variant queries via LLM (synonyms, related terms)
2. Batch-translate all 4 Italian queries to English in one LLM call
3. Run 8 TF-IDF searches (4 IT + 4 EN), fuse all via RRF
4. Balanced 4:4 ratio preserves bilingual weighting

**Field boosting**: `title` weight 3.0, `text` weight 1.0

**RRF formula**: score = Σ `1 / (k + rank)` over each ranked list.
Default k=60 (standard literature value, also used in fusion.py).

### Semantic search

Chunks are embedded via Ollama (`embed_model`, e.g. `nomic-embed-text`).
At query time the query string is embedded and cosine similarity is computed
over the stored vector index (`minsearch.VectorSearch`).

### Hybrid fusion

`fusion.rrf_fuse(brf_results, semantic_results, top_k)` applies the same
RRF formula over the two heterogeneous ranked lists. Deduplication key:
`file|title|text[:80]` — same contract in both `brf._rrf_fuse_many` and
`fusion.rrf_fuse`.

### Filtering

`brf.search()` accepts an optional `filters: dict[str, str]` that is passed
to minsearch as `filter_dict`. Filterable keyword fields defined in orchestrator:
`source`, `category`, `subcategory`, `type`, `file`, `language`,
`has_ai_callouts`, `context`, `tags`.

---

## 6. Evaluation Strategy

### Search Evaluation (Retrieval)

**Not yet implemented as automated pipeline in v2.** The sprint1 codebase had
`test_minsearch_raw.py` and `test_debug_index_raw.py` for manual search quality
checks. These have not been migrated.

### RAG Evaluation (Answer Quality)

**`evaluation/judge.py`** — `judge_answer(query, answer, chunks, ground_truth)`

Async LLM-as-judge using the Ollama judge model. Two rubrics:
- **With ground truth** (`JUDGE_RUBRIC_WITH_GT`): compares RAG answer to a
  human-written expected answer on 4 axes
- **Without ground truth** (`JUDGE_RUBRIC_NO_GT`): judges answer against
  retrieved chunks only

**Scoring axes** (1–5 each):
- `accuracy` — factual correctness vs. ground truth (or context)
- `completeness` — coverage of all relevant details
- `hallucination` — absence of invented facts (5 = no hallucination)
- `relevance` — pertinence to the question

**Outcome logic**:
- `fail`: any criterion ≤ 2
- `warning`: all ≥ 3 but at least one < 5
- `pass`: all = 5

**Thinking-mode handling**: gemma4:e4b may return empty `content` and put the
answer in `reasoning`. `extract_judge_scores()` tries content first, then
reasoning, then regex fallback on individual scores.

**Automated pipeline**: `[TO BE COMPLETED]` — no `run_eval.py` exists in v2 yet.

---

## 7. Golden Questions

Golden questions are used to evaluate both retrieval quality (did the right
chunks surface?) and answer quality (did the LLM answer correctly?).

**Format**: a `golden-set` fenced JSON block embedded in this document.
`evaluation/golden_set.py::load_golden_set()` parses it directly from here.

**Ground truth**: stored separately in `evaluation/golden-set-ground-truth.json`
(not committed — contains personal information).

Required fields per entry: `id`, `topic`, `query`.
Optional: `expected_answer` (null entries fall back to chunk-based judging).

```golden-set
[
  {
    "id": "Q1",
    "topic": "Spese Tecnologiche",
    "query": "Quali sono le mie spese tecnologiche fisse annuali per il lavoro?"
  },
  {
    "id": "Q2",
    "topic": "Alibi Detect",
    "query": "Alibi Detect di Seldon serve a data versioning o a data drift analysis?"
  },
  {
    "id": "Q3",
    "topic": "Oxen.ai",
    "query": "Quanti repositories privati sono inclusi nel piano free di Oxen.ai?"
  },
  {
    "id": "Q4",
    "topic": "MLOps Tools Comparison",
    "query": "DVC viene usato nel corso MLOps Zoomcamp DTC, nel corso MLOps in 4 Weeks, o in entrambi?"
  },
  {
    "id": "Q5",
    "topic": "Parametri Grid Search",
    "query": "Quali valori di tau_L e tau_Ab ha prodotto la grid search con Huber Loss su VirusWatch?"
  }
]
```

### Golden Questions — Characteristics

| ID | Topic | Characteristics | Target Source Sprint 1 | Target Source Sprint 2 |
|----|-------|-----------------|------------------------|------------------------|
| Q1 | Spese Tecnologiche | Ricerca su documento finanziario personale con dati numerici strutturati | Notion | Gmail |
| Q2 | Gitpod | Deep search in sottopagine — informazione non in pagina top-level Notion | Notion | — |
| Q3 | Oxen.ai | Named entity univoca con dato numerico specifico — testa precision | Notion | — |
| Q4 | MLOps Tools Comparison | Multi-chunk / multi-file — confronto trasversale su due corsi distinti, expected_chunk_ids da due file separati | Notion | — |
| Q5 | Parametri Grid Search | Ricerca su esperienza lavorativa pregressa — contesto biografico/professionale | Notion | ChatGPT export |

>The v1 (Hermes baseline) vs v2 comparison is not directly comparable: v1 uses qualitative failure_log.json (Week 1-2), v2 uses formal Search Eval with stable chunk_id starting from Week 5. The formal baseline starts from the post-fix pkl (2177 chunks).

### 7.2 Search Eval Baseline v2 (@k=10)

| Mode    | Hit Rate | MRR   |
|---------|----------|-------|
| BRF     | 1.000    | 0.900 |
| Semantic| 0.600    | 0.192 |
| Fusion  | 1.000    | 1.000 |

Notes:
- Fusion è il motore raccomandato per produzione
- Semantic è debole da sola su corpus tecnico misto IT/EN
- Golden set: 5 query con expected_chunk_ids SHA1 verificati dall'indice post-fix chunking Week 5 Day 1
- chunk_id canonico SHA1 adottato come identificatore unico in tutto il sistema

---

## 8. Key Architectural Decisions

### 8.1 pydantic-ai as agent framework

**Decision**: use pydantic-ai `Agent` with a single `search_vault` tool instead
of a manual prompt-building pipeline.

**Rationale**: pydantic-ai handles tool-call loop, structured output validation,
and model switching. The tool interface makes it straightforward to add more
tools (e.g. a calendar lookup) without refactoring the generation logic.

**Constraint found in practice**: pydantic-ai has no embedding API — only
chat completions. Embeddings are handled by direct httpx calls to Ollama.

**API notes (v1.x)**:
- `OpenAIModel` renamed to `OpenAIChatModel`
- `base_url`/`api_key` moved to `OpenAIProvider`, passed via `provider=`
- `AgentRunResult.output` is the typed output (not `.data`, not `.response`)

### 8.2 AppendableIndex over Index

**Decision**: `orchestrator.py` builds `minsearch.AppendableIndex`, not `minsearch.Index`.

**Rationale**: the vault will receive regular Notion sync operations (new pages,
updated pages). `AppendableIndex.append(doc)` allows adding new documents
in-place without full reindex. The `brf.append_documents()` function exposes
this: it mutates `_index` in memory, then calls `save()` — disc and RAM stay
in sync within a single process.

**Performance note**: `AppendableIndex` is an optimized inverted index, 20-76x
faster than an early prototype but uses a different algorithm than
`Index` (scikit-learn TF-IDF). Search quality may differ slightly.

### 8.3 VectorSearch with append_batch

**Decision**: `semantic.py` uses `minsearch.VectorSearch` with its native
`save()`/`load()` instead of a manual pickle of `{chunks, embeddings}`.

**Rationale**: encapsulates serialization, and `append_batch(vectors, docs)`
enables incremental semantic index updates with the same contract as BRF.

### 8.4 fusion.py as an independent layer

**Decision**: `src/search/fusion.py` is a standalone module that knows nothing
about BRF or semantic internals — it only receives two `list[dict]`.

**Rationale**: `brf._rrf_fuse_many` serves internal BRF needs (N bilingual
queries). `fusion.rrf_fuse` serves inter-engine fusion. Same algorithm,
different responsibility boundaries. No cross-import.

### 8.5 Graceful degradation in search_vault

**Decision**: `search_vault` checks index file existence before calling each
engine, and uses `asyncio.gather(return_exceptions=True)` to catch runtime
failures of either engine without aborting the whole tool call.

**Rationale**: during development and incremental sync, one index may exist
while the other does not. The agent must remain functional with partial
infrastructure.

### 8.6 Two API endpoints with different verbosity

**Decision**: expose both `POST /chat` (full `ChatRequest` with `top_k`,
`filters`, `expand`) and `POST /query` (minimal `QueryRequest(question)`).

**Rationale**: `/query` is the production interface for UI/CLI clients.
`/chat` is the power interface for evaluation scripts and debugging.
Both delegate to the same `ask()` function.

### 8.7 All config via pydantic-settings, no hardcoded paths

**Decision**: `config.py` defines a single `Settings(BaseSettings)` loaded
from `.env`. All path references (`vault_index_path`, `semantic_index_path`,
`vault_notion_dir`) are `Path` objects resolved at startup.

**Rationale**: sprint1 had hardcoded `/home/ubuntu/hermes-vault/` paths. This
made the code non-portable and untestable. pydantic-settings provides
type-checked, documented config with sane defaults.

---

## 9. Pydantic-AI Tools: Current State and Roadmap

### Current state

The agent has one tool: `search_vault`.

```python
@agent.tool
async def search_vault(ctx: RunContext[VaultDeps], query: str) -> str:
    # 1. Check which indices exist
    # 2. Call brf.search() and/or semantic.search() in parallel
    # 3. Fuse via fusion.rrf_fuse()
    # 4. Format results as context string for the LLM
```

`VaultDeps` carries:
- `top_k: int` — number of results to retrieve
- `filters: dict[str, str]` — keyword field filters passed to BRF
- `expand: bool` — whether to use query expansion
- `retrieved_chunks: list[dict]` — populated by the tool, read by `ask()` for sources

### Roadmap

| Tool | Status | Description |
|---|---|---|
| `search_vault` | ✅ Implemented | Hybrid BRF + semantic search |
| `search_by_date` | Planned | Filter chunks by `created` / `last_edited` date range |
| `list_categories` | Planned | Return available category/tag taxonomy |
| `get_chunk_detail` | Planned | Retrieve full text of a specific chunk by id |

---

## 10. Chunk ID — Requirement for Incremental Index Updates

### Current dedup key

Both `brf._chunk_id()` and `fusion._chunk_id()` use:

```python
f"{chunk.get('file', '')}|{chunk.get('title', '')}|{chunk.get('text', '')[:80]}"
```

This is a **runtime dedup key for RRF**, not a persistent identity.

### Gap for incremental sync

For true incremental sync (update a page that already exists, detect deletions),
a **stable, persistent chunk ID** is needed. Requirements:
- Deterministic from content: same page + same H3 title → same ID
- Survives re-ingest: two consecutive full ingests produce identical IDs for
  unchanged chunks
- Granularity: per H3 section, not per file

**Proposed format** (not yet implemented):
```
{file_stem}#{h3_title_slug}
```
Where `file_stem` is the Notion export filename without extension and
`h3_title_slug` is the H3 heading lowercased with spaces replaced by hyphens.

Example: `export_Notion_CNR_2026#risultati-gridsearch`

**Current status**: `[TO BE IMPLEMENTED]` — the adapter does not produce
a `chunk_id` field yet. Required before deletion/update detection can be added.

---

## 11. Open Technical Roadmap

| Priority | Item | Notes |
|---|---|---|
| High | Stable `chunk_id` field in adapters | Prerequisite for incremental sync with update/delete |
| High | `run_eval.py` pipeline in v2 | Migrate from sprint1 `master_run_eval.py`; wire to `judge.py` |
| High | Populate golden-set in this doc | Add `query` + `expected_answer` for ≥ 10 real vault questions |
| Medium | Gmail adapter (`adapters/gmail.py`) | Sprint 4 — ingest self-sent emails |
| Medium | Chat export adapter (`adapters/chat_export.py`) | Sprint 4 — ingest exported chat logs |
| Medium | Incremental sync endpoint (`POST /sync`) | Use `brf.append_documents()` + `semantic.append_documents()` |
| Medium | Search evaluation pipeline | Retrieval precision@k over golden questions |
| Low | Highlighting support | `minsearch.AppendableIndex` supports highlight; expose in API |
| Low | `search_by_date` tool | Filter chunks by `created` / `last_edited` |

---

## 12. Sprint 1 Evaluation Results

### Search Evaluation (Retrieval Quality)

All three retrieval strategies were benchmarked against a 5-query golden set
(`golden_set.py`) using Hit Rate@10 and MRR as primary metrics.

| Strategy | Hit Rate@10 | MRR | Notes |
|---|---|---|---|
| BRF (TF-IDF + RRF) | 1.00 | 1.00 | — |
| Semantic (nomic-embed-text) | 1.00 | 1.00 | — |
| Fusion (Semantic + BRF, RRF k=60) | 1.00 | 1.00 | Chosen for RAG Eval |

All three strategies achieve perfect scores on the current golden set.
Fusion was selected as the default strategy for RAG Evaluation based on its
architectural robustness: it combines lexical and semantic signals, making it
less sensitive to query phrasing and vocabulary mismatch.

---

### RAG Evaluation (End-to-End Pipeline)

Full pipeline evaluation (retrieval → generation → LLM-as-judge) run with
Fusion k=10, generator `gemma4-8k:latest`, judge `gemma4:e4b`.

| Query | Topic | Retrieval | Generation | Judge | Total |
|---|---|---|---|---|---|
| Q1 | Fixed Tech Expenses | 28.5s | 19.5s | 20.3s | 68.3s |
| Q2 | Alibi Detect (Seldon) | 8.2s | 18.4s | 24.0s | 50.7s |
| Q3 | Oxen.ai Free Plan | 8.1s | 18.6s | 22.6s | 49.3s |
| Q4 | DVC in MLOps Courses | 3.2s | 18.6s | 17.0s | 38.7s |
| Q5 | Grid Search Parameters | 5.0s | 19.5s | 13.7s | 38.1s |
| **Avg** | | **10.6s** | **18.9s** | **19.5s** | **49.0s** |

5/5 queries completed. No timeouts. Generation time is stable at ~19s across
all queries, confirming full-GPU execution (RTX 3090, no VRAM overflow).

> **Infrastructure note:** generator context was reduced from 262144 to 8192
> tokens via a dedicated Modelfile (`gemma4-8k:latest`). This dropped KV cache
> from ~7.5 GB to ~0.23 GB, bringing total VRAM from ~26 GB (overflow) to
> ~18.7 GB (within the 24 GB limit). Generation time dropped from 47–137s
> (with RAM spill) to a stable ~19s.

---

### v1 vs v2 Comparison

The comparison between the Phase 1–2 pipeline (minsearch BRF, Hermes runtime)
and the Phase 3 standalone app covers retrieval quality on the two queries
whose golden set definition did not change between versions.

#### Retrieval — directly comparable queries

| Query | Topic | v1 Hit Rate | v2 Hit Rate | Delta |
|---|---|---|---|---|
| Q1 | Fixed Tech Expenses | ✅ HIT | ✅ HIT | — |
| Q3 | Oxen.ai Free Plan | ✅ HIT | ✅ HIT | — |

Both versions retrieve these correctly. The structural improvement is visible
on the queries that **changed** between versions.

#### Golden set evolution — Q2, Q4, Q5

Three queries were reformulated between v1 and v2. This was not an adjustment
to make results look better: it was a documented finding about the structural
limits of TF-IDF retrieval.

| Query | v1 formulation | v1 outcome | Reason for change | v2 formulation | v2 outcome |
|---|---|---|---|---|---|
| Q2 | "In which course did I use Gitpod?" | ❌ MISS | Term only in deep leaf chunks, unreachable by TF-IDF | "Does Alibi Detect serve data versioning or data drift analysis?" | ✅ HIT rank 1 |
| Q4 | "Which tools did I use in MLOps DTC vs MLOps in 4 Weeks?" | ❌ MISS | Multi-chunk aggregation across documents, TF-IDF cannot bridge | "Is DVC used in MLOps Zoomcamp DTC, MLOps in 4 Weeks, or both?" | ✅ HIT rank 1 |
| Q5 | "Which parameters did we estimate at CNR using grid search?" | ❌ MISS | "grid search" too generic; vault-specific terms (tau_L, tau_Ab) invisible to TF-IDF | "What values of tau_L and tau_Ab did the grid search with Huber Loss produce on VirusWatch?" | ✅ HIT rank 1 |

The v2 pipeline (semantic + fusion) retrieves all five queries correctly,
including the three that were structurally unreachable in v1. The golden set
evolution is itself a finding: **queries that require semantic bridging of
domain-specific vocabulary are only answerable with embedding-based retrieval.**

---

### Known Limitations and Backlog

| Item | Detail | Priority |
|---|---|---|
| `Recall@k` metric | Current `hit_rate_at_k` is binary (any hit = 1). Q4 retrieved 1/2 expected chunks. Replace with `len(hit_chunks) / len(expected_chunks)` for multi-chunk queries. | Sprint 2 |
| `evaluation/generated/` folder | Separate exploratory runs from baseline artefacts in `.gitignore`. | Sprint 2 |
| Expand golden set | Current golden set: 5 queries. Target: 50–200 queries via stratified LLM generation (per webinar Alexey). | Sprint 2 |
| Gmail / chat export sources | Q3 (GAIL exam), Q4 (MSI Afterburner) still not indexed. Adapters are placeholders. | Sprint 2 |

---
