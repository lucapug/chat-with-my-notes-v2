"""Golden queries for Phase 3 evaluation.

This module contains the bilingual Golden Question set used by the
RAG evaluation pipeline and the upcoming Search Eval pipeline.

Design rationale:
- User-facing queries are in Italian.
- Each query is translated internally to English via gemma4:e4b,
  enabling dual-language TF-IDF retrieval.
- The system performs both Italian and English retrieval,
  then fuses rankings with Reciprocal Rank Fusion (RRF, k=60).
- The bilingual layer is an implementation detail; the user always
  interacts in Italian.
- `expected_chunk_ids` is included for Week 5 Search Eval once chunk_id
  stability is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query_it: str
    query_en: str
    topic: str
    source_hint: str
    characteristic: str
    query_type: str
    target_sources: str
    expected_chunk_ids: list[str] = field(default_factory=list)


GOLDEN_SET: list[GoldenQuery] = [
    GoldenQuery(
        id="Q1",
        query_it="Quali sono le mie spese tecnologiche fisse annuali per il lavoro?",
        query_en="What are my fixed annual technology expenses for work?",
        topic="Spese Tecnologiche",
        source_hint="Cross-source (Notion + Gmail)",
        characteristic="Numerically verifiable",
        query_type="aggregation",
        target_sources="data/notion/ + data/gmail_self/",
    ),
    GoldenQuery(
        id="Q2",
        query_it="In quale corso ho utilizzato Gitpod?",
        query_en="Which course did I use Gitpod in?",
        topic="Gitpod",
        source_hint="data/notion/",
        characteristic="Deep retrieval on nested subpages",
        query_type="lookup",
        target_sources="data/notion/",
    ),
    GoldenQuery(
        id="Q3",
        query_it="Quanti repositories privati sono inclusi nel piano free di Oxen.ai?",
        query_en="How many private repositories are included in the Oxen.ai free plan?",
        topic="Oxen.ai",
        source_hint="data/notion/",
        characteristic="Cross-language BRF + callout content",
        query_type="lookup",
        target_sources="data/notion/",
    ),
    GoldenQuery(
        id="Q4",
        query_it="Quali differenti tools e tecnologie ho usato nel corso MLOps di DTC rispetto a quelle usate in MLOps in 4 Weeks?",
        query_en="What different tools and technologies did I use in the DTC MLOps course compared to those used in MLOps in 4 Weeks?",
        topic="MLOps Tools Comparison",
        source_hint="data/notion/",
        characteristic="Multi-chunk aggregation and comparison",
        query_type="comparison",
        target_sources="data/notion/",
    ),
    GoldenQuery(
        id="Q5",
        query_it="Quali parametri abbiamo stimato durante il lavoro al CNR usando la grid search?",
        query_en="What parameters did we estimate during CNR work using grid search?",
        topic="Parametri Grid Search",
        source_hint="data/notion/ CNR + data/chat_exports/chatgpt/",
        characteristic="Explicit cross-source synthesis",
        query_type="aggregation",
        target_sources="data/notion/ CNR + data/chat_exports/chatgpt/",
    ),
]


def load_golden_set() -> list[GoldenQuery]:
    """Return the static Golden Question set for evaluation."""
    return GOLDEN_SET
