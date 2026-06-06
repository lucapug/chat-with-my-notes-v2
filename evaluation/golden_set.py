"""Golden queries for Phase 3 evaluation.

This module loads the golden-set fenced JSON block from the canonical
project document `docs/CONCEPT_DOC.md`.

Design rationale:
- The golden question set is curated in the source-of-truth documentation.
- Evaluation code reads the JSON block directly instead of duplicating data.
- Queries are authored in Italian for the user-facing interface.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONCEPT_DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "CONCEPT_DOC.md"
GOLDEN_SET_FENCE_RE = re.compile(r"^```golden-set\s*\n(.*?)\n```", re.DOTALL | re.MULTILINE)

# TODO Sprint 2: Q1 depends on Gmail source and Q5 depends on ChatGPT export source.
# Both data sources are empty in Sprint 1, so the golden set mapping must be revisited.
EXPECTED_CHUNK_ID_OVERRIDES: dict[str, list[str]] = {
    "Q2": [
        "3feba411fe4cf3c673ba0dd1e14a23d5d0952f4e"
    ],
    "Q4": [
        "51f516af9113a69796ce8e195f5d3a8b363dedf9",
        "db991e4d79f9f6b641e6b08521dc0e58a61a98d3",
    ],
}


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    topic: str
    query: str
    query_it: str
    query_en: str
    source_hint: str
    characteristic: str
    query_type: str
    target_sources: str
    expected_chunk_ids: list[str]
    expected_answer: str | None = None


def parse_golden_set_block(text: str) -> list[GoldenQuery]:
    match = GOLDEN_SET_FENCE_RE.search(text)
    if not match:
        raise ValueError("Could not find a `golden-set` fenced JSON block in CONCEPT_DOC.md")

    block = match.group(1).strip()
    if block == "[TO BE COMPLETED]":
        raise ValueError("The `golden-set` block in CONCEPT_DOC.md is not completed")

    try:
        raw_list = json.loads(block)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in `golden-set` block of CONCEPT_DOC.md") from exc

    if not isinstance(raw_list, list):
        raise ValueError("The `golden-set` block must contain a JSON list")

    queries: list[GoldenQuery] = []
    for item in raw_list:
        if not isinstance(item, dict):
            raise ValueError("Each golden-set entry must be a JSON object")

        query = str(item["query"])
        expected_chunk_ids = item.get("expected_chunk_ids", [])
        if expected_chunk_ids is None:
            expected_chunk_ids = []
        elif not isinstance(expected_chunk_ids, list):
            raise ValueError("`expected_chunk_ids` must be a list if provided")

        query_id = str(item["id"])
        if not expected_chunk_ids and query_id in EXPECTED_CHUNK_ID_OVERRIDES:
            expected_chunk_ids = EXPECTED_CHUNK_ID_OVERRIDES[query_id]

        queries.append(
            GoldenQuery(
                id=str(item["id"]),
                topic=str(item["topic"]),
                query=query,
                query_it=str(item.get("query_it", query)),
                query_en=str(item.get("query_en", "")),
                source_hint=str(item.get("source_hint", "")),
                characteristic=str(item.get("characteristic", "")),
                query_type=str(item.get("query_type", "")),
                target_sources=str(item.get("target_sources", "")),
                expected_chunk_ids=[str(x) for x in expected_chunk_ids],
                expected_answer=item.get("expected_answer"),
            )
        )

    return queries


def load_golden_set(concept_doc_path: str | Path | None = None) -> list[GoldenQuery]:
    """Return the golden questions by parsing `docs/CONCEPT_DOC.md`."""
    if concept_doc_path is None:
        concept_doc_path = DEFAULT_CONCEPT_DOC_PATH

    text = Path(concept_doc_path).read_text(encoding="utf-8")
    return parse_golden_set_block(text)


GOLDEN_SET: list[GoldenQuery] = load_golden_set()
