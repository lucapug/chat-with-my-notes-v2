"""Evaluation runner for Phase 3.

This module loads the Phase 3 golden query set and the ground truth
answers used exclusively by the judge evaluation pipeline.

WARNING: The ground truth file is meant for evaluation only and must never
be passed into any generator model prompt, router context, or retrieval
pipeline. It is loaded only by this runner.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evaluation.golden_set import GOLDEN_SET, GoldenQuery, load_golden_set

logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "evaluation" / "golden-set-ground-truth.json"


def load_golden_queries() -> list[GoldenQuery]:
    """Load the golden queries used by the evaluation pipeline."""
    return GOLDEN_SET


def load_ground_truth(gt_path: str | Path | None = None) -> dict[str, str | None]:
    """Load the ground truth answers for golden queries.

    WARNING: ground truth is loaded here for judge evaluation only.
    WARNING: DO NOT pass these answers into any agent/router/generator prompt.
    WARNING: This file must never be part of the retrieval or generation context.
    """
    if gt_path is None:
        gt_path = DEFAULT_GROUND_TRUTH_PATH

    gt_path = Path(gt_path)

    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "golden_set" not in data:
        raise ValueError(f"Invalid ground truth format in {gt_path}")

    gt_map: dict[str, str | None] = {}
    for item in data["golden_set"]:
        qid = item["id"]
        gt_map[qid] = item.get("expected_answer")

    return gt_map


def extract_chunk_ids(results: list[dict[str, Any]]) -> set[str]:
    """Return the canonical chunk_id values present in a list of search result chunks."""
    chunk_ids: set[str] = set()
    for chunk in results:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            chunk_ids.add(str(chunk_id))
        else:
            logger.warning("Search result missing chunk_id, skipping chunk: %s", chunk)
    return chunk_ids


def match_expected_chunk_ids(results: list[dict[str, Any]], expected_chunk_ids: list[str]) -> list[str]:
    """Return the subset of expected_chunk_ids that appear in the search results by direct chunk_id."""
    actual_ids = extract_chunk_ids(results)
    return [chunk_id for chunk_id in expected_chunk_ids if chunk_id in actual_ids]


def get_ground_truth_for_query(query_id: str, gt_path: str | Path | None = None) -> str | None:
    ground_truth = load_ground_truth(gt_path)
    return ground_truth.get(query_id)


def main() -> int:
    golden_queries = load_golden_queries()
    ground_truth = load_ground_truth()

    print(f"Loaded {len(golden_queries)} golden queries.")
    print(f"Loaded {len(ground_truth)} ground truth entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
