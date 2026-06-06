"""Evaluation runner for Phase 3.

This module loads the Phase 3 golden query set and runs Search Eval against
BRF, semantic, and hybrid retrieval.

The search evaluation pipeline compares retrieved chunk_ids with the
expected chunk_ids from the golden set. Output is written to a JSON file
under evaluation/ and printed as a human-readable summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings
from evaluation.golden_set import GOLDEN_SET, GoldenQuery
from src.search import brf, semantic
from src.search.fusion import rrf_fuse

logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "evaluation" / "golden-set-ground-truth.json"
DEFAULT_SEARCH_EVAL_OUTPUT = Path(__file__).resolve().parent / "evaluation" / "search_eval.json"
VALID_MODES = {"brf", "semantic", "fusion", "all"}
SEARCH_EVAL_SCHEMA_VERSION = "v1"


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


def extract_chunk_ids(results: list[dict[str, Any]]) -> list[str]:
    """Return the ordered canonical chunk_id values present in a list of search result chunks."""
    chunk_ids: list[str] = []
    for chunk in results:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            chunk_ids.append(str(chunk_id))
        else:
            logger.warning("Search result missing chunk_id, skipping chunk: %s", chunk)
    return chunk_ids


def match_expected_chunk_ids(results: list[dict[str, Any]], expected_chunk_ids: list[str]) -> list[str]:
    """Return the subset of expected_chunk_ids that appear in the search results by direct chunk_id."""
    actual_ids = set(extract_chunk_ids(results))
    return [chunk_id for chunk_id in expected_chunk_ids if chunk_id in actual_ids]


def compute_query_metrics(expected_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> dict[str, Any]:
    """Compute search evaluation metrics for a single query."""
    hit_chunk_ids = [cid for cid in retrieved_chunk_ids if cid in expected_chunk_ids]
    first_hit_rank = next(
        (rank for rank, cid in enumerate(retrieved_chunk_ids, start=1) if cid in expected_chunk_ids),
        None,
    )
    reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return {
        "expected_chunk_ids": expected_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "hit_chunk_ids": hit_chunk_ids,
        "hit_rate_at_k": 1 if hit_chunk_ids else 0,
        "reciprocal_rank": reciprocal_rank,
        "first_hit_rank": first_hit_rank,
        "rag_eval": None,
    }


def summarize_mode_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total_queries = len(metrics)
    queries_with_hits = sum(1 for m in metrics if m["hit_rate_at_k"] > 0)
    queries_without_hits = total_queries - queries_with_hits
    average_hit_rate = sum(m["hit_rate_at_k"] for m in metrics) / total_queries if total_queries else 0.0
    mrr = sum(m["reciprocal_rank"] for m in metrics) / total_queries if total_queries else 0.0

    return {
        "total_queries": total_queries,
        "queries_with_hits": queries_with_hits,
        "queries_without_hits": queries_without_hits,
        "average_hit_rate": average_hit_rate,
        "mrr": mrr,
    }


def format_index_meta() -> dict[str, str]:
    return {
        "vault_index_path": str(settings.vault_index_path),
        "semantic_index_path": str(settings.semantic_index_path),
    }


def build_run_id(mode: str) -> str:
    now = datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"search_eval_{mode}_{now}"


async def retrieve_results(query: GoldenQuery, top_k: int, mode: str) -> list[dict[str, Any]]:
    query_it = query.query_it

    if mode == "brf":
        return await brf.search(query_it, top_k=top_k)
    if mode == "semantic":
        return await semantic.search(query_it, top_k=top_k)
    if mode == "fusion":
        brf_results = await brf.search(query_it, top_k=top_k)
        semantic_results = await semantic.search(query_it, top_k=top_k)
        return rrf_fuse(brf_results, semantic_results, top_k)

    raise ValueError(f"Unsupported mode: {mode}")


async def run_search_eval(golden_queries: list[GoldenQuery], top_k: int, mode: str) -> dict[str, Any]:
    run_modes = [mode] if mode != "all" else ["brf", "semantic", "fusion"]
    all_results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    per_mode_metrics: dict[str, list[dict[str, Any]]] = {m: [] for m in run_modes}

    for query in golden_queries:
        query_results: dict[str, Any] = {
            "question": query.id,
            "query": query.query,
            "topic": query.topic,
            "expected_chunk_ids": query.expected_chunk_ids,
            "runs": {},
            "rag_eval": None,
        }

        for run_mode in run_modes:
            results = await retrieve_results(query, top_k, run_mode)
            retrieved_ids = extract_chunk_ids(results)
            metrics = compute_query_metrics(query.expected_chunk_ids, retrieved_ids)
            query_results["runs"][run_mode] = metrics
            per_mode_metrics[run_mode].append(metrics)

        all_results.append(query_results)

    for run_mode in run_modes:
        summary[run_mode] = summarize_mode_metrics(per_mode_metrics[run_mode])

    return {
        "run_id": build_run_id(mode),
        "schema_version": SEARCH_EVAL_SCHEMA_VERSION,
        "date": datetime.now(datetime.UTC).isoformat() + "Z",
        "mode": mode,
        "top_k": top_k,
        "score_scale": "hit_rate_at_k,mrr",
        "index_meta": format_index_meta(),
        "summary": summary,
        "results": all_results,
        "rag_eval": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Search Eval over the golden query set.")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="all", help="Search mode to evaluate.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved chunks to evaluate.")
    parser.add_argument("--output", type=Path, default=DEFAULT_SEARCH_EVAL_OUTPUT, help="Output JSON file path.")
    return parser.parse_args()


def print_summary(report: dict[str, Any]) -> None:
    print("Search Eval Summary")
    print("===================")
    print(f"run_id: {report['run_id']}")
    print(f"mode: {report['mode']}")
    print(f"top_k: {report['top_k']}")
    print(f"date: {report['date']}")
    print("")

    for mode, summary in report["summary"].items():
        print(f"Mode: {mode}")
        print(f"  total_queries: {summary['total_queries']}")
        print(f"  queries_with_hits: {summary['queries_with_hits']}")
        print(f"  queries_without_hits: {summary['queries_without_hits']}")
        print(f"  average_hit_rate: {summary['average_hit_rate']:.3f}")
        print(f"  mrr: {summary['mrr']:.3f}")
        print("")


def save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote search eval report to {output_path}")


def main() -> int:
    args = parse_args()
    if args.mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {args.mode}")

    golden_queries = list(GOLDEN_SET)
    report = asyncio.run(run_search_eval(golden_queries, args.top_k, args.mode))
    print_summary(report)
    save_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
