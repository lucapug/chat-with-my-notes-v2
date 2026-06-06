from run_eval import (
    compute_query_metrics,
    extract_chunk_ids,
    match_expected_chunk_ids,
    summarize_mode_metrics,
)


def test_extract_chunk_ids_skips_missing_chunk_id() -> None:
    results = [
        {"chunk_id": "abc123"},
        {"file": "export_Notion.md", "title": "Test"},
    ]

    assert extract_chunk_ids(results) == ["abc123"]


def test_match_expected_chunk_ids_direct_chunk_id() -> None:
    results = [
        {"chunk_id": "abc123"},
        {"chunk_id": "def456"},
    ]

    matched = match_expected_chunk_ids(results, ["abc123", "missing", "def456"])
    assert matched == ["abc123", "def456"]


def test_compute_query_metrics_no_hits() -> None:
    metrics = compute_query_metrics(
        expected_chunk_ids=["abc", "def"],
        retrieved_chunk_ids=["ghi", "jkl"],
    )

    assert metrics["hit_rate_at_k"] == 0
    assert metrics["reciprocal_rank"] == 0.0
    assert metrics["first_hit_rank"] is None
    assert metrics["hit_chunk_ids"] == []


def test_compute_query_metrics_with_hits() -> None:
    metrics = compute_query_metrics(
        expected_chunk_ids=["x", "y"],
        retrieved_chunk_ids=["a", "y", "x"],
    )

    assert metrics["hit_rate_at_k"] == 1
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["first_hit_rank"] == 2
    assert metrics["hit_chunk_ids"] == ["y", "x"]


def test_summarize_mode_metrics() -> None:
    summary = summarize_mode_metrics([
        {"hit_rate_at_k": 1, "reciprocal_rank": 1.0},
        {"hit_rate_at_k": 0, "reciprocal_rank": 0.0},
        {"hit_rate_at_k": 1, "reciprocal_rank": 0.5},
    ])

    assert summary["total_queries"] == 3
    assert summary["queries_with_hits"] == 2
    assert summary["queries_without_hits"] == 1
    assert summary["average_hit_rate"] == 2 / 3
    assert summary["mrr"] == (1.0 + 0.0 + 0.5) / 3
