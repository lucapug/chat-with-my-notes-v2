from run_eval import extract_chunk_ids, match_expected_chunk_ids


def test_extract_chunk_ids_skips_missing_chunk_id() -> None:
    results = [
        {"chunk_id": "abc123"},
        {"file": "export_Notion.md", "title": "Test"},
    ]

    assert extract_chunk_ids(results) == {"abc123"}


def test_match_expected_chunk_ids_direct_chunk_id() -> None:
    results = [
        {"chunk_id": "abc123"},
        {"chunk_id": "def456"},
    ]

    matched = match_expected_chunk_ids(results, ["abc123", "missing", "def456"])
    assert matched == ["abc123", "def456"]
