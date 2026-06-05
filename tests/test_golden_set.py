from evaluation.golden_set import GoldenQuery, load_golden_set, parse_golden_set_block


def test_parse_golden_set_block_preserves_original_fields() -> None:
    raw_doc = '''Some text before
```golden-set
[
  {
    "id": "Q1",
    "topic": "Test",
    "query": "Domanda di test?",
    "expected_answer": "Risposta attesa"
  }
]
```
Some text after
'''

    queries = parse_golden_set_block(raw_doc)
    assert len(queries) == 1
    query = queries[0]
    assert isinstance(query, GoldenQuery)
    assert query.id == "Q1"
    assert query.topic == "Test"
    assert query.query == "Domanda di test?"
    assert query.query_it == "Domanda di test?"
    assert query.query_en == ""
    assert query.source_hint == ""
    assert query.characteristic == ""
    assert query.query_type == ""
    assert query.target_sources == ""
    assert query.expected_chunk_ids == []
    assert query.expected_answer == "Risposta attesa"


def test_load_golden_set_from_concept_doc_contains_all_fields() -> None:
    queries = load_golden_set()
    assert len(queries) > 0
    for query in queries:
        assert isinstance(query.id, str) and query.id
        assert isinstance(query.topic, str) and query.topic
        assert isinstance(query.query, str) and query.query
        assert isinstance(query.query_it, str)
        assert isinstance(query.query_en, str)
        assert isinstance(query.source_hint, str)
        assert isinstance(query.characteristic, str)
        assert isinstance(query.query_type, str)
        assert isinstance(query.target_sources, str)
        assert isinstance(query.expected_chunk_ids, list)
        assert query.expected_answer is None or isinstance(query.expected_answer, str)


def test_load_golden_set_contains_expected_chunk_ids_for_q2_and_q4() -> None:
    queries = load_golden_set()
    q2 = next(q for q in queries if q.id == "Q2")
    q4 = next(q for q in queries if q.id == "Q4")

    assert q2.expected_chunk_ids == [
        "export_Notion_Week1KickstartingAnMlProject_24052026.md|Manage project dependencies with `pip` and `venv`|0"
    ]
    assert q4.expected_chunk_ids == [
        "export_Notion_W4WorkflowOrchestrationPrefect_24052026.md|Prefect config and Profiles|1",
        "export_Notion_Week1KickstartingAnMlProject_24052026.md|Move data versioning from Git to DVC|0",
    ]
