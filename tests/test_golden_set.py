from evaluation.golden_set import GoldenQuery, load_golden_set, parse_golden_set_block


def test_parse_golden_set_block_preserves_original_fields() -> None:
    raw_doc = '''Some text before
```golden-set
[
  {
    "id": "QX",
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
    assert query.id == "QX"
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


def test_load_golden_set_contains_expected_chunk_ids_for_q1_q2_q3_q4_q5() -> None:
    queries = load_golden_set()
    q1 = next(q for q in queries if q.id == "Q1")
    q2 = next(q for q in queries if q.id == "Q2")
    q3 = next(q for q in queries if q.id == "Q3")
    q4 = next(q for q in queries if q.id == "Q4")
    q5 = next(q for q in queries if q.id == "Q5")

    assert q1.expected_chunk_ids == [
        "76ada2eed43a4afc7a73d2247cd0e84608db75e2",
    ]
    assert q2.expected_chunk_ids == [
        "e411cd037e6dee95a6972a30f365fcae94eecbd3",
    ]
    assert q3.expected_chunk_ids == [
        "4d313c5e085aeaab2fd991f0544f862842f99cf3",
    ]
    assert q4.expected_chunk_ids == [
        "2ded0e693f64be5b54cda41b20a24b50d4f185ba",
        "db991e4d79f9f6b641e6b08521dc0e58a61a98d3",
    ]
    assert q5.expected_chunk_ids == [
        "094d931cec9d08e9c78809375365e553eabfa70c",
        "7fe2bf9a4b3df99362f4acbc28d433242ce0607a",
    ]
