from evaluation.judge import extract_judge_scores


def test_extract_judge_scores_from_content_json() -> None:
    msg = {
        "content": '{"accuracy": 5, "completeness": 4, "hallucination": 5, "relevance": 5, "outcome": "pass", "reasoning": "ok"}',
        "reasoning": "",
    }
    result = extract_judge_scores(msg)
    assert result["outcome"] == "pass"
    assert result["accuracy"] == 5


def test_extract_judge_scores_from_reasoning_json() -> None:
    msg = {
        "content": "",
        "reasoning": "```json\n{\"accuracy\": 3, \"completeness\": 3, \"hallucination\": 3, \"relevance\": 3, \"outcome\": \"warning\", \"reasoning\": \"ok\"}\n```",
    }
    result = extract_judge_scores(msg)
    assert result["outcome"] == "warning"
    assert result["relevance"] == 3


def test_extract_judge_scores_regex_fallback() -> None:
    msg = {
        "content": "",
        "reasoning": "accuracy: 4\ncompleteness: 5\nhallucination: 4\nrelevance: 5\nResult: pass",
    }
    result = extract_judge_scores(msg)
    assert result["outcome"] == "pass"
    assert result["completeness"] == 5


def test_extract_judge_scores_parse_error() -> None:
    msg = {"content": "", "reasoning": "n/a"}
    result = extract_judge_scores(msg)
    assert result.get("parse_error") is True
    assert "raw_content" in result
    assert "raw_reasoning" in result


def test_extract_judge_scores_from_truncated_json_block() -> None:
    msg = {
        "content": "```json\n{\n  \"accuracy\": 3,\n  \"completeness\": 3,\n  \"hallucination\": 5,\n  \"relevance\": 5,\n  \"outcome\": \"warning\",\n  \"reasoning\": \"La risposta è corretta fino a un certo punto",
    }
    result = extract_judge_scores(msg)
    assert result["outcome"] == "warning"
    assert result["accuracy"] == 3
    assert result["relevance"] == 5
