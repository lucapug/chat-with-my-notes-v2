import httpx
import pytest
from fastapi.testclient import TestClient

import api.router as api_router
from api.schemas import ChatRequest, ChatResponse, ChunkResult
from main import app


def test_ingest_offline_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_ingest(source: str = "notion", output_path: str | None = None) -> int:
        return 3

    monkeypatch.setattr(api_router, "run_ingest", fake_run_ingest)
    client = TestClient(app)
    resp = client.post("/ingest", json={"source": "notion"})
    assert resp.status_code == 200
    assert resp.json() == {"documents_indexed": 3, "source": "notion"}


def test_query_offline_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ask(request: ChatRequest, usage_limits=None) -> ChatResponse:
        return ChatResponse(
            answer="mocked answer",
            sources=[
                ChunkResult(
                    title="test note",
                    text_snippet="test content",
                    source="notion",
                    category="test",
                    file="test.md",
                )
            ],
        )

    monkeypatch.setattr(api_router, "ask", fake_ask)
    client = TestClient(app)
    resp = client.post("/query", json={"question": "hello"})
    assert resp.status_code == 200
    # /query endpoint extracts source titles from ChatResponse.sources
    assert resp.json() == {"answer": "mocked answer", "sources": ["test note"]}


@pytest.mark.integration
def test_ingest_integration_smoke() -> None:
    resp = httpx.post(
        "http://localhost:8000/ingest",
        json={"source": "gmail"},
        timeout=120.0,
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("source") == "gmail"
    assert isinstance(payload.get("documents_indexed"), int)


@pytest.mark.integration
def test_query_integration_smoke() -> None:
    resp = httpx.post(
        "http://localhost:8000/query",
        json={"question": "Quali sono le mie spese?"},
        timeout=360.0,
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload.get("answer"), str)
    assert isinstance(payload.get("sources"), list)
