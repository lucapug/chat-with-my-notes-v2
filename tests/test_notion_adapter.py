import json

from src.ingest.adapters import notion
from config import settings


def test_ingest_single_chunk_no_h3(tmp_path, monkeypatch) -> None:
    md = """---
title: Test Note
category: personal
---
This is a single chunk.
"""
    md_path = tmp_path / "export_Notion_Test_07052026.md"
    md_path.write_text(md, encoding="utf-8")

    pages_map = tmp_path / "notion_pages_map.json"
    pages_map.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(settings, "vault_notion_dir", tmp_path)
    monkeypatch.setattr(settings, "vault_pages_map", pages_map)

    docs = notion.ingest()
    assert len(docs) == 1
    assert docs[0]["title"] == "Test Note"
    assert "single chunk" in docs[0]["text"]


def test_ingest_h3_chunks_with_preamble(tmp_path, monkeypatch) -> None:
    md = """---
category: work
---
Intro preamble.

### Section One
Content one.

### Section Two
Content two.
"""
    md_path = tmp_path / "export_Notion_Work_07052026.md"
    md_path.write_text(md, encoding="utf-8")

    pages_map = tmp_path / "notion_pages_map.json"
    pages_map.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(settings, "vault_notion_dir", tmp_path)
    monkeypatch.setattr(settings, "vault_pages_map", pages_map)

    docs = notion.ingest()
    assert len(docs) == 2
    assert docs[0]["title"] == "Section One"
    assert "Intro preamble" in docs[0]["text"]
    assert docs[1]["title"] == "Section Two"


def test_ingest_tags_serialized(tmp_path, monkeypatch) -> None:
    md = """---
tags:
  - alpha
  - beta
---
Body.
"""
    md_path = tmp_path / "export_Notion_Tags_07052026.md"
    md_path.write_text(md, encoding="utf-8")

    pages_map = tmp_path / "notion_pages_map.json"
    pages_map.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(settings, "vault_notion_dir", tmp_path)
    monkeypatch.setattr(settings, "vault_pages_map", pages_map)

    docs = notion.ingest()
    assert len(docs) == 1
    assert json.loads(docs[0]["tags"]) == ["alpha", "beta"]


def test_ingest_language_detection_italian(tmp_path, monkeypatch) -> None:
    md = """---
---
Questo è un testo che parla di una cosa.
"""
    md_path = tmp_path / "export_Notion_Lang_07052026.md"
    md_path.write_text(md, encoding="utf-8")

    pages_map = tmp_path / "notion_pages_map.json"
    pages_map.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(settings, "vault_notion_dir", tmp_path)
    monkeypatch.setattr(settings, "vault_pages_map", pages_map)

    docs = notion.ingest()
    assert len(docs) == 1
    assert docs[0]["language"] == "it"


def test_ingest_skips_empty_file(tmp_path, monkeypatch) -> None:
    md_path = tmp_path / "export_Notion_Empty_07052026.md"
    md_path.write_text("", encoding="utf-8")

    pages_map = tmp_path / "notion_pages_map.json"
    pages_map.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(settings, "vault_notion_dir", tmp_path)
    monkeypatch.setattr(settings, "vault_pages_map", pages_map)

    docs = notion.ingest()
    assert docs == []
