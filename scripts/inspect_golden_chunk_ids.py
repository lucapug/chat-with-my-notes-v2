#!/usr/bin/env python3
"""Inspect golden-set expected_chunk_ids and map them to actual chunk_id values.

This helper is meant to be run once before updating `evaluation/golden_set.py`.
It reads `data/vault_index.pkl`, loads the golden set, and resolves each
expected_chunk_id value to a real `chunk_id` where possible.

Usage:
  python scripts/inspect_golden_chunk_ids.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import minsearch
from config import settings
from evaluation.golden_set import load_golden_set

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
KEY_RE = re.compile(r"^(?P<file>[^|]+)\|(?P<title>.+?)\|(?P<sub_index>\d+)$")


def derive_chunk_id_from_expected(expected: str) -> str | None:
    """Convert a legacy expected key into the actual SHA1 chunk_id if possible."""
    if SHA1_RE.fullmatch(expected):
        return expected
    match = KEY_RE.fullmatch(expected)
    if not match:
        return None
    file = match.group("file")
    title = match.group("title")
    sub_index = int(match.group("sub_index"))
    raw = f"{file}|{title}|{sub_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_indexes(docs: list[dict[str, str]]) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Build lookup dictionaries for file and file/title matching."""
    by_file: dict[str, list[dict[str, str]]] = {}
    by_file_title: dict[str, list[dict[str, str]]] = {}
    for doc in docs:
        file = doc.get("file", "")
        title = doc.get("title", "")
        by_file.setdefault(file, []).append(doc)
        by_file_title.setdefault(f"{file}|{title}", []).append(doc)
    return {"by_file": by_file, "by_file_title": by_file_title}


def resolve_expected_chunk_id(expected: str, docs: list[dict[str, str]], index_by: dict[str, dict[str, list[dict[str, str]]]]) -> dict[str, object]:
    """Resolve one expected_chunk_id entry to actual candidates."""
    result: dict[str, object] = {
        "expected": expected,
        "resolved_chunk_id": None,
        "match_type": None,
        "candidates": [],
    }

    derived = derive_chunk_id_from_expected(expected)
    if derived is not None:
        if any(doc.get("chunk_id") == derived for doc in docs):
            result.update({"resolved_chunk_id": derived, "match_type": "derived_sha1"})
            return result
        result["resolved_chunk_id"] = derived
        result["match_type"] = "derived_sha1_not_found"
        return result

    # Fallback: try file|title match
    parts = expected.split("|")
    if len(parts) >= 2:
        file = parts[0]
        title = "|".join(parts[1:])
        key = f"{file}|{title}"
        candidates = index_by["by_file_title"].get(key, [])
        if candidates:
            result["candidates"] = [doc["chunk_id"] for doc in candidates]
            result["match_type"] = "file_title_candidates"
            return result

    # Fallback: match by file only
    file = parts[0]
    candidates = index_by["by_file"].get(file, [])
    if candidates:
        result["candidates"] = [doc["chunk_id"] for doc in candidates]
        result["match_type"] = "file_candidates"
        return result

    result["match_type"] = "no_match"
    return result


def main() -> int:
    index_path = Path(settings.vault_index_path)
    if not index_path.exists():
        print(f"ERROR: vault index not found at {index_path}")
        return 1

    index = minsearch.AppendableIndex.load(str(index_path))
    docs = getattr(index, "docs", [])
    if not docs:
        print(f"ERROR: loaded index has no docs")
        return 2

    index_by = build_indexes(docs)
    golden_queries = load_golden_set()

    print(f"Loaded {len(docs)} chunks from {index_path}")
    print(f"Loaded {len(golden_queries)} golden queries")
    print()

    for query in golden_queries:
        print(f"=== {query.id} — {query.topic}")
        if not query.expected_chunk_ids:
            print("  no expected_chunk_ids")
            print()
            continue

        for expected in query.expected_chunk_ids:
            resolved = resolve_expected_chunk_id(expected, docs, index_by)
            print(f"  expected: {expected}")
            print(f"    match_type: {resolved['match_type']}")
            if resolved["resolved_chunk_id"]:
                print(f"    resolved_chunk_id: {resolved['resolved_chunk_id']}")
            if resolved["candidates"]:
                print(f"    candidates: {resolved['candidates']}")
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
