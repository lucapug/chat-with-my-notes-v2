"""Notion adapter — reads exported Markdown files and chunks by H3 sections.

Migrated from adapters/notion.py (sprint1-hermes-backup).
Paths via config.settings. Chunking logic preserved intact.

Implements the chunking strategy from CONCEPT_DOC.md §2:
- Standard notes → per H3 section
- CNR notes with images → per H3 section (H3 = implicit caption)
- Root pages → SKIP (parent_type: workspace)
- Short notes (no H3) → whole document as single chunk
"""

import glob
import json
import logging
import os
import re
from typing import Any

import yaml

from config import settings

logger = logging.getLogger(__name__)

# H3 heading pattern — captures the title after "### "
H3_PATTERN = re.compile(r"^###\s+(.+)$", re.MULTILINE)

# Image reference pattern in Markdown
IMAGE_PATTERN = re.compile(r"!\[.*?\]\(")


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and return (metadata_dict, body_text).

    Returns ({}, full_content) if no valid frontmatter is found.
    """
    if not content.startswith("---\n"):
        return {}, content

    end = content.find("\n---\n", 4)
    if end == -1:
        return {}, content

    fm_str = content[4:end]
    body = content[end + 5:]

    try:
        metadata = yaml.safe_load(fm_str) or {}
    except yaml.YAMLError:
        return {}, content

    return metadata, body


def _detect_has_images(text: str) -> bool:
    """Check if text contains Markdown image references."""
    return bool(IMAGE_PATTERN.search(text))


def _detect_language(metadata: dict, text: str) -> str:
    """Determine language from frontmatter or heuristic."""
    if "language" in metadata and metadata["language"]:
        return metadata["language"]
    italian_markers = re.compile(
        r"\b(che|per|una|del|della|dei|degli|sono|stato|stata|anche|ancora|perché|più|può|questo|questa|quello)\b",
        re.IGNORECASE,
    )
    if italian_markers.search(text):
        return "it"
    return "en"


def _build_chunk(
    text: str,
    metadata: dict,
    filename: str,
    h3_title: str | None = None,
) -> dict:
    """Build a uniform document dict from chunk text and file metadata."""
    if h3_title:
        title = h3_title
    elif metadata.get("title"):
        title = metadata["title"]
    else:
        # Derive from filename: export_Notion_Slug_07052026.md → Slug
        stem = os.path.splitext(filename)[0]
        parts = stem.split("_")
        slug_parts = []
        for p in parts[2:]:  # skip "export" and "Notion"
            if re.fullmatch(r"\d{8}", p):
                break
            slug_parts.append(p)
        title = " ".join(slug_parts) if slug_parts else stem

    # Tags: minsearch keyword fields must be strings, so serialize list
    tags_val = metadata.get("tags", [])
    if isinstance(tags_val, list):
        tags_str = json.dumps(tags_val, ensure_ascii=False)
    else:
        tags_str = str(tags_val) if tags_val else ""

    return {
        "title": title,
        "text": text,
        "source": metadata.get("source", "notion"),
        "category": metadata.get("category", ""),
        "subcategory": metadata.get("subcategory") or "",
        "type": metadata.get("type", "nota"),
        "file": filename,
        "language": _detect_language(metadata, text),
        "created": metadata.get("created", ""),
        "last_edited": metadata.get("last_edited", ""),
        "has_images": _detect_has_images(text) if not metadata.get("has_images") else True,
        "has_ai_callouts": metadata.get("has_ai_callouts", False),
        "context": metadata.get("context") or "",
        "tags": tags_str,
    }


def _chunk_by_h3(body: str, filename: str, metadata: dict) -> list[dict]:
    """Split body into chunks at H3 boundaries.

    Each H3 section becomes a separate chunk. Content before the first H3
    (preamble) is attached to the first H3 chunk, or becomes a standalone
    chunk if there are no H3 headings at all.
    """
    h3_matches = list(H3_PATTERN.finditer(body))

    if not h3_matches:
        return [_build_chunk(body.strip(), metadata, filename)]

    chunks = []
    first_h3_start = h3_matches[0].start()
    preamble = body[:first_h3_start].strip()

    for i, match in enumerate(h3_matches):
        h3_title = match.group(1).strip()
        section_start = match.start()
        section_end = h3_matches[i + 1].start() if i + 1 < len(h3_matches) else len(body)
        section_text = body[section_start:section_end].strip()

        if i == 0 and preamble:
            section_text = preamble + "\n\n" + section_text

        chunks.append(_build_chunk(section_text, metadata, filename, h3_title=h3_title))

    return chunks


def _is_root_page(metadata: dict, root_titles: set[str]) -> bool:
    """Stub: root pages are now exported intentionally. Always returns False."""
    return False


def ingest() -> list[dict[str, Any]]:
    """Read all Notion exports, chunk by H3, and return uniform documents."""
    notion_dir = str(settings.vault_notion_dir)
    pages_map_path = str(settings.vault_pages_map)

    root_titles: set[str] = set()
    try:
        with open(pages_map_path, encoding="utf-8") as f:
            pages_data = json.load(f)
        root_titles = {
            p["title"] for p in pages_data
            if p.get("parent_type") == "workspace"
        }
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not load pages map (%s), root page detection disabled", e)

    md_files = sorted(glob.glob(os.path.join(notion_dir, "*.md")))
    if not md_files:
        logger.warning("No .md files found in %s", notion_dir)
        return []

    documents: list[dict] = []
    stats = {
        "files_scanned": 0,
        "files_skipped_empty": 0,
        "files_skipped_root": 0,
        "h3_chunks": 0,
        "single_chunk_docs": 0,
        "errors": 0,
    }

    for filepath in md_files:
        stats["files_scanned"] += 1
        filename = os.path.basename(filepath)

        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.error("Error reading %s: %s", filename, e)
            stats["errors"] += 1
            continue

        if not content.strip():
            stats["files_skipped_empty"] += 1
            continue

        metadata, body = _parse_frontmatter(content)

        if _is_root_page(metadata, root_titles):
            stats["files_skipped_root"] += 1
            continue

        if not body.strip():
            stats["files_skipped_empty"] += 1
            continue

        chunks = _chunk_by_h3(body, filename, metadata)

        if len(chunks) == 1:
            stats["single_chunk_docs"] += 1
        else:
            stats["h3_chunks"] += len(chunks)

        documents.extend(chunks)

    total_chunks = stats["h3_chunks"] + stats["single_chunk_docs"]
    logger.info(
        "Notion ingest: %d files scanned, %d total chunks (%d H3, %d single), "
        "%d skipped empty, %d errors",
        stats["files_scanned"],
        total_chunks,
        stats["h3_chunks"],
        stats["single_chunk_docs"],
        stats["files_skipped_empty"],
        stats["errors"],
    )

    return documents
