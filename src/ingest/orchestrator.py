"""Ingest orchestrator — migrated from vault_ingest.py (sprint1-hermes-backup).

Collects documents from configured adapters, builds a minsearch index,
and saves it to settings.vault_index_path (or --output if run as CLI).

Usage:
    python -m src.ingest.orchestrator --source notion
    python -m src.ingest.orchestrator --source all
    python -m src.ingest.orchestrator --source notion --output /path/to/index.pkl
"""

import argparse
import asyncio
import logging
import time
from importlib import import_module

import minsearch

from config import settings
from src.search import semantic

logger = logging.getLogger(__name__)

ADAPTERS: dict[str, tuple[str, str]] = {
    "notion": ("src.ingest.adapters.notion", "Notion"),
    "gmail": ("src.ingest.adapters.gmail", "Gmail Self"),
    "chat_export": ("src.ingest.adapters.chat_export", "Chat Export"),
}

TEXT_FIELDS = ["title", "text"]
KEYWORD_FIELDS = [
    "source",
    "category",
    "subcategory",
    "type",
    "file",
    "language",
    "has_ai_callouts",
    "context",
    "tags",
]


def _load_adapter(source: str):
    if source not in ADAPTERS:
        raise ValueError(f"Unknown source: {source!r}. Available: {list(ADAPTERS)}")
    module_path, display_name = ADAPTERS[source]
    module = import_module(module_path)
    return module.ingest, display_name


async def run_ingest(source: str = "notion", output_path: str | None = None) -> int:
    """Run ingestion for the given source. Returns number of indexed documents.

    Args:
        source: Adapter name to run, or 'all' for every adapter.
        output_path: Override for the index file path.
                     Defaults to settings.vault_index_path.
    """
    sources = list(ADAPTERS) if source == "all" else [source]
    all_docs: list[dict] = []
    t_total = time.monotonic()

    for src in sources:
        ingest_fn, display_name = _load_adapter(src)
        logger.info("Running %s adapter...", display_name)
        t0 = time.monotonic()
        docs = ingest_fn()
        elapsed = time.monotonic() - t0
        logger.info("%s: %d documents (%.1fs)", display_name, len(docs), elapsed)
        all_docs.extend(docs)

    if not all_docs:
        logger.warning("No documents produced — index not written.")
        return 0

    index = minsearch.AppendableIndex(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS)
    index.fit(all_docs)

    dest = output_path or str(settings.vault_index_path)
    index.save(dest)

    try:
        await semantic.embed_documents(all_docs)
    except Exception:
        logger.exception("Semantic index build failed")

    total_elapsed = time.monotonic() - t_total
    logger.info(
        "Index saved to %s — %d documents indexed in %.1fs",
        dest,
        len(all_docs),
        total_elapsed,
    )
    return len(all_docs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest vault notes into a minsearch index"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="notion",
        choices=[*ADAPTERS, "all"],
        help="Data source to ingest (default: notion). 'all' runs every adapter.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for the serialized index. "
            "Defaults to VAULT_INDEX_PATH from .env / config.py."
        ),
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only ingest new/changed files (not yet implemented).",
    )
    args = parser.parse_args()

    if args.incremental:
        logger.warning("--incremental is not yet implemented. Running full ingestion.")

    logging.basicConfig(level=settings.log_level)
    asyncio.run(run_ingest(source=args.source, output_path=args.output))


if __name__ == "__main__":
    main()
