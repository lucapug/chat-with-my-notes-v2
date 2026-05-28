"""Golden set loader for RAG evaluation.

Migrated from master_run_eval.py (sprint1-hermes-backup).
Parses the ```golden-set``` fenced block from CONCEPT_DOC.md.
"""

import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_golden_set(concept_doc_path: str | Path) -> list[dict]:
    """Extract the golden-set JSON block from CONCEPT_DOC.md.

    Looks for a fenced code block tagged 'golden-set' and parses the
    JSON inside. Exits with error if the block is missing or invalid —
    this is intentional: a missing block means the doc needs updating.
    """
    with open(concept_doc_path, encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"```golden-set\s*\n(.*?)```", content, re.DOTALL)
    if not match:
        logger.error(
            "No ```golden-set``` block found in %s. "
            "Add a fenced JSON block with 'golden-set' tag.",
            concept_doc_path,
        )
        sys.exit(1)

    try:
        golden_set = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in golden-set block: %s", e)
        sys.exit(1)

    required_keys = {"id", "topic", "query"}
    for item in golden_set:
        missing = required_keys - set(item.keys())
        if missing:
            logger.error(
                "Golden-set entry %s missing keys: %s", item.get("id", "?"), missing
            )
            sys.exit(1)

    return golden_set


def load_ground_truth(gt_path: str | Path) -> dict[str, str | None]:
    """Load ground truth answers from golden-set-ground-truth.json.

    Returns a dict mapping question id → expected_answer (str or None).
    Logs a warning for entries where expected_answer is still null.
    """
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    gt_map: dict[str, str | None] = {}
    for item in data["golden_set"]:
        qid = item["id"]
        answer = item.get("expected_answer")
        if answer is None:
            logger.warning(
                "%s has no expected_answer — falling back to chunk-based judging.", qid
            )
        gt_map[qid] = answer

    return gt_map
