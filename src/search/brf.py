"""Bilingual Retrieval Fusion (BRF) search engine.

Migrated from vault_search.py (sprint1-hermes-backup).
Paths via config.settings. HTTP via httpx (async).

BRF Protocol:
  Standard: IT query → EN translation → 2 minsearch runs → RRF fusion
  Expand:   3 IT variants + batch EN translation → 8 searches → RRF fusion
"""

import json
import logging
import re
from collections import Counter

import httpx
import minsearch

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_BOOST: dict[str, float] = {"title": 3.0, "text": 1.0}
RRF_K = 60

_index: minsearch.Index | None = None


def _load_index() -> minsearch.Index:
    global _index
    if _index is None:
        _index = minsearch.Index.load(str(settings.vault_index_path))
        logger.info("Loaded index from %s", settings.vault_index_path)
    return _index


def get_index_info() -> dict:
    idx = _load_index()
    docs = idx.docs if hasattr(idx, "docs") else []
    return {
        "chunk_count": len(docs),
        "sources": sorted({d.get("source", "?") for d in docs}),
        "by_category": dict(Counter(d.get("category", "?") for d in docs).most_common()),
        "by_language": dict(Counter(d.get("language", "?") for d in docs)),
        "text_fields": idx.text_fields,
        "keyword_fields": idx.keyword_fields,
    }


def _chunk_id(chunk: dict) -> str:
    """Stable dedup key: file + title + first 80 chars of text."""
    return f"{chunk.get('file', '')}|{chunk.get('title', '')}|{chunk.get('text', '')[:80]}"


def _rrf_fuse_many(result_lists: list[list[dict]], top_k: int, k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion over N ranked lists. Deduplicates by chunk_id."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            cid = _chunk_id(chunk)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = chunk
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked[:top_k]]


def _minsearch(query: str, top_k: int, filters: dict | None) -> list[dict]:
    idx = _load_index()
    kwargs: dict = {"num_results": top_k, "boost_dict": DEFAULT_BOOST}
    if filters:
        kwargs["filter_dict"] = filters
    return idx.search(query, **kwargs)


def _extract_str_list(msg: dict, n: int | None = None) -> list[str] | None:
    """Extract a JSON string array from an Ollama message.

    Handles gemma4:e4b thinking mode where content may be empty and
    the result is in the reasoning field instead.
    """
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning") or "").strip()
    source = content if content else reasoning
    if not source:
        return None

    # Strip markdown code fences
    source = re.sub(r"^```(?:json)?\s*\n?", "", source)
    source = re.sub(r"\n?```\s*$", "", source)

    match = re.search(r"\[.*?\]", source, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                result = [str(v) for v in items if isinstance(v, str)]
                return result[:n] if n else result
        except json.JSONDecodeError:
            pass

    # Fallback: extract quoted strings
    quoted = re.findall(r'"([^"]+)"', source)
    if quoted:
        return quoted[:n] if n else quoted

    return None


async def _translate_query(query_it: str) -> str | None:
    """Translate Italian query to English via Ollama judge model.

    Returns translated string, or None on failure (BRF falls back to IT-only).
    """
    payload = {
        "model": settings.ollama_judge_model,
        "messages": [{
            "role": "user",
            "content": (
                "Translate to English. Reply ONLY the translation, "
                "no explanation:\n\n" + query_it
            ),
        }],
        "max_tokens": 100,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.ollama_judge_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning") or "").strip()
            result = content if content else reasoning
            return result if result else None
    except Exception as e:
        logger.warning("BRF _translate_query failed: %s", e)
        return None


async def _translate_batch(queries: list[str]) -> list[str | None]:
    """Translate multiple Italian queries to English in one LLM call."""
    if not queries:
        return []

    numbered = "\n".join(f'{i + 1}. "{q}"' for i, q in enumerate(queries))
    payload = {
        "model": settings.ollama_judge_model,
        "messages": [{
            "role": "user",
            "content": (
                "Translate these Italian queries to English. "
                "Reply ONLY with a JSON array of strings, same order, "
                "no explanation:\n\n" + numbered
            ),
        }],
        "max_tokens": 2000,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.ollama_judge_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            extracted = _extract_str_list(msg, n=len(queries))
            if extracted and len(extracted) == len(queries):
                return extracted
            # Fallback: translate individually
            results = []
            for q in queries:
                results.append(await _translate_query(q))
            return results
    except Exception as e:
        logger.warning("BRF _translate_batch failed: %s", e)
        return [None] * len(queries)


async def _expand_query(query: str, n: int = 3) -> list[str]:
    """Generate n alternative Italian queries via LLM (query expansion)."""
    payload = {
        "model": settings.ollama_judge_model,
        "messages": [{
            "role": "user",
            "content": (
                f"Genera {n} query alternative in italiano per la seguente domanda. "
                "Usa sinonimi, termini tecnici correlati e riformulazioni che "
                "potrebbero trovare le stesse informazioni in una ricerca testuale.\n\n"
                "Rispondi SOLO con un array JSON di stringhe, nient'altro. "
                f"Esattamente {n} elementi.\n\n"
                f"Domanda: {query}"
            ),
        }],
        "max_tokens": 2000,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.ollama_judge_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            extracted = _extract_str_list(msg, n=n)
            return extracted if extracted else []
    except Exception as e:
        logger.warning("BRF _expand_query failed: %s", e)
        return []


async def search(
    query: str,
    top_k: int = 5,
    filters: dict[str, str] | None = None,
    expand: bool = False,
) -> list[dict]:
    """BRF search over the vault index.

    Standard mode (expand=False):
      1. IT query → minsearch → results_IT
      2. IT→EN translation → minsearch → results_EN
      3. RRF fusion → top_k results

    Expansion mode (expand=True):
      1. Generate 3 IT variants via LLM
      2. Batch-translate original + variants to EN
      3. 8 searches total (4 IT + 4 EN), 4:4 balanced fusion via RRF
    """
    if expand:
        it_queries = [query] + await _expand_query(query, n=3)
        en_translations = await _translate_batch(it_queries)
        en_queries = [t for t in en_translations if t]
        all_results = [_minsearch(q, top_k * 2, filters) for q in it_queries]
        all_results += [_minsearch(q, top_k * 2, filters) for q in en_queries]
        return _rrf_fuse_many(all_results, top_k)
    else:
        results_it = _minsearch(query, top_k * 2, filters)
        query_en = await _translate_query(query)
        if query_en:
            results_en = _minsearch(query_en, top_k * 2, filters)
            return _rrf_fuse_many([results_it, results_en], top_k)
        return results_it[:top_k]
