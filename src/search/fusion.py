"""Reciprocal Rank Fusion of BRF (lexical) + Semantic (vector) results.

Provides a single public function:
  rrf_fuse(brf_results, semantic_results, top_k, k=60) -> list[dict]

Algorithm: RRF score = Σ 1/(k + rank) over each ranked list.
Deduplication key: canonical chunk_id field (fallback to file|title|text[:80] for backward compatibility).

Reference implementation: sprint1-hermes-backup vault_search._rrf_fuse_many.
"""

RRF_K = 60


def _chunk_id(chunk: dict) -> str:
    """Use canonical chunk_id for fusion deduplication, fallback to legacy key."""
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{chunk.get('file', '')}|{chunk.get('title', '')}|{chunk.get('text', '')[:80]}"


def rrf_fuse(
    brf_results: list[dict],
    semantic_results: list[dict],
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:
    """Fuse a BRF ranked list and a semantic ranked list via Reciprocal Rank Fusion.

    Each list contributes 1/(k + rank) to the score of every chunk it contains.
    Chunks appearing in both lists accumulate scores from both.
    Deduplication is by canonical chunk_id when available.

    Args:
        brf_results:      Ranked list from brf.search() — flat chunk dicts.
        semantic_results: Ranked list from semantic.search() — flat chunk dicts.
        top_k:            Maximum number of results to return.
        k:                RRF constant (default 60, standard literature value).

    Returns:
        Up to top_k chunk dicts ordered by descending RRF score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for result_list in (brf_results, semantic_results):
        for rank, chunk in enumerate(result_list, start=1):
            cid = _chunk_id(chunk)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked[:top_k]]
