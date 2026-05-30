"""Semantic search via Ollama /embeddings endpoint + minsearch.VectorSearch.

Provides:
  embed_documents(chunks)  — generate embeddings and build a VectorSearch index
                             persisted via minsearch's native save/load.
  search(query, top_k)     — cosine similarity search; return format identical
                             to brf.search() (flat chunk dicts, no score key).

Index format: minsearch.VectorSearch pickle (via VectorSearch.save/load).
"""

import logging

import httpx
import minsearch
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

_vector_index: minsearch.VectorSearch | None = None


# ── Embedding ─────────────────────────────────────────────────────────────────

async def embed(texts: list[str]) -> list[list[float]]:
    """Call Ollama /embeddings for a list of texts.

    Uses settings.ollama_embed_url and settings.embed_model.
    Returns a list of float vectors (one per input text).
    Raises httpx.HTTPError on network or API failure.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.ollama_embed_url}/embeddings",
            json={"model": settings.embed_model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()

    # Ollama returns {"embeddings": [[...], ...]} for batch input
    return data["embeddings"]


# ── Index build ───────────────────────────────────────────────────────────────

async def embed_documents(chunks: list[dict]) -> None:
    """Generate embeddings for chunk dicts and save the VectorSearch index.

    Each chunk must have a "text" field. All other fields are preserved and
    returned verbatim by search().

    Saves to settings.semantic_index_path via minsearch.VectorSearch.save().
    """
    global _vector_index

    if not chunks:
        logger.warning("embed_documents called with empty chunk list — nothing saved.")
        return

    texts = [c.get("text", "") for c in chunks]
    logger.info("Generating embeddings for %d chunks via %s...", len(texts), settings.embed_model)

    vectors = await embed(texts)
    embeddings = np.array(vectors, dtype=np.float32)

    vs = minsearch.VectorSearch()
    vs.fit(embeddings, chunks)

    dest = settings.semantic_index_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    vs.save(dest)
    _vector_index = vs

    logger.info(
        "Semantic index saved to %s (%d chunks, embedding dim %d)",
        dest,
        len(chunks),
        embeddings.shape[1] if embeddings.ndim == 2 else -1,
    )


async def append_documents(chunks: list[dict]) -> int:
    """Embed new chunks and append them to the existing VectorSearch index.

    Loads the index from disk if not already in memory, calls append_batch()
    to extend it in-place, then re-saves to settings.semantic_index_path.

    Returns the number of appended chunks.
    Raises FileNotFoundError if the index doesn't exist yet (run embed_documents first).
    """
    if not chunks:
        return 0
    vs = _load_index()
    texts = [c.get("text", "") for c in chunks]
    logger.info(
        "Generating embeddings for %d new chunks via %s...", len(texts), settings.embed_model
    )
    vectors = await embed(texts)
    new_embeddings = np.array(vectors, dtype=np.float32)
    vs.append_batch(new_embeddings, chunks)
    dest = settings.semantic_index_path
    vs.save(dest)
    logger.info(
        "Appended %d chunks to semantic index (%s)", len(chunks), dest
    )
    return len(chunks)


# ── Search ────────────────────────────────────────────────────────────────────

def _load_index() -> minsearch.VectorSearch:
    global _vector_index
    if _vector_index is None:
        path = settings.semantic_index_path
        if not path.exists():
            raise FileNotFoundError(
                f"Semantic index not found at {path}. "
                "Run embed_documents() to build it first."
            )
        _vector_index = minsearch.VectorSearch.load(path)
        logger.info("Loaded semantic index from %s", path)
    return _vector_index


async def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the semantic index by cosine similarity.

    Returns up to top_k chunk dicts with the same structure as brf.search().
    """
    vs = _load_index()
    query_vec_list = await embed([query])
    query_vec = np.array(query_vec_list[0], dtype=np.float32)
    return vs.search(query_vec, num_results=top_k)

    chunks: list[dict] = index["chunks"]
    embeddings: np.ndarray = index["embeddings"]

    query_vec_list = await embed([query])
    query_vec = np.array(query_vec_list[0], dtype=np.float32)

    scores = _cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        chunk = dict(chunks[i])
        chunk["score"] = float(scores[i])
        results.append(chunk)

    return results
