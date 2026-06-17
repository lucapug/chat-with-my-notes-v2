import asyncio
import logging
from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from api.schemas import ChatRequest, ChatResponse, ChunkResult
from config import settings
from src.search import brf, semantic
from src.search.fusion import rrf_fuse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sei l'assistente di Luca. Rispondi alla sua domanda "
    "usando ESCLUSIVAMENTE le informazioni presenti nei chunk "
    "forniti dallo strumento search_vault. "
    "Regole: "
    "- cita il testo rilevante tra virgolette; "
    "- indica sempre la fonte: [fonte: titolo, category, file]; "
    "- se l'informazione NON è nei chunk: "
    "  'Non ho trovato questa informazione nelle tue note.'; "
    "- NON usare conoscenza esterna; "
    "- rispondi in italiano, conciso. "
    "- CRITICAL: Quando riporti valori numerici, sii ESPlicito su cosa ogni numero si riferisce. "
    "Usa il formato 'X si riferisce a Y' o 'per Z: valore = X' per evitare confusione. "
    "Se il contesto menziona più valori numerici (es. tau_L=45, tau_Ab=20), riportali tutti separatamente con etichette chiare."
)


@dataclass
class VaultDeps:
    top_k: int
    filters: dict[str, str]
    expand: bool
    retrieved_chunks: list[dict] = field(default_factory=list)


_model = OpenAIChatModel(
    settings.ollama_generation_model,
    provider=OpenAIProvider(
        base_url=settings.ollama_generation_url,
        api_key="ollama",  # Ollama does not require a real key
    ),
    settings=OpenAIModelSettings(max_tokens=1024),
)

_generation_agent = Agent(
    _model,
    system_prompt=SYSTEM_PROMPT,
)

agent: Agent[VaultDeps, str] = Agent(
    _model,
    deps_type=VaultDeps,
    system_prompt=SYSTEM_PROMPT,
)

# NOTE: This production-facing path uses pydantic-ai and Ollama via
# OpenAIChatModel. There is a known mapping concern between Ollama's
# `num_predict` setting and OpenAI-style `max_tokens`/`max_completion_tokens`.
# The batch evaluation path in run_eval.py uses direct httpx calls instead,
# because eval offline does not need the agent overhead and the mapping issue
# should be validated separately when integrating with FastAPI.
async def generate_answer(query: str, context: str, usage_limits: UsageLimits | None = None) -> str:
    user_prompt = f"Domanda: {query}\n\nContesto:\n{context}"
    result = await _generation_agent.run(
        user_prompt,
        usage_limits=usage_limits,
    )
    return str(result.output).strip()


@agent.tool
async def search_vault(ctx: RunContext[VaultDeps], query: str) -> str:
    """Cerca nelle note personali di Luca. Restituisce i chunk più rilevanti."""
    logger.debug("search_vault: start query=%r top_k=%s filters=%s expand=%s", query, ctx.deps.top_k, ctx.deps.filters, ctx.deps.expand)
    top_k = ctx.deps.top_k
    filters = ctx.deps.filters
    expand = ctx.deps.expand

    brf_ok = settings.vault_index_path.exists()
    sem_ok = settings.semantic_index_path.exists()

    if not brf_ok and not sem_ok:
        logger.warning("search_vault: neither vault_index.pkl nor semantic_index.pkl found.")
        ctx.deps.retrieved_chunks = []
        return "Nessun indice disponibile. Esegui prima l'ingestione."

    if brf_ok and sem_ok:
        brf_task = brf.search(query=query, top_k=top_k * 2, filters=filters, expand=expand)
        sem_task = semantic.search(query=query, top_k=top_k * 2)
        raw = await asyncio.gather(brf_task, sem_task, return_exceptions=True)
        brf_results = raw[0] if not isinstance(raw[0], BaseException) else []
        sem_results = raw[1] if not isinstance(raw[1], BaseException) else []
        if isinstance(raw[0], BaseException):
            logger.warning("search_vault: BRF search failed: %s", raw[0])
        if isinstance(raw[1], BaseException):
            logger.warning("search_vault: semantic search failed: %s", raw[1])
        results = rrf_fuse(brf_results, sem_results, top_k=top_k)
    elif brf_ok:
        logger.warning("search_vault: semantic index not found — falling back to BRF only.")
        results = await brf.search(query=query, top_k=top_k, filters=filters, expand=expand)
    else:
        logger.warning("search_vault: BRF index not found — falling back to semantic only.")
        results = await semantic.search(query=query, top_k=top_k)

    ctx.deps.retrieved_chunks = results
    if not results:
        logger.debug("search_vault: end (no results)")
        return "Nessun chunk trovato nel vault."

    MAX_TOOL_CHUNK_TEXT = 300
    parts = []
    for r in results:
        text = r.get("text", "") or ""
        snippet = text[:MAX_TOOL_CHUNK_TEXT]
        if len(text) > MAX_TOOL_CHUNK_TEXT:
            snippet += "..."
        parts.append(
            f"[{r.get('title', '')}] "
            f"(category: {r.get('category', '')}, file: {r.get('file', '')})\n"
            f"{snippet}"
        )
    logger.debug("search_vault: end results=%s", len(results))
    return "\n---\n".join(parts)


async def ask(request: ChatRequest, usage_limits: UsageLimits | None = None) -> ChatResponse:
    deps = VaultDeps(
        top_k=request.top_k,
        filters=request.filters,
        expand=request.expand,
    )
    logger.warning("ask: start query=%r", request.query)
    logger.debug("ask: start query=%r", request.query)
    result = await agent.run(request.query, deps=deps, usage_limits=usage_limits)
    logger.warning("ask: agent.run completed")
    logger.debug("ask: agent.run completed")
    logger.warning("ask: assembling response sources=%s", len(deps.retrieved_chunks))
    logger.debug("ask: assembling response sources=%s", len(deps.retrieved_chunks))
    sources = [
        ChunkResult(
            title=r.get("title", ""),
            text_snippet=r.get("text", "")[:300],
            source=r.get("source", ""),
            category=r.get("category", ""),
            file=r.get("file", ""),
        )
        for r in deps.retrieved_chunks
    ]
    return ChatResponse(answer=result.output, sources=sources)
