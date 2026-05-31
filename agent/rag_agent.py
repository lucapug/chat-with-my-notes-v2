import asyncio
import logging
from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
    "- rispondi in italiano, conciso."
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
)

agent: Agent[VaultDeps, str] = Agent(
    _model,
    deps_type=VaultDeps,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool
async def search_vault(ctx: RunContext[VaultDeps], query: str) -> str:
    """Cerca nelle note personali di Luca. Restituisce i chunk più rilevanti."""
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
        return "Nessun chunk trovato nel vault."
    parts = [
        f"[{r.get('title', '')}] "
        f"(category: {r.get('category', '')}, file: {r.get('file', '')})\n"
        f"{r.get('text', '')[:2500]}"
        for r in results
    ]
    return "\n---\n".join(parts)


async def ask(request: ChatRequest) -> ChatResponse:
    deps = VaultDeps(
        top_k=request.top_k,
        filters=request.filters,
        expand=request.expand,
    )
    result = await agent.run(request.query, deps=deps)
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
