import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic_ai.usage import UsageLimits

from agent.rag_agent import ask
from api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from src.ingest.orchestrator import run_ingest
from src.search.brf import get_index_info

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        usage_limits = UsageLimits(request_limit=5, tool_calls_limit=3)
        return await ask(request, usage_limits=usage_limits)
    except Exception as e:
        logger.error("chat error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    try:
        count = await run_ingest(source=request.source)
        return IngestResponse(documents_indexed=count, source=request.source)
    except Exception as e:
        logger.error("ingest error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/index/info")
async def index_info() -> dict:
    return get_index_info()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        chat_req = ChatRequest(query=request.question)
        usage_limits = UsageLimits(request_limit=5, tool_calls_limit=3)
        chat_resp = await ask(chat_req, usage_limits=usage_limits)
        sources = list(dict.fromkeys(
            s.title for s in chat_resp.sources if s.title
        ))
        return QueryResponse(answer=chat_resp.answer, sources=sources)
    except Exception as e:
        logger.error("query error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
