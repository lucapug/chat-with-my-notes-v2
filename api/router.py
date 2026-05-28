import asyncio
import logging

from fastapi import APIRouter, HTTPException

from agent.rag_agent import ask
from api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from src.ingest.orchestrator import run_ingest
from src.search.brf import get_index_info

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await ask(request)
    except Exception as e:
        logger.error("chat error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    try:
        count = await asyncio.to_thread(run_ingest, source=request.source)
        return IngestResponse(documents_indexed=count, source=request.source)
    except Exception as e:
        logger.error("ingest error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/index/info")
async def index_info() -> dict:
    return get_index_info()
