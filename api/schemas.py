from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, str] = {}
    expand: bool = False


class ChunkResult(BaseModel):
    title: str
    text_snippet: str
    source: str
    category: str
    file: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]


class IngestRequest(BaseModel):
    source: str = "notion"  # notion | gmail | chat_export | all


class IngestResponse(BaseModel):
    documents_indexed: int
    source: str
