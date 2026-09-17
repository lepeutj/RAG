from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund policy?"])
    top_k: int | None = Field(None, ge=1, le=20)


class SourceChunk(BaseModel):
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[SourceChunk]


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_created: int


class StatsResponse(BaseModel):
    chunks_indexed: int
    embedding_provider: str
    llm_provider: str


class HealthResponse(BaseModel):
    status: str = "ok"
