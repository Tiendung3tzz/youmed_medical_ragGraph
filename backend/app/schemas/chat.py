from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    include_debug: bool = True


class ChatResponse(BaseModel):
    question: str
    answer: str
    cypher: str = ""
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None
    answer_error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    qdrant_hits: List[Dict[str, Any]] = Field(default_factory=list)


class GraphOnlyResponse(BaseModel):
    question: str
    cypher: str = ""
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    qdrant_hits: List[Dict[str, Any]] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    app: str
    env: str
