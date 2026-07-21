from fastapi import APIRouter, Depends

from app.schemas.chat import ChatRequest, ChatResponse, GraphOnlyResponse
from app.services.dependencies import get_rag_service
from app.services.graph_rag_service import YouMedGraphRAGService

router = APIRouter(prefix="/chat", tags=["chat"])

def _hide_debug(result: dict) -> dict:
    result["cypher"] = ""
    result["rows"] = []
    result["qdrant_hits"] = []
    result["candidate_section_ids"] = []
    result["section_ids"] = []
    result["debug"] = {}
    return result

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, service: YouMedGraphRAGService = Depends(get_rag_service)) -> ChatResponse:
    result = service.ask_graph_with_answer(req.message)
    if not req.include_debug:
        result["cypher"] = ""
        result["rows"] = []
    return ChatResponse(**result)


@router.post("/graph", response_model=GraphOnlyResponse)
def graph_only(req: ChatRequest, service: YouMedGraphRAGService = Depends(get_rag_service)) -> GraphOnlyResponse:
    result = service.ask_graph(req.message)
    if not req.include_debug:
        result["cypher"] = ""
        result["rows"] = []
    return GraphOnlyResponse(**result)

@router.post("/hybrid", response_model=ChatResponse)
def hybrid_chat(req: ChatRequest, service: YouMedGraphRAGService = Depends(get_rag_service)) -> ChatResponse:
    result = service.ask_hybrid(req.message)
    if not req.include_debug:
        result = _hide_debug(result)
    return ChatResponse(**result)