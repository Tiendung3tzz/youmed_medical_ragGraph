from functools import lru_cache

from app.core.config import get_settings
from app.services.graph_rag_service import YouMedGraphRAGService


@lru_cache
def get_rag_service() -> YouMedGraphRAGService:
    return YouMedGraphRAGService(get_settings())
