from __future__ import annotations

import json

from app.core.config import get_settings
from app.vector.embedding_service import EmbeddingService
from app.vector.qdrant_store import QdrantSectionStore


def main() -> None:
    settings = get_settings()
    qdrant_store = QdrantSectionStore(settings)
    embedding_service = EmbeddingService(settings)
    vector = embedding_service.embed_query("DNA là gì?")
    hits = qdrant_store.search_sections(vector, top_k=5)
    report = {
        "collection": settings.qdrant_collection,
        "count": qdrant_store.count(),
        "sample_hits": hits,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
