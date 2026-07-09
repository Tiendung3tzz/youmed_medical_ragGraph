from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import Settings


class QdrantSectionStore:
    """Qdrant collection for Section-level semantic retrieval.

    Payload contains stable Neo4j IDs. Runtime flow:
    search text -> Qdrant returns section_id -> Neo4j enriches by section_id.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout,
        )
        self.collection_name = settings.qdrant_collection

    @staticmethod
    def point_id(section_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"youmed-section:{section_id}"))

    def ensure_collection(self, vector_size: int, reset: bool = False) -> None:
        if reset:
            self.client.delete_collection(self.collection_name, timeout=self.settings.qdrant_timeout)
        collections = self.client.get_collections().collections
        exists = any(item.name == self.collection_name for item in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def upsert_sections(self, rows: Iterable[Dict[str, Any]], vectors: List[List[float]]) -> int:
        points: List[models.PointStruct] = []
        for row, vector in zip(rows, vectors):
            section_id = str(row.get("section_id") or "").strip()
            if not section_id or not vector:
                continue
            points.append(
                models.PointStruct(
                    id=self.point_id(section_id),
                    vector=vector,
                    payload={
                        "section_id": section_id,
                        "heading": row.get("heading"),
                        "text": row.get("text"),
                        "concepts": row.get("concepts") or [],
                        "articles": row.get("articles") or [],
                        "relation_types": row.get("relation_types") or [],
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search_sections(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not query_vector:
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold if score_threshold and score_threshold > 0 else None,
        )

        hits = response.points if hasattr(response, "points") else response
        output: List[Dict[str, Any]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            output.append(
                {
                    "point_id": str(hit.id),
                    "score": float(hit.score),
                    "section_id": payload.get("section_id"),
                    "heading": payload.get("heading"),
                    "concepts": payload.get("concepts") or [],
                    "relation_types": payload.get("relation_types") or [],
                    "articles": payload.get("articles") or [],
                }
            )
        return output

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection_name, exact=True)
        return int(result.count)
