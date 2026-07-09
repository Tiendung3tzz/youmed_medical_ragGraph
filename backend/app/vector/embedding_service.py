from __future__ import annotations

from functools import cached_property
from typing import Iterable, List

from app.core.config import Settings


class EmbeddingService:
    """Small wrapper around FastEmbed.

    The model is loaded lazily so regular Cypher-only chat does not pay the
    embedding model startup cost until hybrid retrieval is used.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.settings.embedding_model_name)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        values = [str(text or "") for text in texts]
        if not values:
            return []
        return [vector.tolist() for vector in self._model.embed(values)]

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []
