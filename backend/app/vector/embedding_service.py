from __future__ import annotations

import logging
from functools import cached_property
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding service dùng sentence-transformers.

    Dùng cho BAAI/bge-m3 để tạo vector 1024 dim,
    khớp với Qdrant collection hiện tại.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def _model(self) -> SentenceTransformer:
        logger.info(
            "Loading embedding model with sentence-transformers | model=%s",
            self.settings.embedding_model_name,
        )

        return SentenceTransformer(self.settings.embedding_model_name)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        values = [str(text or "") for text in texts]

        if not values:
            return []

        vectors = self._model.encode(
            values,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [
            np.asarray(vector, dtype=np.float32).tolist()
            for vector in vectors
        ]

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        vector = vectors[0] if vectors else []

        logger.info(
            "Embedding query generated | model=%s | dim=%s",
            self.settings.embedding_model_name,
            len(vector),
        )

        return vector