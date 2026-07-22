from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Iterable, List

import numpy as np

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding wrapper.

    - FastEmbed: dùng cho các model được FastEmbed hỗ trợ.
    - sentence-transformers: fallback cho BAAI/bge-m3 hoặc model FastEmbed không hỗ trợ.

    Lý do:
    Qdrant collection hiện tại của bạn đang là vector 1024 dim.
    BAAI/bge-m3 trả 1024 dim nhưng FastEmbed trong container không hỗ trợ model này.
    Vì vậy phải load bge-m3 bằng sentence-transformers.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def _backend_and_model(self) -> tuple[str, Any]:
        model_name = self.settings.embedding_model_name

        if model_name == "BAAI/bge-m3" or "bge-m3" in model_name.lower():
            return self._load_sentence_transformer(model_name)

        try:
            from fastembed import TextEmbedding

            supported_models = TextEmbedding.list_supported_models()
            supported_names = set()

            for item in supported_models:
                if isinstance(item, dict):
                    name = (
                        item.get("model")
                        or item.get("model_name")
                        or item.get("name")
                    )
                    if name:
                        supported_names.add(name)

            if model_name in supported_names:
                logger.info(
                    "Loading embedding model with FastEmbed | model=%s",
                    model_name,
                )
                return "fastembed", TextEmbedding(model_name=model_name)

            logger.warning(
                "Model is not supported by FastEmbed. "
                "Fallback to sentence-transformers | model=%s",
                model_name,
            )

        except Exception as exc:
            logger.warning(
                "FastEmbed failed. Fallback to sentence-transformers | error=%r",
                exc,
            )

        return self._load_sentence_transformer(model_name)

    @staticmethod
    def _load_sentence_transformer(model_name: str) -> tuple[str, Any]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing sentence-transformers. "
                "Add `sentence-transformers` and `torch` to requirements.txt."
            ) from exc

        logger.info(
            "Loading embedding model with sentence-transformers | model=%s",
            model_name,
        )

        model = SentenceTransformer(model_name)
        return "sentence_transformers", model

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        values = [str(text or "") for text in texts]

        if not values:
            return []

        backend, model = self._backend_and_model

        if backend == "fastembed":
            vectors = list(model.embed(values))
            return [
                np.asarray(vector, dtype=np.float32).tolist()
                for vector in vectors
            ]

        if backend == "sentence_transformers":
            vectors = model.encode(
                values,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [
                np.asarray(vector, dtype=np.float32).tolist()
                for vector in vectors
            ]

        raise RuntimeError(f"Unsupported embedding backend: {backend}")

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        vector = vectors[0] if vectors else []

        logger.info(
            "Embedding query generated | model=%s | dim=%s",
            self.settings.embedding_model_name,
            len(vector),
        )

        return vector