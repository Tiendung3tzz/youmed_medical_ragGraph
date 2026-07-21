from __future__ import annotations

import logging
from functools import cached_property
from typing import Iterable, List, Sequence

from app.core.config import Settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Optional cross-encoder reranker for Qdrant candidates.

    This wrapper intentionally fails soft. If the package/model is not available,
    retrieval still works with Qdrant + graph-aware bonuses.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def _model(self):
        if not self.settings.reranker_enabled:
            return None

        try:
            import torch
            from sentence_transformers.cross_encoder import CrossEncoder
        except Exception as exc:  # pragma: no cover - depends on optional runtime package
            logger.warning("Reranker disabled because sentence-transformers/torch is unavailable: %r", exc)
            return None

        device = self.settings.reranker_device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading reranker model %s on %s", self.settings.reranker_model_name, device)
        return CrossEncoder(
            self.settings.reranker_model_name,
            device=device,
            max_length=self.settings.reranker_max_length,
        )

    def predict(self, question: str, candidate_texts: Sequence[str]) -> List[float]:
        if not candidate_texts:
            return []

        model = self._model
        if model is None:
            return [0.0 for _ in candidate_texts]

        pairs = [[question, text] for text in candidate_texts]
        try:
            scores = model.predict(
                pairs,
                batch_size=self.settings.reranker_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [float(x) for x in scores]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Reranker predict failed; using zero scores: %r", exc)
            return [0.0 for _ in candidate_texts]
