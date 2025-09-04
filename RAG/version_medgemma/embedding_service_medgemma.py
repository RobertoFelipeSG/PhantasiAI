import torch
import logging
from core.config_new import settings
from typing import Optional
from sentence_transformers import SentenceTransformer
from services.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

class EmbeddingService(BaseEmbedder):
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        cfg_model = model_name or settings.EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
        cfg_provider = (provider or settings.EMBEDDING_PROVIDER or "local").lower()
        cfg_device = device if device is not None else settings.EMBEDDING_DEVICE
        cfg_batch_size = (
            batch_size if batch_size is not None else settings.EMBEDDING_BATCH_SIZE
        )

        self.model_name = cfg_model
        self.provider = cfg_provider
        self.device = cfg_device
        self.batch_size = cfg_batch_size

        self._embedding_model = SentenceTransformer(self.model_name, device=self.device) 

        logger.info(f"Initialized EmbeddingService with model {self.model_name} (provider={self.provider})")

    def get_embeddings(self) -> BaseEmbedder:
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self._embedding_model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embedding_model.encode(
            [text], batch_size=1, convert_to_numpy=True
        )[0].tolist()
