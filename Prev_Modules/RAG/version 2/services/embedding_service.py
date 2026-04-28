import logging
from core.config import settings
from typing import Optional
from services.embedders.base import BaseEmbedder
from services.embedders.ollama import OllamaEmbedder
from services.embedders.sentence_transformer import SentenceTransformerEmbedder
from services.embedders.nv_embedder import NVEmbedder

logger = logging.getLogger(__name__)

MODEL_PROVIDER_MAP = {
    "mxbai-embed-large": "ollama",
    "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16": "ollama",
    "jasper_en_vision_language_v1": "ollama",
    "Losspost/stella_en_1.5b_v5": "ollama",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence_transformers",
    "nvidia/NV-Embed-v2": "sentence_transformers",
    "rjmalagon/gte-qwen2-7b-instruct:f16": "ollama"
}


class EmbeddingService(BaseEmbedder):
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        cfg_model = model_name or settings.EMBEDDING_MODEL
        cfg_provider = (provider or settings.EMBEDDING_PROVIDER or "auto").lower()
        cfg_device = device if device is not None else settings.EMBEDDING_DEVICE
        cfg_batch_size = (
            batch_size if batch_size is not None else settings.EMBEDDING_BATCH_SIZE
        )

        self.model_name = cfg_model
        self.provider = cfg_provider

        embedding_model = self._build_embedder(
            cfg_model, cfg_provider, cfg_device, cfg_batch_size
        )
        self._embedding_model: BaseEmbedder = embedding_model

    def _build_embedder(
        self, model_name: str, provider: str, device: Optional[str], batch_size: int
    ) -> BaseEmbedder:
        if provider in ("ollama", "sentence_transformers"):
            return self._load_embedder(provider, model_name, device, batch_size)

        hinted = MODEL_PROVIDER_MAP.get(model_name)
        if hinted:
            try:
                return self._load_embedder(hinted, model_name, device, batch_size)
            except Exception as e:
                logger.warning(
                    f"Auto load via hinted provider '{hinted}' failed for {model_name}: {e}. "
                    "Falling back to alternate providers."
                )

    def _load_embedder(
        self, provider: str, model_name: str, device: Optional[str], batch_size: int
    ) -> BaseEmbedder:
        if provider == "ollama":
            return OllamaEmbedder(model_name)
        elif provider == "sentence_transformers":
            return SentenceTransformerEmbedder(
                model_name, device=device, batch_size=batch_size
            )
        else:
            raise ValueError(f"Unsupported embedder provider: {provider}")

    def get_embeddings(self) -> BaseEmbedder:
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self._embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embedding_model.embed_query(text)
