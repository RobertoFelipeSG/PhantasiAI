from services.embedders.base import BaseEmbedder
import logging
from langchain_community.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)


class OllamaEmbedder(BaseEmbedder):
    """
    Embedder using Ollama.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._embedder = OllamaEmbeddings(model=model_name)
        logger.info(f"Initialized OllamaEmbedder with model: {model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        return self._embedder.embed_query(text)
