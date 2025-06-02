from langchain_ollama import OllamaEmbeddings
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.model = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL
        )
        logger.info(f"Initialized embedding service with model: {settings.EMBEDDING_MODEL}")

    def get_embeddings(self):
        return self.model
