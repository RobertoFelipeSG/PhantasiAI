from services.embedders.base import BaseEmbedder
from sentence_transformers import SentenceTransformer
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder using Sentence Transformers.
    """

    NV_QUERY_PREFIX = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "

    def __init__(
        self, model_name: str, device: Optional[str] = None, batch_size: int = 32
    ):
        self.model_name = model_name
        logger.info(
            f"Loading SentenceTransformer model: {model_name} (device={device})"
        )
        self._model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        self.batch_size = batch_size

        # Detect if this is the NVIDIA model
        self.is_nv = model_name.lower() == "nvidia/nv-embed-v2"
        if self.is_nv:
            self._model.max_seq_length = 32768
            self._model.tokenizer.padding_side = "right"
            self.eos = self._model.tokenizer.eos_token

    def _add_eos(self, texts: list[str]) -> list[str]:
        return [t + self.eos for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.is_nv:
            texts = self._add_eos(texts)
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
            ).tolist()
        else:
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
            ).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        if self.is_nv:
            q = self._add_eos([text])
            return self._model.encode(
                q,
                batch_size=1,
                prompt=self.NV_QUERY_PREFIX,
                normalize_embeddings=True,
            )[0].tolist()
        else:
            return self._model.encode(text).tolist()
