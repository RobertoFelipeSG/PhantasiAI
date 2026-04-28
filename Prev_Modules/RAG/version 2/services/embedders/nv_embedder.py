# services/embedders/nv_embed.py
from sentence_transformers import SentenceTransformer
import torch
from typing import Optional, List
from services.embedders.base import BaseEmbedder

NV_QUERY_PREFIX = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
NV_PASSAGE_PREFIX = ""  # none for passages

def _add_eos(texts: List[str], eos_token: str) -> List[str]:
    return [t + eos_token for t in texts]

class NVEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"
        self.eos = self.model.tokenizer.eos_token

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        texts = _add_eos(texts, self.eos)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        return emb.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        q = _add_eos([text], self.eos)
        emb = self.model.encode(
            q,
            batch_size=1,
            prompt=NV_QUERY_PREFIX,
            normalize_embeddings=True,
        )[0]
        return emb.tolist()
