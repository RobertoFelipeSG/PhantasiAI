from typing import List, Optional
from services.embedders.base import BaseEmbedder
import google.generativeai as genai

class GoogleEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, api_key: str, device: Optional[str] = None, batch_size: int = 32):
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size

        genai.configure(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = genai.embeddings.create(
                model=self.model_name,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = genai.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding