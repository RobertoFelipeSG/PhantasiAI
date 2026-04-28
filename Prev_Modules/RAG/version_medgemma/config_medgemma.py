import os
import torch
from pydantic_settings import BaseSettings
from typing import Optional, Literal


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "RAG Application"
    DEBUG: bool = False

    # Data Settings
    DATA_DIR: str = "data"
    DB_DIR: str = "db"

    # Model Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    LLM_MODEL: Literal[
        "google/medgemma-4b-it",
        "google/medgemma-4b-pt",
        "google/medgemma-27b-it",
        "google/medgemma-27b-text-it"
    ] = "google/medgemma-4b-it"
    LLM_PROVIDER: str = "huggingface"
    LLM_DEVICE: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    # LLM_TORCH_DTYPE: str = "bfloat16"
    # LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")  # Google API key

    # Vector Store Settings
    COLLECTION_NAME: str = "documents"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 1000
    CHROMA_DISTANCE_METRIC: str = "cosine"

    # Logging
    LOG_LEVEL: str = "INFO"
    VERBOSE: bool = True
    LOG_FORMAT: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    class Config:
        case_sensitive = True


settings = Settings()
