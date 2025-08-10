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
    EMBEDDING_MODEL: Literal[
        "mxbai-embed-large",
        "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16",
        "jasper_en_vision_language_v1",
        "Losspost/stella_en_1.5b_v5",
        "nvidia/NV-Embed-v2",
        "rjmalagon/gte-qwen2-7b-instruct:f16"
    ] = "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16"
    EMBEDDING_PROVIDER: str = "auto"
    EMBEDDING_DEVICE: Optional[str] = None
    EMBEDDING_BATCH_SIZE: int = 32

    LLM_MODEL: str = "llama3:8b"

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
