from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "RAG Application"
    DEBUG: bool = False

    # Data Settings
    DATA_DIR: str = "data"
    DB_DIR: str = "db"

    # Model Settings
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    LLM_MODEL: str = "deepseek-coder"

    # Vector Store Settings
    COLLECTION_NAME: str = "documents"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3

    # Logging
    LOG_LEVEL: str = "INFO"
    VERBOSE: bool = True
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    class Config:
        case_sensitive = True


settings = Settings()
