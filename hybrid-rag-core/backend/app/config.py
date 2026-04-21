"""Application configuration — loaded from .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    EMBED_DIMS: int = 1024
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RERANK_TOP_K: int = 5
    RETRIEVE_TOP_K: int = 20
    MEMORY_WINDOW: int = 6
    MEMORY_SUMMARIZE_THRESHOLD: int = 12

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
