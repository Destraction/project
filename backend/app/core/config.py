from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://cocktail_admin:secret@db/cocktail_db"
    REDIS_URL: str = "redis://redis:6379"
    ENABLE_OLLAMA: bool = True
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"

    class Config:
        env_file = ".env"

settings = Settings()