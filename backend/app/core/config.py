from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://cocktail_admin:secret@db/cocktail_db"
    REDIS_URL: str = "redis://redis:6379"

    class Config:
        env_file = ".env"

settings = Settings()