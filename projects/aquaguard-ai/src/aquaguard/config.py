from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AQUAGUARD_", env_file=".env")

    env: str = "development"
    risk_threshold: float = Field(default=80, ge=0, le=100)
    confirmation_frames: int = Field(default=3, ge=1, le=300)
    alarm_cooldown_seconds: float = Field(default=10, ge=0)
    database_url: str = "sqlite:///./aquaguard.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
