from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aquaguard.assembly import CameraRuntimeConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AQUAGUARD_", env_file=".env")

    env: str = "development"
    risk_threshold: float = Field(default=80, ge=0, le=100)
    confirmation_frames: int = Field(default=3, ge=1, le=300)
    alarm_cooldown_seconds: float = Field(default=10, ge=0)
    evidence_pre_seconds: float = Field(default=30, ge=0)
    evidence_post_seconds: float = Field(default=60, ge=0)
    database_url: str = "sqlite:///./aquaguard.db"
    evidence_directory: Path = Path("./data/evidence")
    cameras: tuple[CameraRuntimeConfig, ...] = ()
    evaluation_audit_capacity: int = Field(default=1000, ge=1)
    audit_database_path: Path | None = None
    event_database_path: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
