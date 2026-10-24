"""Configuration settings for MessMate backend."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_backend_root = Path(__file__).resolve().parent.parent.parent
_env_file = str(_backend_root / ".env") if (_backend_root / ".env").exists() else ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    DATABASE_URL: str = "sqlite+aiosqlite:///./messmate.db"
    SYNC_DATABASE_URL: str = "sqlite:///./messmate.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "messmate-production-secure-hmac-sha256-key-v1-super-secret-512bit"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-lite-latest"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"

    @property
    def is_production(self) -> bool:
        """Return True if running in production."""
        return self.ENVIRONMENT.lower() in ("production", "prod")

    @property
    def is_test(self) -> bool:
        """Return True if running in test environment."""
        return self.ENVIRONMENT.lower() in ("test", "testing")

    model_config = SettingsConfigDict(
        env_file=_env_file,
        extra="ignore",
        case_sensitive=True,
    )


settings = Settings()
