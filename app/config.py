"""Settings, read from environment variables only (no secrets in code)."""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("app")

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_db_url(url: str) -> str:
    # Render/Heroku hand out postgres:// URLs; SQLAlchemy 2 wants an explicit driver.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


@dataclass
class Settings:
    app_env: str = "development"
    database_url: str = ""
    secret_key: str = ""
    admin_user: str = ""
    admin_password: str = ""
    base_url: str = "http://localhost:8000"
    upload_dir: Path = field(default_factory=lambda: ROOT / "data" / "uploads")
    storage_backend: str = "local"   # "local" (disk) | "db" (photo bytes in the database)
    max_upload_bytes: int = 8 * 1024 * 1024
    cookie_secure: bool = False
    anthropic_api_key: str = ""
    fake_llm: bool = False
    claude_model: str = "claude-sonnet-5-5"
    claude_fallbacks: bool = True
    interview_max_user_turns: int = 20
    interview_min_user_turns: int = 8

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key) and not self.fake_llm


def load_settings() -> Settings:
    env = os.environ.get("APP_ENV", "development").strip().lower()
    data_dir = Path(os.environ.get("DATA_DIR", str(ROOT / "data")))
    db_url = os.environ.get("DATABASE_URL", "").strip() or f"sqlite:///{data_dir / 'app.db'}"
    secret = os.environ.get("SECRET_KEY", "").strip()
    if not secret:
        if env == "production":
            raise RuntimeError("SECRET_KEY must be set in production")
        secret = secrets.token_urlsafe(48)
        log.warning("SECRET_KEY not set: using a random per-process key (sessions reset on restart)")
    s = Settings(
        app_env=env,
        database_url=_normalize_db_url(db_url),
        secret_key=secret,
        admin_user=os.environ.get("ADMIN_USER", "").strip(),
        admin_password=os.environ.get("ADMIN_PASSWORD", ""),
        base_url=os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/"),
        upload_dir=Path(os.environ.get("UPLOAD_DIR", str(data_dir / "uploads"))),
        storage_backend=os.environ.get("STORAGE", "local").strip().lower(),
        max_upload_bytes=int(os.environ.get("MAX_UPLOAD_MB", "8")) * 1024 * 1024,
        cookie_secure=_bool("COOKIE_SECURE", env == "production"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        fake_llm=_bool("FAKE_LLM", False),
        claude_model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5").strip(),
        claude_fallbacks=_bool("CLAUDE_FALLBACKS", True),
    )
    return s


settings = load_settings()
