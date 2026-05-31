"""Runtime configuration — loaded once from environment / .env at import time."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    provider: str
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str
    max_iterations: int
    sandbox_timeout: int
    sandbox_memory_mb: int
    log_level: str
    runs_dir: Path


def load_config() -> Config:
    return Config(
        provider=os.environ.get("PROVIDER", "groq"),
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "10")),
        sandbox_timeout=int(os.environ.get("SANDBOX_TIMEOUT", "30")),
        sandbox_memory_mb=int(os.environ.get("SANDBOX_MEMORY_MB", "1024")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        runs_dir=Path(os.environ.get("RUNS_DIR", "runs")),
    )
