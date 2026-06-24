from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    pinecone_api_key: str | None
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    embedding_model: str
    embedding_dimension: int
    reranker_model: str
    gemini_model: str
    max_retries: int
    upload_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "rag-index"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            embedding_dimension=_int_from_env("EMBEDDING_DIMENSION", 768),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            max_retries=_int_from_env("MAX_RETRIES", 2),
            upload_dir=Path(os.getenv("UPLOAD_DIR", "/tmp/corrective-rag/uploads")),
        )

    @property
    def missing_env(self) -> list[str]:
        missing = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        return missing

    @property
    def configured(self) -> bool:
        return not self.missing_env


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
