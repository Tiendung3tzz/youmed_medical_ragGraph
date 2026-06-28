from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "YouMed GraphRAG Chat"
    app_env: str = "dev"
    debug: bool = True

    api_prefix: str = "/api"
    frontend_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    llm_provider: str = "groq"  # groq | google
    groq_api_key: str | None = None
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    google_api_key: str | None = None
    google_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    graph_top_k: int = 10
    answer_max_rows: int = 5
    answer_max_text_chars: int = 900
    enhanced_schema: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("frontend_origins", mode="before")
    @classmethod
    def parse_origins(cls, value):
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
