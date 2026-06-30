from functools import lru_cache
import json

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "YouMed GraphRAG API"
    app_env: str = "local"
    debug: bool = False
    api_prefix: str = "/api"

    # CORS
    # Để string để tránh pydantic tự json.loads list[str] và lỗi.
    frontend_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173,"
        "http://localhost:3000,"
        "http://localhost"
    )

    # LLM
    llm_provider: str = "groq"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    groq_api_key: str = ""
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    enhanced_schema: bool = True

    # GraphRAG
    max_rows: int = 10
    graph_top_k: int = 10
    answer_max_rows: int = 5
    answer_max_text_chars: int = 900
    max_answer_rows: int = 5
    max_text_chars: int = 1200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def frontend_origins_list(self) -> list[str]:
        raw = (self.frontend_origins or "").strip()

        if not raw:
            return []

        if raw.startswith("["):
            try:
                value = json.loads(raw)
                if isinstance(value, list):
                    return [str(x).strip() for x in value if str(x).strip()]
            except json.JSONDecodeError:
                pass

        return [x.strip() for x in raw.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()