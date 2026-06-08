from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
    )

    # LLM — Generation
    ollama_generation_url: str = "http://localhost:11434/v1"
    ollama_generation_model: str = "llama3"

    # LLM — Judge (BRF translation + query expansion)
    ollama_judge_url: str = "http://localhost:11434/v1"
    ollama_judge_model: str = "llama3"

    # Index
    vault_index_path: Path = Path("data/vault_index.pkl")

    # Embedding
    ollama_embed_url: str = "http://localhost:11434/v1"
    embed_model: str = "nomic-embed-text"
    semantic_index_path: Path = Path("data/semantic_index.pkl")

    # Data sources
    vault_notion_dir: Path = Path("data/notion")
    vault_pages_map: Path = Path("data/notion_pages_map.json")

    # Fallback cloud — Z.AI
    zai_api_key: Optional[str] = None
    zai_base_url: str = "https://api.z.ai/api/paas/v4/"
    zai_model: str = "glm-4.7"

    # API
    app_port: int = 8000
    log_level: str = "INFO"


settings = Settings()
