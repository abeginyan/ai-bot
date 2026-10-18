import os


class Settings:
    """Runtime configuration, all overridable via environment variables."""

    # Directory holding the universe_docs/*.txt knowledge base.
    knowledge_base_dir: str = os.getenv("KNOWLEDGE_BASE_DIR", "/knowledge_base")

    # Embedding model used for retrieval.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    # How many documents to retrieve by default.
    default_top_k: int = int(os.getenv("RAG_TOP_K", "4"))

    # OpenAI configuration used for answer generation.
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")  # optional, for proxies

    # Browser origins allowed to call this service directly (comma-separated).
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
        ).split(",")
        if origin.strip()
    ]


settings = Settings()
