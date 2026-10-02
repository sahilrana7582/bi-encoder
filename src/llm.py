from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from langchain_openai import ChatOpenAI

class Settings(BaseSettings):

    llm_api_key: SecretStr
    llm_model: str = "gpt-6-luna"
    llm_timeout_seconds: float = Field(default=30, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()


def create_llm_model(settings: Settings) -> ChatOpenAI:
    model = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds
    )
    return model