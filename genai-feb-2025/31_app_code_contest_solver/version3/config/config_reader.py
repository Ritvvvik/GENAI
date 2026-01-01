from typing import Tuple, Type
from pydantic import BaseModel
from pathlib import Path
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class LLMTomlSettings(BaseModel):
    name: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int


class LogfireTomlSettings(BaseModel):
    token: str


class EmbedderTomlSettings(BaseModel):
    model: str
    token: str
    normalize: bool


class RerankerTomlSettings(BaseModel):
    model: str
    token: str
    top_k: int


class FilePathsTomlSettings(BaseModel):
    src_dir: str
    kb_dir: str
    vector_db_dir: str
    eval_file: str
    results_file: str


class RetrieverTomlSettings(BaseModel):
    top_k: int


class TomlSettings(BaseSettings):
    model_config = SettingsConfigDict(toml_file=Path(__file__).parent / "config.toml")

    max_iter: int
    llm_generator: LLMTomlSettings
    logfire: LogfireTomlSettings
    embedder: EmbedderTomlSettings
    reranker: RerankerTomlSettings
    file_paths: FilePathsTomlSettings
    retriever: RetrieverTomlSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


settings = TomlSettings()

if __name__ == "__main__":
    print(settings)
