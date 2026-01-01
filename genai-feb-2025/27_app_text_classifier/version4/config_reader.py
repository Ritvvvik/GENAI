from typing import Tuple, Type
from pydantic import BaseModel
from pathlib import Path
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class EmbedderTomlSettings(BaseModel):
    model: str
    token: str
    normalize: bool


class FilePathsTomlSettings(BaseModel):
    src_dir: str
    train_file: str
    embeddings_dir: str
    eval_file: str
    results_dir: str
    eval_file_with_response: str
    model_dir: str
    results_report_file: str
    results_matrix_file: str


class TomlSettings(BaseSettings):
    model_config = SettingsConfigDict(toml_file=Path(__file__).parent / "config.toml")

    file_paths: FilePathsTomlSettings
    embedder: EmbedderTomlSettings

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
