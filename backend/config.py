from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    scrape_timeout: int = 15

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
