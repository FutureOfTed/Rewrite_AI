from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BACKEND_BASE_URL: str = "http://localhost:8080"
    MLOPS_CALLBACK_TOKEN: str = "default_token"
    
    # Timeout settings
    HTTP_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
