from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BACKEND_BASE_URL: str = "http://3.34.47.36:8080"
    MLOPS_CALLBACK_TOKEN: str = "T7vu0Sb8TkCyLr1Or8+OC2mHgm94/kis1qdvLoKkDAGkAsTchmMrx1lNkwkmxY9x"
    MLOPS_PULL_TOKEN: str = "G3RMjLElw33n1/24I0teC3d3KpvdRuaLEMebPB5ZNaO9yaDZqslVhNSQe/hT2nlA"
    # Backend -> AI webhook auth token. If empty, fallback to callback token.
    MLOPS_WEBHOOK_TOKEN: str = ""
    
    # Timeout settings
    HTTP_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()