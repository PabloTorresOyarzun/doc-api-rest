from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Configuration
    API_TITLE: str = "Customs Document Processor API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    
    # SGD Configuration
    SGD_BASE_URL: str = "https://backend.juanleon.cl"
    SGD_BEARER_TOKEN: str
    
    # Azure Document Intelligence Configuration
    AZURE_ENDPOINT: str
    AZURE_KEY: str
    AZURE_LOCAL_ENDPOINT: str = "http://azure-di-custom:5000"
    
    # Processing Configuration
    MAX_WORKERS: int = 12
    OCR_DPI: int = 300
    HEADER_PERCENTAGE: float = 0.30
    
    # Directories
    DATA_INPUT_DIR: str = "./data/input"
    DATA_OUTPUT_DIR: str = "./data/output"
    TRAINING_DATA_DIR: str = "./data/training_data"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()