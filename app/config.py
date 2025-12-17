import os
# CHANGE: Import from pydantic_settings instead of pydantic
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Magazine Enhancer"
    GCS_BUCKET_ORIGINAL: str = os.getenv("GCS_BUCKET_ORIGINAL", "photo_enhance")
    GCS_BUCKET_ENHANCED: str = os.getenv("GCS_BUCKET_ENHANCED", "photo_enhance")
    
    class Config:
        env_file = ".env"

settings = Settings()