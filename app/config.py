import os

class Settings:
    PROJECT_NAME: str = "Magazine-Grade Image Enhancer"
    
    # Storage Settings
    # In production, you set these via 'export GCS_BUCKET_ORIGINAL=real-bucket-name'
    # For now, we use defaults.
    GCS_BUCKET_ORIGINAL = os.getenv("GCS_BUCKET_ORIGINAL", "photo_enhance")
    GCS_BUCKET_ENHANCED = os.getenv("GCS_BUCKET_ENHANCED", "photo_enhance")

    # Project ID from the URL (o3o-aimodel)
    GCP_PROJECT_ID = "o3o-aimodel"
    
    # Path to Google Cloud Credentials file (JSON)
    # If this file is missing, our gcs.py service simply mocks the upload (safe for dev)
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

    # Image constraints
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    MAX_IMAGE_SIZE_MB = 10

settings = Settings()
