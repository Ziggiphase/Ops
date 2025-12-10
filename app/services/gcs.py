import os
from google.cloud import storage
from datetime import datetime
import uuid

# If we are in local dev/codespace, we might Mock this if no creds are present
class GCSService:
    def __init__(self):
        # We try to connect. If it fails (no creds), we log a warning.
        try:
            self.client = storage.Client()
            self.valid = True
        except Exception as e:
            print(f"⚠️ GCS Warning: Could not connect to Google Cloud. {e}")
            print("⚠️ Files will be saved LOCALLY only.")
            self.valid = False

    def upload_file(self, file_path: str, bucket_name: str, folder="images"):
        """
        Uploads a local file to GCS and returns the public URL.
        """
        if not self.valid:
            return f"http://localhost/mock/{os.path.basename(file_path)}"

        try:
            bucket = self.client.bucket(bucket_name)
            
            # Create a unique filename to prevent overwrites
            filename = f"{folder}/{uuid.uuid4()}_{os.path.basename(file_path)}"
            blob = bucket.blob(filename)
            
            blob.upload_from_filename(file_path)
            
            # Make public (optional, depends on client security needs)
            # blob.make_public()
            
            return blob.public_url
        except Exception as e:
            print(f"❌ Upload Failed: {e}")
            return None