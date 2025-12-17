import os
import requests
from pathlib import Path

# Define the models we need (Updated RealESRGAN URL)
MODELS = {
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
}

# In Docker, we put weights here
WEIGHTS_DIR = Path("/app/weights")

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"✅ {dest_path.name} already exists.")
        return
    
    print(f"⬇️ Downloading {dest_path.name}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded {dest_path.name}")
    except Exception as e:
        print(f"❌ Failed to download {dest_path.name}: {e}")

def setup():
    # Create directory if it doesn't exist
    if not WEIGHTS_DIR.exists():
        print(f"📂 Creating weights directory: {WEIGHTS_DIR}")
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in MODELS.items():
        dest_path = WEIGHTS_DIR / filename
        download_file(url, dest_path)

if __name__ == "__main__":
    setup()
