import os
import requests
from tqdm import tqdm

# Define paths
WEIGHTS_DIR = "app/models/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# URLs for the specific models the client needs
MODELS = {
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "parsing_parsenet.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth"
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"✅ {os.path.basename(dest_path)} already exists.")
        return
    
    print(f"⬇️ Downloading {os.path.basename(dest_path)}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

if __name__ == "__main__":
    print("🚀 Starting Model Setup for Codespaces...")
    for filename, url in MODELS.items():
        download_file(url, os.path.join(WEIGHTS_DIR, filename))
    print("✨ All models ready!")