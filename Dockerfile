FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install "Universal" PyTorch (CUDA 11.7 support)
# This version works on GPU if present, and falls back to CPU automatically.
RUN pip3 install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# The "Holy Trinity"
RUN pip3 install --no-cache-dir basicsr==1.4.2 gfpgan realesrgan

# Install requirements
COPY requirements.txt .
RUN sed -i '/torch/d' requirements.txt
RUN sed -i '/basicsr/d' requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# 3. Download Weights (Updated Links)
WORKDIR /app/weights
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# 4. Copy Application Code
WORKDIR /app
COPY app /app/app
COPY *.py /app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]