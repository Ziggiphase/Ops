FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Setup CodeFormer
RUN git clone https://github.com/sczhou/CodeFormer.git CodeFormer

# 2. Install Dependencies
# --- FIX STARTS HERE ---
# We must copy the requirements file BEFORE we try to install from it
COPY requirements.txt . 
# --- FIX ENDS HERE ---

RUN pip3 install --no-cache-dir -r CodeFormer/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 uninstall -y basicsr

# 3. Copy Your Patched Code
COPY basicsr /app/basicsr
COPY app /app/app
COPY *.py /app/

# 4. Link the Local Library
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/CodeFormer"

# 5. Download Weights
WORKDIR /app/CodeFormer
RUN python3 scripts/download_pretrained_models.py CodeFormer
RUN python3 scripts/download_pretrained_models.py facelib

WORKDIR /app
RUN python3 setup_models.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
