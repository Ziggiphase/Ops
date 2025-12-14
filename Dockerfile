FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies (Fixes libGL.so.1 Error)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Setup CodeFormer in a separate system folder
WORKDIR /installations
RUN git clone https://github.com/sczhou/CodeFormer.git CodeFormer

# 3. Install Dependencies
WORKDIR /app
COPY requirements.txt .

# Install CodeFormer requirements first
RUN pip3 install --no-cache-dir -r /installations/CodeFormer/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 uninstall -y basicsr

# 4. Copy Application Code
COPY basicsr /app/basicsr
COPY app /app/app
COPY *.py /app/

# 5. FORCE-CREATE VERSION FILE (Fixes 'basicsr.version' syntax error)
RUN echo "__version__ = '1.4.2'" > /app/basicsr/version.py && \
    echo "__gitsha__ = 'unknown'" >> /app/basicsr/version.py

# 6. Set Environment Paths
ENV PYTHONPATH="${PYTHONPATH}:/app:/installations/CodeFormer"

# 7. Download Weights
WORKDIR /installations/CodeFormer
RUN python3 scripts/download_pretrained_models.py CodeFormer
RUN python3 scripts/download_pretrained_models.py facelib

WORKDIR /app
RUN python3 setup_models.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]