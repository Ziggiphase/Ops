# Use NVIDIA CUDA base image (Ubuntu 22.04 + CUDA 11.8 + cuDNN 8)
# This ensures GPU drivers are pre-installed.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install System Dependencies (Python + OpenGL for OpenCV)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python Dependencies
# We copy requirements first to cache this layer
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Fix for the torchvision/BasicSR compatibility issue we found earlier
# We apply the patch automatically during build
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py

# Copy Application Code
COPY . .

# Download Models (Built-in Step)
# This ensures the container has weights ready on startup
RUN python3 setup_models.py

# Expose Port
EXPOSE 8000

# Run the Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]