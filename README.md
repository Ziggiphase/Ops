e.

```markdown
# 📸 Magazine Enhancer AI

A high-performance AI API designed to restore and upscale vintage magazine scans. It combines **GFPGAN** (for face restoration) and **Real-ESRGAN** (for background upscaling) into a single, seamless pipeline.

## 🚀 Features
* **Face Restoration:** Automatically detects and repairs damaged, blurry, or low-resolution faces using GFPGAN.
* **Smart Upscaling:** Sharpen background details and text by 2x-4x using Real-ESRGAN.
* **Auto-Scaling:** Automatically resizes massive inputs to prevent memory crashes on standard laptops.
* **Dockerized:** Runs consistently on any machine (Windows/Mac/Linux) with zero dependency conflicts.
* **Hybrid Mode:** Supports both **Local Mode** (offline) and **Cloud Mode** (Google Cloud Storage).

---

## 🛠️ Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* **NVIDIA Drivers** (Optional, only required for GPU acceleration).

---

## ⚡ Quick Start (CPU & Standard Laptops)
*Recommended for most users.*

1.  **Build and Start the Server:**
    Open your terminal in this project folder and run:
    ```bash
    docker-compose up --build
    ```
    *Note: The first run may take a few minutes to download the AI models (approx. 300MB).*

2.  **Access the API:**
    Open your browser to: [http://localhost:8000/docs](http://localhost:8000/docs)

3.  **Test an Image:**
    * Click **POST /api/v1/enhance**.
    * Click **Try it out**.
    * Upload an image and click **Execute**.

4.  **View Results:**
    Do not close the terminal. Go to your file explorer and open the `temp_uploads` folder inside this project. Your enhanced image will appear there automatically.

---

## 🚀 Hardware Acceleration (NVIDIA GPU Users)
*For users with NVIDIA GPUs (CUDA) for faster processing.*

Run this command to unlock GPU access:
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

```

---

## ☁️ Configuration: Local vs. Cloud

### **1. Local Mode (Default)**

* **Behavior:** Images are saved directly to your computer in the `temp_uploads` folder.
* **Setup:** No configuration needed. Just run the app.

### **2. Google Cloud Mode (Production)**

* **Behavior:** Images are uploaded to a Google Cloud Storage bucket.
* **Setup:**
1. Place your Google Cloud service account key file in the root directory and name it `credentials.json`.
2. Restart the container. The app will automatically detect the key and switch to Cloud Mode.


> **⚠️ IMPORTANT:** Ensure you create a bucket named `photo_enhance` in your Google Cloud Console before adding your credentials file.


> If you want to use a different bucket name, you **must** update the `GCS_BUCKET_ORIGINAL` and `GCS_BUCKET_ENHANCED` variables in the `docker-compose.yml` file before starting the app.



---

## 📁 Project Structure

* `Dockerfile`: Defines the custom AI environment (Python 3.9, PyTorch 1.13 + CUDA 11.7).
* `docker-compose.yml`: Main configuration with volume mapping for local file access.
* `app/services/ai_engine.py`: The core logic combining GFPGAN and Real-ESRGAN.
* `app/weights/`: Stores the AI models (downloaded automatically on first run).
* `temp_uploads/`: Shared folder where enhanced images appear locally.

---

## ⚠️ Troubleshooting

**1. "Port already allocated"**

* **Fix:** Ensure no other service is running on port 8000. If needed, change the port in `docker-compose.yml`.

**2. Slow Processing**

* **Context:** On CPU-only machines, large images may take 1-3 minutes to process. This is normal behavior for deep learning models running without a GPU.

**3. "Out of Memory" or Container Crash**

* **Fix:** The system automatically resizes images larger than 1200px to prevent crashes. If it still crashes, try closing other heavy applications (Chrome tabs, Photoshop) to free up RAM.

**4. Permission Denied when deleting `temp_uploads**`

* **Fix:** Since Docker creates the files, they may be owned by "root". Use sudo to delete them:
```bash
sudo rm -rf temp_uploads/*

```



---

**Built with ❤️ using GFPGAN and Real-ESRGAN.**

```

```
