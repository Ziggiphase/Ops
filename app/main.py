from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints
import uvicorn
import os

# Create the App
app = FastAPI(
    title="Magazine-Grade Image Enhancer",
    description="API for Flutter App: Upscaling (Real-ESRGAN) + Face Fix (GFPGAN/CodeFormer) + Color Grading",
    version="1.0.0"
)

# CORS Security (Critical for Mobile Apps)
# Allows the Flutter app to send requests without getting blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to the specific App domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect the /enhance endpoint
app.include_router(endpoints.router, prefix="/api/v1", tags=["Enhancement"])

@app.get("/")
def health_check():
    """
    Simple route to check if server is running.
    """
    return {"status": "online", "gpu_enabled": False, "message": "System Ready"}

if __name__ == "__main__":
    # Local Development Entry Point
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)