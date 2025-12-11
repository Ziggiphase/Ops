from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.config import settings  # <--- ADD THIS IMPORT
import shutil
import os
import uuid
from app.services.ai_engine import AIEngine
from app.core.image_proc import MagazineEnhancer
from app.services.gcs import GCSService

router = APIRouter()

# Initialize services ONCE at startup to save time
ai = AIEngine()
gcs = GCSService()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    bucket_original: str = Form(settings.GCS_BUCKET_ORIGINAL), # Default or from App
    bucket_enhanced: str = Form(settings.GCS_BUCKET_ENHANCED)
):
    try:
        # 1. Save uploaded file locally first
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(TEMP_DIR, unique_name)
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. (Optional) Upload Original to GCS Bucket 1
        original_url = gcs.upload_file(input_path, bucket_original, folder="originals")

        # 3. Run AI Pipeline (Face Restore + Upscale)
        ai_output_path = os.path.join(TEMP_DIR, f"ai_{unique_name}")
        ai.enhance(input_path, ai_output_path)

        # 4. Run Magazine Post-Processing (Color/Light)
        final_output_path = os.path.join(TEMP_DIR, f"final_{unique_name}")
        MagazineEnhancer.apply_magazine_look(ai_output_path, final_output_path)

        # 5. Upload Result to GCS Bucket 2
        enhanced_url = gcs.upload_file(final_output_path, bucket_enhanced, folder="enhanced")

        # 6. Cleanup local files
        # os.remove(input_path)
        # os.remove(ai_output_path)
        # os.remove(final_output_path)

        return {
            "status": "success",
            "original_url": original_url,
            "enhanced_url": enhanced_url,
            "message": "Image processed with Magazine-Grade pipeline"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
