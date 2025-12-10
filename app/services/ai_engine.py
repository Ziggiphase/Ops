import os
import cv2
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

class AIEngine:
    def __init__(self):
        # 1. Detect Hardware (The "Hybrid" Requirement)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.fp16 = True
            print("🚀 AI Engine: GPU Detected! (High Speed Mode)")
        else:
            self.device = torch.device('cpu')
            self.fp16 = False
            print("⚠️ AI Engine: No GPU found. Running in CPU Mode (Slower).")

        self.weights_dir = "app/models/weights"
        
        # 2. Initialize Face Restorer (GFPGAN)
        # We use this to fix eyes/skin before upscaling
        self.face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None, # We will use RealESRGAN for BG
            device=self.device
        )

        # 3. Initialize Background Upscaler (Real-ESRGAN)
        # This handles the clothes, hair details, and background
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        
        self.upsampler = RealESRGANer(
            scale=2,
            model_path=os.path.join(self.weights_dir, "RealESRGAN_x2plus.pth"),
            model=model,
            tile=0, # 0 = auto-split for large images to prevent RAM crash
            tile_pad=10,
            pre_pad=0,
            half=self.fp16,
            device=self.device
        )

    def enhance(self, img_path: str, output_path: str):
        """
        Runs the Hybrid Pipeline:
        1. Read Image
        2. Face Restoration (GFPGAN)
        3. Background Upscaling (RealESRGAN)
        """
        # Read image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")

        print("⚡ Starting Face Restoration...")
        # GFPGAN inference
        # cropped_faces, restored_faces, restored_img
        _, _, restored_img = self.face_enhancer.enhance(
            img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True
        )

        print("⚡ Starting Background Upscaling...")
        # Real-ESRGAN inference on the face-restored image
        output, _ = self.upsampler.enhance(restored_img, outscale=2)

        # Save result
        cv2.imwrite(output_path, output)
        return output_path