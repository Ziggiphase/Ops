import os
import torch
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

class AIEngine:
    def __init__(self):
        # 1. Setup Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("⚡ AI Engine: Running on NVIDIA GPU")
        else:
            self.device = torch.device('cpu')
            print("⚠️ AI Engine: Running on CPU (Explicit Fallback)")

        # 2. Setup Background Upsampler (Real-ESRGAN)
        print("⚡ Loading Real-ESRGAN...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='/app/weights/RealESRGAN_x2plus.pth',
            model=model,
            tile=200,       # Increased slightly (100 was too small/slow, 200 is balanced)
            tile_pad=10,
            pre_pad=0,
            half=False,     
            device=self.device
        )

        # 3. Setup Face Enhancer (GFPGAN)
        print("⚡ Loading GFPGAN...")
        self.face_enhancer = GFPGANer(
            model_path='/app/weights/GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
            device=self.device
        )

    def enhance(self, input_path: str, output_path: str):
        print(f"⚡ Processing: {input_path}")
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)

        # --- SAFETY RESIZE START ---
        # If image is massive (>1200px), shrink it to prevent RAM crash.
        # 1200px input -> 2400px output (High Quality, Safe RAM)
        max_dimension = 1200
        height, width = img.shape[:2]
        
        if width > max_dimension or height > max_dimension:
            print(f"⚠️ Image too large ({width}x{height}). Resizing to safe limit...")
            scale_factor = max_dimension / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"✅ Resized to {new_width}x{new_height}")
        # --- SAFETY RESIZE END ---

        # GFPGAN handles the whole pipeline
        _, _, output = self.face_enhancer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        cv2.imwrite(output_path, output)
        return output_path