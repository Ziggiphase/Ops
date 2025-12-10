import os
import cv2
import torch
import subprocess
import shutil
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class AIEngine:
    def __init__(self):
        # 1. Detect Hardware
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.fp16 = True
            print("🚀 AI Engine: GPU Detected! (High Speed Mode)")
        else:
            self.device = torch.device('cpu')
            self.fp16 = False
            print("⚠️ AI Engine: No GPU found. Running in CPU Mode.")

        self.weights_dir = "app/models/weights"
        
        # SMART PATH DETECTION for CodeFormer
        # Logic: Check Docker path first -> Check Local path -> Clone if missing
        docker_path = "/app/CodeFormer"
        local_path = "CodeFormer"

        if os.path.exists(docker_path):
            self.codeformer_dir = docker_path
            print(f"✅ Loaded CodeFormer from Docker Path: {self.codeformer_dir}")
        elif os.path.exists(local_path):
            self.codeformer_dir = local_path
            print(f"✅ Loaded CodeFormer from Local Path: {self.codeformer_dir}")
        else:
            print("⚠️ CodeFormer not found. Cloning it now (Auto-Setup)...")
            try:
                subprocess.run(["git", "clone", "https://github.com/sczhou/CodeFormer.git", "CodeFormer"], check=True)
                self.codeformer_dir = local_path
                
                # Auto-install requirements if we just cloned it
                print("⚙️ Installing CodeFormer dependencies...")
                subprocess.run(["pip", "install", "-r", f"{local_path}/requirements.txt"], check=True)
                subprocess.run(["python", f"{local_path}/basicsr/setup.py", "develop"], check=True)
            except Exception as e:
                print(f"❌ Failed to clone CodeFormer: {e}")
                self.codeformer_dir = None

        # 2. Initialize Background Upscaler (Real-ESRGAN)
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

    def run_codeformer(self, input_path):
        """
        Runs CodeFormer as a subprocess command because it is a script, not a library.
        """
        if not self.codeformer_dir:
            print("⚠️ CodeFormer directory missing. Skipping face restoration.")
            return input_path

        print("⚡ Running CodeFormer Process...")
        
        # Define output directory (CodeFormer requires a folder, not a file path)
        # We use a temp subfolder to avoid filename conflicts
        temp_output_dir = "temp_uploads/results"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Command to run inference_codeformer.py
        # -w 0.7: Fidelity weight (0.7 is the sweet spot between restoration and reality)
        # --input_path: The image to fix
        # --output_path: Where to save it
        cmd = [
            "python", "inference_codeformer.py",
            "-w", "0.7",
            "--input_path", os.path.abspath(input_path),
            "--output_path", os.path.abspath(temp_output_dir),
            "--has_aligned" # Assume unaligned faces (standard photos)
        ]
        
        try:
            # IMPORTANT: We run inside the CodeFormer directory so it finds its own config files
            subprocess.run(cmd, cwd=self.codeformer_dir, check=True)
        except Exception as e:
            print(f"❌ CodeFormer Failed (likely no GPU or missing repo): {e}")
            return input_path

        # CodeFormer saves the result in: temp_uploads/results/final_results/<filename>.png
        filename = os.path.basename(input_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # CodeFormer forces output to be .png usually
        result_path = os.path.join(temp_output_dir, "final_results", f"{name_no_ext}.png")
        
        # Verify if file exists, otherwise look for .jpg variant or return original
        if os.path.exists(result_path):
            return result_path
        
        print(f"⚠️ CodeFormer finished but output file not found at {result_path}")
        return input_path

    def enhance(self, img_path: str, output_path: str):
        """
        Runs the Hybrid Pipeline:
        1. CodeFormer (Face Restoration)
        2. Real-ESRGAN (Background Upscaling)
        """
        
        # Step 1: Face Restoration
        # This returns the path to the face-fixed image (or original if failed)
        restored_path = self.run_codeformer(img_path)
        
        # Read the result
        img = cv2.imread(restored_path, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback if OpenCV fails to read the CodeFormer output
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        print("⚡ Starting Background Upscaling (Real-ESRGAN)...")
        
        # Step 2: Global Upscaling
        # output is the raw numpy array image
        output, _ = self.upsampler.enhance(img, outscale=2)

        # Save final result
        cv2.imwrite(output_path, output)
        return output_path