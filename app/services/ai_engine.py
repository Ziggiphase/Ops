import os
import torch
import cv2
import numpy as np
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from app.config import settings

# CodeFormer Import
from basicsr.utils.registry import ARCH_REGISTRY

class AIEngine:
    def __init__(self):
        # 1. Explicit Device Control (CPU-First fallback)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("⚡ AI Engine: Running on NVIDIA GPU")
        else:
            self.device = torch.device('cpu')
            print("⚠️ AI Engine: Running on CPU (Explicit Fallback)")

        # 2. Load CodeFormer (Face Restoration)
        print("⚡ Loading CodeFormer...")
        self.codeformer = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(self.device)
        
        # Load weights
        ckpt_path = '/installations/CodeFormer/weights/CodeFormer/codeformer.pth'
        checkpoint = torch.load(ckpt_path, map_location=self.device)['params_ema']
        self.codeformer.load_state_dict(checkpoint)
        self.codeformer.eval()

        # 3. Load Real-ESRGAN (Background Upscaling)
        print("⚡ Loading Real-ESRGAN...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.upsampler = RealESRGANer(
            scale=2,
            model_path='app/models/weights/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False, # Force FP32 for CPU compatibility
            device=self.device
        )

    def enhance(self, input_path: str, output_path: str):
        # Read Image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        
        # Step 1: Face Restoration (CodeFormer)
        # w=0.8 means HIGH FIDELITY (Less hallucination, more realism)
        # w=0.5 is default (more fake details)
        print("⚡ Running CodeFormer (Fidelity: 0.8)...")
        face_helper = self._get_face_helper(img)
        
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    # 'w' is the Balance Weight. 
                    # 0.1 = Artificial/Sharp. 
                    # 0.9 = Real/Blurry. 
                    # 0.8 is the Editorial Sweet Spot.
                    output_face = self.codeformer(cropped_face_t, w=0.8, adain=True)[0]
                    restored_face = tensor2img(output_face, rgb2bgr=True, min_max=(-1, 1))
                del output_face
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Face Error: {e}")
                restored_face = cropped_face # Fallback

            face_helper.add_restored_face(restored_face, cropped_face)

        # Paste faces back
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()

        # Step 2: Background Upscaling
        print("⚡ Upscaling Background...")
        output, _ = self.upsampler.enhance(restored_img, outscale=2)

        # Save
        cv2.imwrite(output_path, output)
        return output_path

    def _get_face_helper(self, img):
        # Helper to instantiate FaceHelper cleanly
        from facexlib.utils.face_restoration_helper import FaceRestorationHelper
        face_helper = FaceRestorationHelper(
            1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', use_parse=True, device=self.device
        )
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()
        return face_helper