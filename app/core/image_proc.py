import cv2
import numpy as np
from PIL import Image, ImageEnhance

class MagazineEnhancer:
    """
    Implements the client's specific 'Magazine-Grade' post-processing recipe.
    This runs AFTER the AI model upscaling.
    """

    @staticmethod
    def apply_magazine_look(image_path: str, output_path: str):
        # Load image with PIL for color/exposure work
        img = Image.open(image_path).convert("RGB")

        # 1. Exposure Lift (+8% average)
        # Client asked for +5-12%, we target the middle ground
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.08)

        # 2. Vibrance / Saturation (+8% average)
        # Client asked for +5-11% to avoid oversaturated skin
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.08)

        # 3. Contrast Boost (+8% average)
        # Client asked for +6-10% for "premium depth"
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.08)

        # Convert to OpenCV format for structural work
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

        # 4. Warm Tone Bias (+5% Red channel boost)
        # Client asked for "Warm tone bias +3-7%"
        # We slightly increase the Red channel and decrease Blue channel slightly
        b, g, r = cv2.split(open_cv_image)
        r = cv2.addWeighted(r, 1.05, 0, 0, 0) # +5% Red
        b = cv2.addWeighted(b, 0.98, 0, 0, 0) # -2% Blue (to warm it up)
        open_cv_image = cv2.merge([b, g, r])

        # 5. Micro-Detail Enhancement (Unsharp Masking)
        # Client asked for "Micro-detail enhancement +10-20%"
        # We use a Gaussian Blur subtraction method to sharpen edges
        gaussian_3 = cv2.GaussianBlur(open_cv_image, (0, 0), 3.0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1.15, gaussian_3, -0.15, 0)

        # Save the final result
        cv2.imwrite(output_path, open_cv_image)
        return output_path