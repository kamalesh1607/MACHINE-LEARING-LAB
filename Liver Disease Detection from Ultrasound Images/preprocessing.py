"""
Enhanced Image Preprocessing Module for Liver Ultrasound Classification
Pipeline: Input -> Resize -> Grayscale -> Denoise (Gaussian Blur) -> Contrast Enhancement (Hist Eq / CLAHE) -> Normalization
"""

import cv2
import numpy as np
from PIL import Image
import io


def load_image_as_rgb(image_input) -> np.ndarray:
    """
    Safely load an input image (PIL Image, file buffer, file path, or numpy array)
    and return an RGB numpy array (uint8).
    """
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        elif image_input.shape[2] == 3:
            return image_input.copy()
        return image_input

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    if isinstance(image_input, (str, bytes, io.BytesIO)) or hasattr(image_input, "read"):
        if hasattr(image_input, "read"):
            image_input.seek(0)
            pil_img = Image.open(image_input)
            return np.array(pil_img.convert("RGB"))
        elif isinstance(image_input, str):
            pil_img = Image.open(image_input)
            return np.array(pil_img.convert("RGB"))
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
            return np.array(pil_img.convert("RGB"))

    raise ValueError("Unsupported image input format.")


def preprocess_image(
    image_input,
    target_size=(128, 128),
    blur_kernel=(5, 5),
    blur_sigma=0,
    enhancement_method="Histogram Equalization",
    clahe_clip_limit=2.0,
    clahe_grid_size=(8, 8)
) -> dict:
    """
    Executes the complete preprocessing pipeline for liver ultrasound images.

    Stages:
    1. Input Ultrasound Image
    2. Resize to target_size (default: 128x128)
    3. Grayscale conversion
    4. Noise reduction via Gaussian Blur
    5. Contrast enhancement via Histogram Equalization or CLAHE
    6. Pixel normalization (0.0 to 1.0)
    """
    # 1. Load Original RGB
    original_rgb = load_image_as_rgb(image_input)

    # 2. Resize to 128x128
    resized_rgb = cv2.resize(original_rgb, target_size, interpolation=cv2.INTER_AREA)

    # 3. Grayscale Conversion
    grayscale = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)

    # 4. Noise Reduction (Gaussian Blur)
    # Ensure kernel dimensions are positive odd integers
    k_w = max(1, blur_kernel[0] if blur_kernel[0] % 2 != 0 else blur_kernel[0] + 1)
    k_h = max(1, blur_kernel[1] if blur_kernel[1] % 2 != 0 else blur_kernel[1] + 1)
    denoised = cv2.GaussianBlur(grayscale, (k_w, k_h), blur_sigma)

    # 5. Contrast Enhancement
    if enhancement_method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=clahe_grid_size)
        enhanced = clahe.apply(denoised)
    else:
        # Standard Global Histogram Equalization
        enhanced = cv2.equalizeHist(denoised)

    # 6. Pixel Normalization (0.0 - 1.0)
    normalized_float = enhanced.astype(np.float32) / 255.0
    normalized_display = (normalized_float * 255.0).astype(np.uint8)

    # Compute intensity difference / edge gradient map
    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_norm = np.clip((gradient_magnitude / (gradient_magnitude.max() + 1e-6)) * 255, 0, 255).astype(np.uint8)

    stats = {
        "original_shape": original_rgb.shape,
        "processed_shape": enhanced.shape,
        "raw_min": int(grayscale.min()),
        "raw_max": int(grayscale.max()),
        "raw_mean": float(grayscale.mean()),
        "raw_std": float(grayscale.std()),
        "enhanced_min": int(enhanced.min()),
        "enhanced_max": int(enhanced.max()),
        "enhanced_mean": float(enhanced.mean()),
        "enhanced_std": float(enhanced.std()),
        "norm_min": float(normalized_float.min()),
        "norm_max": float(normalized_float.max()),
        "contrast_gain": float(enhanced.std() / (grayscale.std() + 1e-6))
    }

    return {
        "original": original_rgb,
        "resized": resized_rgb,
        "grayscale": grayscale,
        "denoised": denoised,
        "enhanced": enhanced,
        "normalized": normalized_float,
        "normalized_display": normalized_display,
        "gradient_map": gradient_norm,
        "stats": stats
    }
