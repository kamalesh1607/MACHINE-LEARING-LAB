"""
Dataset Management, Dynamic Class Discovery, Feature Batch Extraction, and Sample Generator
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features, extract_hog_features, extract_glcm_features
from preprocessing import preprocess_image

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_dataset(dataset_dir: str) -> dict:
    """
    Scans a folder-based dataset directory dynamically.
    Auto-discovers class folders and image files.
    """
    if not os.path.exists(dataset_dir) or not os.path.isdir(dataset_dir):
        return {
            "exists": False,
            "error": f"Directory '{dataset_dir}' does not exist.",
            "classes": [],
            "class_counts": {},
            "total_images": 0,
            "image_items": []
        }

    # Find subdirectories that contain image files
    subdirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ]
    subdirs.sort()

    classes = []
    class_counts = {}
    image_items = []  # list of dicts: {'path': ..., 'class_name': ..., 'class_idx': ...}

    for class_name in subdirs:
        class_path = os.path.join(dataset_dir, class_name)
        valid_files = []
        for root, _, files in os.walk(class_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in VALID_IMAGE_EXTENSIONS:
                    valid_files.append(os.path.join(root, file))

        if len(valid_files) > 0:
            classes.append(class_name)
            class_counts[class_name] = len(valid_files)
            for fpath in valid_files:
                image_items.append({
                    "path": fpath,
                    "class_name": class_name,
                    "filename": os.path.basename(fpath)
                })

    # Assign class indices
    for item in image_items:
        item["class_idx"] = classes.index(item["class_name"])

    return {
        "exists": True,
        "classes": classes,
        "class_counts": class_counts,
        "total_images": len(image_items),
        "image_items": image_items,
        "error": None if len(classes) > 0 else f"No image subfolders found in '{dataset_dir}'."
    }


def extract_dataset_features(
    image_items: list,
    progress_callback=None
) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """
    Extracts HOG + GLCM features from all images in the dataset.

    Returns:
        X: numpy array of shape (N, feature_dim)
        y: numpy array of shape (N,) with class names
        class_names: list of unique class names
        summary: dict containing counts and timings
    """
    X_list = []
    y_list = []
    total = len(image_items)

    for idx, item in enumerate(image_items):
        img_path = item["path"]
        class_name = item["class_name"]

        try:
            # Read image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                # Try PIL fallback
                pil_img = Image.open(img_path).convert("RGB")
                img_rgb = np.array(pil_img)
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Preprocess
            pre = preprocess_image(img_rgb)
            enhanced = pre["enhanced"]

            # Extract HOG and GLCM
            hog_feat = extract_hog_features(enhanced, visualize=False)
            glcm_feat, _ = extract_glcm_features(enhanced)

            combined_feat = np.hstack([hog_feat, glcm_feat])
            X_list.append(combined_feat)
            y_list.append(class_name)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

        if progress_callback:
            progress_callback(idx + 1, total)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    unique_classes = sorted(list(set(y_list)))

    summary = {
        "total_processed": len(X_list),
        "feature_dim": X.shape[1] if len(X) > 0 else 0,
        "classes": unique_classes,
        "class_distribution": {c: int(np.sum(y == c)) for c in unique_classes}
    }

    return X, y, unique_classes, summary


def perform_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42
) -> tuple:
    """
    Performs train-test split with safe stratification handling.
    """
    # Check if stratification is safe
    classes, counts = np.unique(y, return_counts=True)
    can_stratify = len(classes) > 1 and all(c >= 2 for c in counts)
    stratify_param = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    split_info = {
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_pct": round(len(X_train) / len(X) * 100, 1),
        "test_pct": round(len(X_test) / len(X) * 100, 1),
        "stratified": can_stratify,
        "classes": list(classes),
        "train_class_counts": {c: int(np.sum(y_train == c)) for c in classes},
        "test_class_counts": {c: int(np.sum(y_test == c)) for c in classes}
    }

    return X_train, X_test, y_train, y_test, split_info


def generate_sample_dataset(
    base_dir: str = "dataset",
    samples_per_class: int = 20
) -> dict:
    """
    Generates a realistic synthetic ultrasound dataset for academic demonstration
    with distinctive acoustic texture profiles:
    - Normal: Regular fine speckle, uniform parenchymal echogenicity.
    - Fatty_Liver: Bright diffuse hyperechoic texture, higher speckle density, posterior attenuation.
    - Cirrhosis: Coarse heterogeneous nodular texture, fibrotic reticular bands, acoustic shadowing.
    """
    os.makedirs(base_dir, exist_ok=True)
    classes = ["Normal", "Fatty_Liver", "Cirrhosis"]
    created_counts = {}

    rng = np.random.default_rng(seed=42)

    for class_name in classes:
        class_folder = os.path.join(base_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        created_counts[class_name] = 0

        for i in range(samples_per_class):
            img_filename = f"{class_name.lower()}_sample_{i+1:03d}.png"
            img_path = os.path.join(class_folder, img_filename)

            # Generate synthetic ultrasound B-mode texture image (256x256)
            size = (256, 256)
            y_coords, x_coords = np.mgrid[0:size[0], 0:size[1]]

            # Base sector probe masking (curvilinear ultrasound fan geometry)
            center_x, center_y = size[1] / 2.0, -40.0
            dist_from_probe = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
            angle = np.arctan2(x_coords - center_x, y_coords - center_y)
            sector_mask = (dist_from_probe >= 60) & (dist_from_probe <= 240) & (np.abs(angle) <= np.radians(40))

            if class_name == "Normal":
                # Medium baseline intensity, Rayleigh/Gamma distributed speckle
                base_intensity = 110.0 + 10.0 * np.sin(dist_from_probe / 20.0)
                # Speckle noise
                speckle = rng.gamma(shape=3.0, scale=12.0, size=size)
                # Fine texture
                smooth_bg = cv2.GaussianBlur(rng.normal(0, 10, size), (11, 11), 0)
                texture = base_intensity + speckle + smooth_bg

            elif class_name == "Fatty_Liver":
                # Bright hyperechoic liver (steatosis) with high echogenicity in near field and depth attenuation
                depth_attenuation = np.exp(-(y_coords - 50.0) / 180.0)
                depth_attenuation = np.clip(depth_attenuation, 0.4, 1.2)
                base_intensity = 160.0 * depth_attenuation
                # Dense bright speckle
                speckle = rng.gamma(shape=5.0, scale=16.0, size=size)
                smooth_bg = cv2.GaussianBlur(rng.normal(0, 8, size), (7, 7), 0)
                texture = base_intensity + speckle + smooth_bg

            elif class_name == "Cirrhosis":
                # Coarse heterogeneous nodular texture with irregular hyperechoic fibrotic reticular bands
                base_intensity = 120.0
                # Coarse nodules
                nodules = cv2.resize(rng.normal(0, 35, (32, 32)), size, interpolation=cv2.INTER_CUBIC)
                # Fibrotic bands
                fibrosis = np.abs(cv2.Sobel(nodules, cv2.CV_64F, 1, 1, ksize=5)) * 1.5
                speckle = rng.gamma(shape=2.5, scale=18.0, size=size)
                texture = base_intensity + nodules + fibrosis + speckle

            # Apply ultrasound fan sector mask and dark background
            texture = np.clip(texture, 0, 255).astype(np.float32)
            texture[~sector_mask] = rng.uniform(0, 12, size)[~sector_mask]

            # Add probe grid depth lines / acoustic markers
            img_uint8 = texture.astype(np.uint8)
            # Add subtle acoustic beam noise
            beam_noise = (rng.uniform(-5, 5, (1, size[1]))).astype(np.int16)
            img_uint8 = np.clip(img_uint8.astype(np.int16) + beam_noise, 0, 255).astype(np.uint8)

            cv2.imwrite(img_path, img_uint8)
            created_counts[class_name] += 1

    return {
        "status": "success",
        "dataset_dir": base_dir,
        "classes": classes,
        "created_counts": created_counts,
        "total_created": sum(created_counts.values())
    }
