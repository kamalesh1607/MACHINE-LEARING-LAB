"""
Feature Extraction Module for Liver Ultrasound Images
Implements:
1. Histogram of Oriented Gradients (HOG)
2. Gray-Level Co-occurrence Matrix (GLCM) Texture Features
3. Combined Feature Vector (HOG + GLCM)
"""

import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from preprocessing import preprocess_image


def extract_hog_features(
    image,
    orientations: int = 9,
    pixels_per_cell: tuple = (8, 8),
    cells_per_block: tuple = (2, 2),
    block_norm: str = "L2-Hys",
    visualize: bool = False
):
    """
    Extract HOG (Histogram of Oriented Gradients) features from a preprocessed grayscale image.

    Parameters:
        image: 2D numpy array (grayscale uint8, preferably 128x128).
        orientations: Number of gradient orientation bins (default: 9).
        pixels_per_cell: Size (in pixels) of a cell (default: (8, 8)).
        cells_per_block: Number of cells in each block (default: (2, 2)).
        block_norm: Block normalization method (default: 'L2-Hys').
        visualize: Whether to return HOG visualization image.

    Returns:
        hog_features: 1D numpy array of shape (N,)
        hog_image (optional if visualize=True): 2D numpy array visualization
    """
    # Ensure image is 2D uint8
    if len(image.shape) == 3:
        # If passed an RGB or other image, use preprocessed enhanced stage
        pre = preprocess_image(image)
        image = pre["enhanced"]

    if visualize:
        features, hog_image = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=True
        )
        return features.astype(np.float32), hog_image
    else:
        features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=False
        )
        return features.astype(np.float32)


def extract_glcm_features(
    image,
    distances: list = None,
    angles: list = None,
    levels: int = 256
) -> tuple[np.ndarray, dict]:
    """
    Extract GLCM (Gray-Level Co-occurrence Matrix) second-order texture statistics.

    Properties extracted:
    - Contrast: Local intensity variation
    - Dissimilarity: Difference in gray level pairs
    - Homogeneity: Closeness of element distribution to diagonal
    - Energy: Uniformity / Angular Second Moment
    - Correlation: Linear dependency of gray levels

    Returns:
        glcm_vector: 1D numpy array containing all flattened property metrics.
        glcm_dict: Dictionary with aggregate mean and std for each property (useful for UI display).
    """
    if distances is None:
        distances = [1, 3, 5]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Ensure image is 2D uint8
    if len(image.shape) == 3:
        pre = preprocess_image(image)
        image = pre["enhanced"]
    elif image.dtype != np.uint8:
        # Scale if float
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Compute GLCM matrix
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    glcm_values = []
    glcm_summary = {}

    for prop in properties:
        prop_matrix = graycoprops(glcm, prop)
        # prop_matrix has shape (len(distances), len(angles))
        glcm_values.extend(prop_matrix.flatten().tolist())
        glcm_summary[prop] = {
            "mean": float(np.mean(prop_matrix)),
            "std": float(np.std(prop_matrix)),
            "min": float(np.min(prop_matrix)),
            "max": float(np.max(prop_matrix)),
            "matrix": prop_matrix
        }

    return np.array(glcm_values, dtype=np.float32), glcm_summary


def extract_features(image) -> tuple[np.ndarray, dict]:
    """
    Combines HOG and GLCM feature extraction into a single unified 1D feature vector.

    Pipeline:
    1. Preprocess image (Resize to 128x128, Grayscale, Denoise, Enhance).
    2. Extract HOG feature vector.
    3. Extract GLCM feature vector.
    4. Concatenate HOG + GLCM features into a single 1D numpy array.

    Returns:
        combined_feature_vector: 1D numpy array of shape (hog_dim + glcm_dim,)
        metadata: Dictionary with detailed feature breakdown.
    """
    # Preprocess image to standardized enhanced stage
    pre = preprocess_image(image)
    enhanced_img = pre["enhanced"]

    # Extract HOG with visualization
    hog_feats, hog_img = extract_hog_features(enhanced_img, visualize=True)

    # Extract GLCM
    glcm_feats, glcm_summary = extract_glcm_features(enhanced_img)

    # Concatenate
    combined_feats = np.hstack([hog_feats, glcm_feats])

    metadata = {
        "hog_count": int(len(hog_feats)),
        "glcm_count": int(len(glcm_feats)),
        "total_count": int(len(combined_feats)),
        "feature_shape": combined_feats.shape,
        "first_20_features": combined_feats[:20].tolist(),
        "hog_image": hog_img,
        "glcm_summary": glcm_summary,
        "preprocessing_stages": pre
    }

    return combined_feats, metadata
