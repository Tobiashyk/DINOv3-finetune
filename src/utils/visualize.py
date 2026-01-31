import torch
import numpy as np
from sklearn.decomposition import PCA


def pca_transform_features(features: torch.Tensor, patch_h: int, patch_w: int, n_components: int = 1) -> np.ndarray:
    """
    Apply PCA transformation to features and normalize the result.

    Args:
        features: Tensor of shape (B, N, F) where B is batch size, N is number of patches, F is feature dimension
        patch_h: Height of the patch grid
        patch_w: Width of the patch grid
        n_components: Number of PCA components (default: 1)

    Returns:
        Normalized PCA features of shape (B, patch_h, patch_w, n_components)
    """
    B, N, F = features.shape

    # Apply PCA transformation
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features.reshape(B * N, F).cpu().numpy())
    pca_rgb = pca_features.reshape(B, patch_h, patch_w, n_components)

    # Normalize to [0, 1]
    pca_rgb_norm = (pca_rgb - pca_rgb.min()) / (pca_rgb.max() - pca_rgb.min())

    return pca_rgb_norm
