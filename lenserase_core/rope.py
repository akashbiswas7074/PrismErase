"""
Mask-Aware RoPE (3D Rotary Positional Encoding) for LensErase
Reference: Section 3.1 & Eq. 1, 2 of "LensErase: A Foundation Model for Erase Anything" (arXiv:2610.18072)

Positional Coordinates:
- Source condition tokens: p_cond_i = (m_i, h_i, w_i), where m_i in {0, 1}
  (1 = masked/erased region, 0 = unmasked/preserved region)
- Noisy target tokens:   p_img_i  = (2, h_i, w_i), where 2 is proxy modality coordinate
"""

import math
import numpy as np

def get_3d_rotary_frequencies(head_dim: int, base: float = 10000.0,
                              split_ratios: tuple = (0.25, 0.375, 0.375)):
    """
    Generate inverse frequencies for the 3 axes: (modality, height, width).
    head_dim must be divisible by 2.
    split_ratios defines how head_dim // 2 frequency bands are partitioned across:
      0: Modality axis (mask flag {0, 1, 2})
      1: Height axis (h_i)
      2: Width axis (w_i)
    """
    total_pairs = head_dim // 2
    dim_m = int(total_pairs * split_ratios[0])
    dim_h = int(total_pairs * split_ratios[1])
    dim_w = total_pairs - dim_m - dim_h

    def _inv_freqs(num_bands):
        if num_bands == 0:
            return np.array([])
        return 1.0 / (base ** (np.arange(0, num_bands) / max(1, num_bands)))

    inv_freq_m = _inv_freqs(dim_m)
    inv_freq_h = _inv_freqs(dim_h)
    inv_freq_w = _inv_freqs(dim_w)

    return inv_freq_m, inv_freq_h, inv_freq_w

def compute_3d_rope_embeddings(coords: np.ndarray, head_dim: int, base: float = 10000.0):
    """
    Compute cos and sin rotary embeddings for a batch of 3D coordinates.
    coords: shape (..., 3) containing [modality, h, w]
    Returns:
      cos: shape (..., head_dim)
      sin: shape (..., head_dim)
    """
    inv_m, inv_h, inv_w = get_3d_rotary_frequencies(head_dim, base)

    m_coord = coords[..., 0:1] # (...)
    h_coord = coords[..., 1:2]
    w_coord = coords[..., 2:3]

    angles_m = m_coord * inv_m # (..., dim_m)
    angles_h = h_coord * inv_h # (..., dim_h)
    angles_w = w_coord * inv_w # (..., dim_w)

    angles = np.concatenate([angles_m, angles_h, angles_w], axis=-1) # (..., head_dim//2)
    # Duplicate for full head_dim
    angles = np.concatenate([angles, angles], axis=-1) # (..., head_dim)

    return np.cos(angles), np.sin(angles)

def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply rotary embeddings to vector x:
      x_rot = x * cos + rotate_half(x) * sin
    where rotate_half([-x2, x1]).
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotated_half = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated_half * sin

def build_lenserase_token_coordinates(mask_binary: np.ndarray, patch_size: int = 16):
    """
    Convert a full-resolution binary mask (H, W) into downsampled source token coordinates
    p_cond_i = (m_i, h_i, w_i) and image token coordinates p_img_i = (2, h_i, w_i).
    """
    H, W = mask_binary.shape
    num_h = H // patch_size
    num_w = W // patch_size

    # Max-pooling mask to token grid
    # Reshape and take max over patch
    trimmed_mask = mask_binary[:num_h * patch_size, :num_w * patch_size]
    pooled_mask = trimmed_mask.reshape(num_h, patch_size, num_w, patch_size).max(axis=(1, 3))
    m_i = (pooled_mask > 0.5).astype(np.int32) # {0, 1}

    grid_h, grid_w = np.meshgrid(np.arange(num_h), np.arange(num_w), indexing='ij')

    # Condition branch coords: (m_i, h_i, w_i)
    coords_cond = np.stack([m_i, grid_h, grid_w], axis=-1).reshape(-1, 3)

    # Noisy target image branch coords: (2, h_i, w_i)
    coords_img = np.stack([np.full_like(grid_h, 2), grid_h, grid_w], axis=-1).reshape(-1, 3)

    return coords_cond, coords_img, (num_h, num_w)
