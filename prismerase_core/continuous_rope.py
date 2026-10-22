"""
PrismErase: Continuous-Field Distance-Aware 4D Rotary Positional Encoding
========================================================================
Innovation over LensErase (arXiv:2610.18072):
- LensErase Limitation: Uses discrete scalar RoPE coordinates m_i in {0, 1}.
  This causes abrupt phase jumps at mask boundaries, struggles with semi-transparent
  elements, and cannot represent shadow penumbras extending continuously away from the mask.
- PrismErase Solution: Introduces a Continuous Signed Distance & Boundary Curvature
  4D coordinate system:
      p_prism_i = (Phi_i, kappa_i, h_i, w_i)
  where:
    - Phi_i in [-1.0, 1.0]: Normalized signed distance to mask boundary
      (+1 deep inside erase target, 0 at exact boundary, -1 in distant background)
    - kappa_i in [0.0, 1.0]: Local boundary curvature / gradient intensity
    - (h_i, w_i): 2D spatial coordinates
"""

import math
import numpy as np
from scipy import ndimage

def compute_continuous_distance_field(mask_binary: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Computes smooth, signed normalized distance field Phi in [-1.0, 1.0]
    downsampled to the token grid.
    """
    H, W = mask_binary.shape
    nh, nw = H // patch_size, W // P if 'P' in locals() else (H // patch_size, W // patch_size)
    nh, nw = H // patch_size, W // patch_size

    m_bool = (mask_binary > 0.5)

    # Compute Euclidean distance inside mask and outside mask
    dist_outside = ndimage.distance_transform_edt(~m_bool)
    dist_inside = ndimage.distance_transform_edt(m_bool)

    # Signed distance: positive inside, negative outside
    signed_dist = dist_inside - dist_outside # (H, W)

    # Max distance normalization based on image scale
    max_d = max(H, W) / 4.0
    phi_continuous = np.clip(signed_dist / max_d, -1.0, 1.0).astype(np.float32)

    # Downsample to token grid using average pooling for smooth transition
    trimmed_phi = phi_continuous[:nh * patch_size, :nw * patch_size]
    phi_grid = trimmed_phi.reshape(nh, patch_size, nw, patch_size).mean(axis=(1, 3))

    return phi_grid

def compute_boundary_curvature_field(mask_binary: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Computes normalized boundary curvature / edge strength kappa in [0.0, 1.0]
    at token resolution.
    """
    H, W = mask_binary.shape
    nh, nw = H // patch_size, W // patch_size

    # Sobel gradient magnitude
    m_float = mask_binary.astype(np.float32)
    grad_y = ndimage.sobel(m_float, axis=0)
    grad_x = ndimage.sobel(m_float, axis=1)
    grad_mag = np.hypot(grad_x, grad_y)
    grad_mag = np.clip(grad_mag / (np.max(grad_mag) + 1e-8), 0.0, 1.0)

    # Downsample to token grid
    trimmed_k = grad_mag[:nh * patch_size, :nw * patch_size]
    kappa_grid = trimmed_k.reshape(nh, patch_size, nw, patch_size).max(axis=(1, 3))
    return kappa_grid

def get_4d_continuous_frequencies(head_dim: int, base: float = 10000.0,
                                 split_ratios: tuple = (0.2, 0.1, 0.35, 0.35)):
    """
    Partition head_dim // 2 frequency bands across the 4 axes:
    0: Continuous distance field (Phi_i)
    1: Boundary curvature (kappa_i)
    2: Height axis (h_i)
    3: Width axis (w_i)
    """
    total_pairs = head_dim // 2
    dim_phi = int(total_pairs * split_ratios[0])
    dim_kap = int(total_pairs * split_ratios[1])
    dim_h = int(total_pairs * split_ratios[2])
    dim_w = total_pairs - dim_phi - dim_kap - dim_h

    def _inv_freqs(num_bands):
        if num_bands == 0:
            return np.array([])
        return 1.0 / (base ** (np.arange(0, num_bands) / max(1, num_bands)))

    inv_phi = _inv_freqs(dim_phi)
    inv_kap = _inv_freqs(dim_kap)
    inv_h = _inv_freqs(dim_h)
    inv_w = _inv_freqs(dim_w)

    return inv_phi, inv_kap, inv_h, inv_w

def compute_4d_continuous_rope(coords: np.ndarray, head_dim: int, base: float = 10000.0):
    """
    Computes continuous cos and sin rotary embeddings for coordinates:
    coords: (..., 4) containing [Phi, kappa, h, w]
    """
    inv_phi, inv_kap, inv_h, inv_w = get_4d_continuous_frequencies(head_dim, base)

    phi_coord = coords[..., 0:1]
    kap_coord = coords[..., 1:2]
    h_coord = coords[..., 2:3]
    w_coord = coords[..., 3:4]

    angles_phi = phi_coord * inv_phi
    angles_kap = kap_coord * inv_kap
    angles_h = h_coord * inv_h
    angles_w = w_coord * inv_w

    angles = np.concatenate([angles_phi, angles_kap, angles_h, angles_w], axis=-1)
    angles_full = np.concatenate([angles, angles], axis=-1)

    return np.cos(angles_full), np.sin(angles_full)

def build_prismerase_token_coordinates(mask_binary: np.ndarray, patch_size: int = 16):
    """
    Constructs 4D continuous coordinates for PrismErase condition & image tokens.
    """
    H, W = mask_binary.shape
    nh, nw = H // patch_size, W // patch_size

    phi_grid = compute_continuous_distance_field(mask_binary, patch_size)
    kappa_grid = compute_boundary_curvature_field(mask_binary, patch_size)

    grid_h, grid_w = np.meshgrid(np.arange(nh), np.arange(nw), indexing='ij')

    # Condition branch: continuous (Phi, kappa, h, w)
    coords_cond = np.stack([phi_grid, kappa_grid, grid_h, grid_w], axis=-1).reshape(-1, 4)

    # Noisy target branch: proxy coordinate (2.0, 0.0, h, w)
    proxy_phi = np.full_like(phi_grid, 2.0)
    proxy_kap = np.zeros_like(kappa_grid)
    coords_img = np.stack([proxy_phi, proxy_kap, grid_h, grid_w], axis=-1).reshape(-1, 4)

    return coords_cond, coords_img, (nh, nw)
