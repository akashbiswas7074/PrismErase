"""
TraceFusion Post-Processing Algorithm
Reference: Section 3.3 and Appendix E.5 of "LensErase: A Foundation Model for Erase Anything" (arXiv:2610.18072)

TraceFusion restores unchanged regions from the source image to eliminate VAE reconstruction
artifacts, while preserving edited regions (including associated shadows and reflections)
through residual-based support detection, gamma-boosted dark region recovery, and linear RGB feathering.
"""

import numpy as np
from scipy import ndimage

def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB image in [0, 1] to Linear RGB."""
    img = np.clip(img_srgb, 0.0, 1.0)
    linear = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    return linear

def linear_to_srgb(img_linear: np.ndarray) -> np.ndarray:
    """Convert Linear RGB image in [0, 1] to sRGB."""
    img = np.clip(img_linear, 0.0, 1.0)
    srgb = np.where(img <= 0.0031308, 12.92 * img, 1.055 * (img ** (1.0 / 2.4)) - 0.055)
    return np.clip(srgb, 0.0, 1.0)

def disk_structure(radius: int) -> np.ndarray:
    """Generate a binary disk structuring element with the given radius."""
    r = max(1, int(round(radius)))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x * x + y * y <= r * r).astype(bool)

def compute_luma(img_255: np.ndarray) -> np.ndarray:
    """Compute Rec. 709 luma = 0.2126R + 0.7152G + 0.0722B."""
    return (
        0.2126 * img_255[..., 0] +
        0.7152 * img_255[..., 1] +
        0.0722 * img_255[..., 2]
    )

def tracefusion(source_img: np.ndarray, generated_img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Apply TraceFusion post-processing.
    Args:
      source_img: uint8 array (H, W, 3) or float array [0, 1]
      generated_img: uint8 array (H, W, 3) or float array [0, 1]
      mask: binary uint8 or bool array (H, W), 1 for masked/erased region, 0 for keep
    Returns:
      dict with:
        'fused_output': uint8 array (H, W, 3)
        'alpha': float32 array (H, W) in [0, 1]
        'support_mask': bool array (H, W)
        'primary_support': bool array (H, W)
        'residual_support': bool array (H, W)
        'gamma_support': bool array (H, W)
    """
    # Ensure float32 in [0, 255]
    if source_img.dtype == np.uint8:
        I_src = source_img.astype(np.float32)
    else:
        I_src = (np.clip(source_img, 0.0, 1.0) * 255.0).astype(np.float32)

    if generated_img.dtype == np.uint8:
        I_gen = generated_img.astype(np.float32)
    else:
        I_gen = (np.clip(generated_img, 0.0, 1.0) * 255.0).astype(np.float32)

    mask_bin = (mask > 0.5)
    H, W = I_src.shape[:2]
    s = min(H, W) / 512.0

    # 1. Dilate input selection mask by radius 12s
    rad_dil12 = max(1, int(round(12.0 * s)))
    M_dil = ndimage.binary_dilation(mask_bin, structure=disk_structure(rad_dil12))

    # 2. Gaussian-smooth the signed residual: R = G_sigma * (I_gen - I_src)
    sigma = max(0.5, 5.0 * s)
    residual_signed = I_gen - I_src
    R_smoothed = np.empty_like(residual_signed)
    for c in range(3):
        R_smoothed[..., c] = ndimage.gaussian_filter(residual_signed[..., c], sigma=sigma, mode='reflect')

    # Subtract per-channel median outside M_dil
    outside_mask = ~M_dil
    d_x = np.zeros((H, W), dtype=np.float32)
    if np.any(outside_mask):
        median_offset = np.median(R_smoothed[outside_mask], axis=0) # (3,)
        R_centered = R_smoothed - median_offset
    else:
        R_centered = R_smoothed
    d_x = np.max(np.abs(R_centered), axis=-1) # L-infinity norm across color channels

    # 3. Candidate detection at low (4) and high (10) thresholds
    low_thresh = 4.0
    high_thresh = 10.0
    low_cand = (d_x >= low_thresh)
    high_cand = (d_x >= high_thresh)

    # Morphological opening of low-threshold mask with disk of radius s
    rad_s = max(1, int(round(s)))
    low_opened = ndimage.binary_opening(low_cand, structure=disk_structure(rad_s))

    # 8-connected components filtering
    struct_8conn = ndimage.generate_binary_structure(2, 2)
    labeled, num_features = ndimage.label(low_opened, structure=struct_8conn)

    min_area = max(1, int(round(48.0 * (s ** 2))))
    min_high_area = max(1, int(round(8.0 * (s ** 2))))

    retained_mask = np.zeros((H, W), dtype=bool)
    if num_features > 0:
        slices = ndimage.find_objects(labeled)
        for i in range(1, num_features + 1):
            sl = slices[i - 1]
            if sl is None:
                continue
            comp = (labeled[sl] == i)
            area = np.sum(comp)
            if area < min_area:
                continue
            high_in_comp = np.sum(comp & high_cand[sl])
            if high_in_comp < min_high_area:
                continue

            # If component does not intersect M_dil, require >= 1% pixels above high threshold
            intersects_mdil = np.any(comp & M_dil[sl])
            if not intersects_mdil:
                if high_in_comp < (0.01 * area):
                    continue
            retained_mask[sl] |= comp

    # Close with disk 3s, dilate with disk 5s -> S_res
    rad_3s = max(1, int(round(3.0 * s)))
    rad_5s = max(1, int(round(5.0 * s)))
    S_res = ndimage.binary_closing(retained_mask, structure=disk_structure(rad_3s))
    S_res = ndimage.binary_dilation(S_res, structure=disk_structure(rad_5s))

    # Primary edit support S_0
    S_0 = M_dil | S_res

    # 4. Gamma branch for dark regions (gamma = 0.75)
    gamma = 0.75
    I_src_gamma = 255.0 * ((np.clip(I_src, 0.0, 255.0) / 255.0) ** gamma)
    I_gen_gamma = 255.0 * ((np.clip(I_gen, 0.0, 255.0) / 255.0) ** gamma)

    res_gamma = I_gen_gamma - I_src_gamma
    R_gamma_smooth = np.empty_like(res_gamma)
    for c in range(3):
        R_gamma_smooth[..., c] = ndimage.gaussian_filter(res_gamma[..., c], sigma=sigma, mode='reflect')

    if np.any(outside_mask):
        median_offset_g = np.median(R_gamma_smooth[outside_mask], axis=0)
        R_gamma_centered = R_gamma_smooth - median_offset_g
    else:
        R_gamma_centered = R_gamma_smooth
    d_x_gamma = np.max(np.abs(R_gamma_centered), axis=-1)

    low_cand_g = (d_x_gamma >= low_thresh)
    low_opened_g = ndimage.binary_opening(low_cand_g, structure=disk_structure(rad_s))
    # Only changes not already covered by S_0
    suppl_cand = low_opened_g & (~S_0)

    labeled_g, num_features_g = ndimage.label(suppl_cand, structure=struct_8conn)
    S_0_dilated_s = ndimage.binary_dilation(S_0, structure=disk_structure(rad_s))

    luma_src = compute_luma(I_src)
    luma_gen = compute_luma(I_gen)
    dark_mask = (luma_src <= 96.0) & (luma_gen <= 96.0)

    S_gamma = np.zeros((H, W), dtype=bool)
    if num_features_g > 0:
        slices_g = ndimage.find_objects(labeled_g)
        min_suppl_area = max(1, int(round(8.0 * (s ** 2))))
        for i in range(1, num_features_g + 1):
            sl = slices_g[i - 1]
            if sl is None:
                continue
            comp = (labeled_g[sl] == i)
            area = np.sum(comp)
            if area < min_suppl_area:
                continue
            # Must intersect dilation of S_0 by radius s
            if not np.any(comp & S_0_dilated_s[sl]):
                continue
            # At least half of pixels must have luma <= 96 in both images
            dark_pixels = np.sum(comp & dark_mask[sl])
            if dark_pixels < (0.5 * area):
                continue
            S_gamma[sl] |= comp

    # Final support S = S_0 U S_gamma
    S = S_0 | S_gamma

    # 5. Distance transform and boundary feathering
    if not np.any(S):
        alpha = np.zeros((H, W), dtype=np.float32)
    else:
        # Distance to S outside S, with pixel spacings 1/H and 1/W
        sampling = (1.0 / float(H), 1.0 / float(W))
        dist = ndimage.distance_transform_edt(~S, sampling=sampling)
        alpha = np.maximum(0.0, 1.0 - (dist / 0.1)).astype(np.float32)

    # 6. Linear RGB alpha blending: C^-1 [ alpha * C(I_gen) + (1 - alpha) * C(I_src) ]
    src_linear = srgb_to_linear(I_src / 255.0)
    gen_linear = srgb_to_linear(I_gen / 255.0)

    alpha_3d = alpha[..., np.newaxis]
    fused_linear = alpha_3d * gen_linear + (1.0 - alpha_3d) * src_linear
    fused_srgb = linear_to_srgb(fused_linear)
    fused_uint8 = np.clip(np.round(fused_srgb * 255.0), 0, 255).astype(np.uint8)

    return {
        'fused_output': fused_uint8,
        'alpha': alpha,
        'support_mask': S,
        'primary_support': S_0,
        'residual_support': S_res,
        'gamma_support': S_gamma
    }
