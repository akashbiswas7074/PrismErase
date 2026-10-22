"""
Evaluation Metrics for LensErase
Reference: Section 2.3, Section 4, and Appendix C.2 of "LensErase: A Foundation Model for Erase Anything" (arXiv:2610.18072)

Covers:
1. Reference-Based Objective Metrics:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - LPIPS (Learned Perceptual Image Patch Similarity)
   - Residual Preservation Error
2. Reference-Free VLM Evaluation Protocol (LensErase-Bench):
   - Removal (0-4 / 0-100)
   - Inpainting (0-4 / 0-100)
   - Preservation (0-4 / 0-100)
   - Cleanliness (0-4 / 0-100)
   - Overall Score (weighted: 40% Removal + 20% Inpainting + 20% Preservation + 20% Cleanliness)
   - HiFi Removal Rate (% cases with Removal == 4 and Inpainting >= 3, Preservation >= 3, Cleanliness >= 3)
"""

import numpy as np
from scipy import ndimage

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute PSNR between two images (H, W, 3) in [0, 255]."""
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mse = np.mean((im1 - im2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0,
                   k1: float = 0.01, k2: float = 0.03, win_size: int = 11) -> float:
    """Compute Mean Structural Similarity Index (SSIM) across channels."""
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Gaussian window
    sigma = 1.5
    radius = win_size // 2
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    ssim_channels = []
    channels = im1.shape[2] if im1.ndim == 3 else 1
    for c in range(channels):
        i1 = im1[..., c] if im1.ndim == 3 else im1
        i2 = im2[..., c] if im2.ndim == 3 else im2

        mu1 = ndimage.convolve(i1, kernel, mode='reflect')
        mu2 = ndimage.convolve(i2, kernel, mode='reflect')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = ndimage.convolve(i1 * i1, kernel, mode='reflect') - mu1_sq
        sigma2_sq = ndimage.convolve(i2 * i2, kernel, mode='reflect') - mu2_sq
        sigma12 = ndimage.convolve(i1 * i2, kernel, mode='reflect') - mu1_mu2

        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = numerator / (denominator + 1e-12)
        ssim_channels.append(np.mean(ssim_map))

    return float(np.mean(ssim_channels))

def calculate_preservation_score(source_img: np.ndarray, output_img: np.ndarray,
                                 mask: np.ndarray) -> dict:
    """
    Calculate fidelity metrics strictly outside the dilated edit region.
    High preservation means unchanged background has near-zero deviation.
    """
    H, W = source_img.shape[:2]
    s = min(H, W) / 512.0
    r = max(1, int(round(12.0 * s)))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    disk = (x*x + y*y <= r*r)
    dilated_mask = ndimage.binary_dilation(mask > 0.5, structure=disk)

    bg_mask = ~dilated_mask
    if not np.any(bg_mask):
        return {'bg_psnr': 0.0, 'bg_mae': 0.0, 'bg_rmse': 0.0}

    src_bg = source_img[bg_mask].astype(np.float64)
    out_bg = output_img[bg_mask].astype(np.float64)

    diff = np.abs(src_bg - out_bg)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bg_psnr = float(20.0 * np.log10(255.0 / (rmse + 1e-8)))

    return {
        'bg_psnr': bg_psnr,
        'bg_mae': mae,
        'bg_rmse': rmse
    }

def evaluate_lenserase_bench_rubric(removal_raw: float, inpainting_raw: float,
                                    preservation_raw: float, cleanliness_raw: float) -> dict:
    """
    Convert raw 0-4 rubric ratings into official LensErase-Bench reporting scores (0-100)
    and compute Overall and HiFi Removal status according to Section 2.3.
    """
    # Scale from 0-4 to 0-100
    rem_100 = (removal_raw / 4.0) * 100.0
    inp_100 = (inpainting_raw / 4.0) * 100.0
    pre_100 = (preservation_raw / 4.0) * 100.0
    cln_100 = (cleanliness_raw / 4.0) * 100.0

    # Overall = 40% Removal + 20% Inpainting + 20% Preservation + 20% Cleanliness
    overall = (0.40 * rem_100) + (0.20 * inp_100) + (0.20 * pre_100) + (0.20 * cln_100)

    # HiFi Removal: Removal == 4 and Inpainting >= 3 and Preservation >= 3 and Cleanliness >= 3
    is_hifi = (
        (removal_raw >= 4.0 - 1e-5) and
        (inpainting_raw >= 3.0) and
        (preservation_raw >= 3.0) and
        (cleanliness_raw >= 3.0)
    )

    return {
        'removal': rem_100,
        'inpainting': inp_100,
        'preservation': pre_100,
        'cleanliness': cln_100,
        'overall': overall,
        'is_hifi_removal': is_hifi
    }
