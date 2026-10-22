"""
PrismErase Contract-Guided Erasure & Evaluation Engine
======================================================
Implements:
1. Erase Contract K = (T, E, O):
   - T: Removal Targets (primary entities selected to disappear)
   - E: Associated Visual Effects (shadows, reflections, contact marks)
   - O: Protected Objects (independent nearby objects to strictly preserve)
2. 5-Component Quality Evaluation:
   - Removal, Reconstruction, Preservation, Cleanliness, Outside Fidelity
3. Removal-Aware Reward Gating:
   - r = 0.5^{n_0} * b, penalizing incomplete removals exponentially.
4. HiFi Removal Rate (%) metric.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import ndimage

@dataclass
class EraseContract:
    """
    Formal Erase Contract specifying the edit scope for a given image-mask pair.
    """
    target_names: List[str] = field(default_factory=list)
    associated_effects: List[str] = field(default_factory=list)
    protected_objects: List[str] = field(default_factory=list)
    target_mask: Optional[np.ndarray] = None
    effects_mask: Optional[np.ndarray] = None
    protected_mask: Optional[np.ndarray] = None

def build_erase_contract(image: np.ndarray, mask: np.ndarray,
                         category_hint: str = "general") -> EraseContract:
    """
    Automatically grounds an Erase Contract K = (T, E, O) from an image and binary mask.
    Detects potential associated effects (cast shadows / penumbras) and protected nearby edges.
    """
    H, W = image.shape[:2]
    mask_bin = (mask > 0.5)

    # 1. Target Mask (T)
    T_mask = mask_bin.copy()

    # 2. Associated Effects Mask (E): Detect directional shadows adjacent to mask
    # Cast shadows typically have lower luminance and smooth gradients
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    struct = ndimage.generate_binary_structure(2, 2)
    dilated_mask = ndimage.binary_dilation(mask_bin, structure=struct, iterations=12)
    halo_zone = dilated_mask & (~mask_bin)

    # Identify dark shadow pixels in the halo zone
    local_mean = ndimage.uniform_filter(gray, size=15)
    shadow_candidates = halo_zone & (gray < (local_mean * 0.85)) & (gray < 90.0)
    E_mask = ndimage.binary_opening(shadow_candidates, structure=struct)

    # 3. Protected Objects Mask (O): High-contrast contours outside the dilated zone
    sobel_grad = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
    protected_zone = (~dilated_mask) & (sobel_grad > 35.0)
    O_mask = ndimage.binary_dilation(protected_zone, structure=struct, iterations=2)

    return EraseContract(
        target_names=[f"{category_hint}_target"],
        associated_effects=["cast_shadow", "contact_penumbra"] if np.any(E_mask) else [],
        protected_objects=["independent_background_structures"],
        target_mask=T_mask,
        effects_mask=E_mask,
        protected_mask=O_mask
    )

def evaluate_contract_metrics(source_img: np.ndarray,
                              erased_img: np.ndarray,
                              contract: EraseContract,
                              prior_img: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluates the 5 core dimensions:
    - Removal (0-100)
    - Reconstruction (0-100)
    - Preservation (0-100)
    - Cleanliness (0-100)
    - Outside Fidelity (PSNR dB -> normalized 0-100)
    Returns composite score and HiFi flag.
    """
    T = contract.target_mask
    E = contract.effects_mask
    O = contract.protected_mask
    
    H, W = source_img.shape[:2]
    total_pixels = float(H * W)

    # 1. Removal Score: Measures absence of source features in T and E
    diff = np.abs(erased_img.astype(float) - source_img.astype(float))
    diff_mag = np.max(diff, axis=-1)
    
    # In target region, significant change indicates removal
    t_change = np.mean(diff_mag[T] > 15.0) if np.any(T) else 1.0
    # Suppression of target color signature
    removal_score = float(np.clip(t_change * 100.0, 0.0, 100.0))

    # 2. Preservation Score: In protected region O, diff should be near 0
    if np.any(O):
        o_preservation = 1.0 - np.clip(np.mean(diff_mag[O]) / 30.0, 0.0, 1.0)
    else:
        outside_all = ~(T | E)
        o_preservation = 1.0 - np.clip(np.mean(diff_mag[outside_all]) / 25.0, 0.0, 1.0)
    preservation_score = float(np.clip(o_preservation * 100.0, 0.0, 100.0))

    # 3. Outside Fidelity: PSNR on confident background
    outside_mask = ~(T | (E if E is not None else np.zeros_like(T)))
    mse_outside = np.mean((erased_img[outside_mask].astype(float) - source_img[outside_mask].astype(float)) ** 2)
    if mse_outside < 1e-10:
        psnr_outside = 100.0
    else:
        psnr_val = 10.0 * np.log10((255.0 ** 2) / mse_outside)
        psnr_outside = float(np.clip(psnr_val, 0.0, 100.0))

    # 4. Reconstruction Score: Smoothness and structural coherence inside T
    grad_e = np.hypot(
        ndimage.sobel(erased_img.astype(float).mean(axis=-1), axis=0),
        ndimage.sobel(erased_img.astype(float).mean(axis=-1), axis=1)
    )
    # Penalize unnatural spike gradients inside infill
    inside_spikes = np.mean(grad_e[T] > 120.0) if np.any(T) else 0.0
    reconstruction_score = float(np.clip((1.0 - inside_spikes) * 94.0, 0.0, 100.0))

    # 5. Cleanliness Score: Absence of edge halos and residual artifacts
    boundary_zone = ndimage.binary_dilation(T, iterations=3) & (~T)
    boundary_halo = np.mean(diff_mag[boundary_zone] > 20.0) if np.any(boundary_zone) else 0.0
    cleanliness_score = float(np.clip((1.0 - boundary_halo) * 96.0, 0.0, 100.0))

    # Base reward aggregation: 40% Removal, 20% Inpainting, 20% Preservation, 20% Cleanliness
    overall_score = (
        0.40 * removal_score +
        0.20 * reconstruction_score +
        0.20 * preservation_score +
        0.20 * cleanliness_score
    )

    # Removal-Aware Reward Gating:
    # If removal < 70, incur exponential penalty
    n_zero = 1 if removal_score < 70.0 else 0
    gated_reward = (0.5 ** n_zero) * (overall_score / 100.0)

    # HiFi Removal criteria: Removal >= 85 and others >= 75
    is_hifi = (
        removal_score >= 85.0 and
        reconstruction_score >= 75.0 and
        preservation_score >= 75.0 and
        cleanliness_score >= 75.0
    )

    return {
        'removal': removal_score,
        'reconstruction': reconstruction_score,
        'preservation': preservation_score,
        'cleanliness': cleanliness_score,
        'outside_psnr': psnr_outside,
        'overall': overall_score,
        'gated_reward': float(gated_reward),
        'is_hifi': bool(is_hifi)
    }

class EraseContractEngine:
    """
    Engine wrapper for building contracts and evaluating removal rewards.
    """
    @staticmethod
    def build(image: np.ndarray, mask: np.ndarray, category_hint: str = "general") -> EraseContract:
        return build_erase_contract(image, mask, category_hint)

    @staticmethod
    def evaluate(source_img: np.ndarray, erased_img: np.ndarray,
                 contract: EraseContract, prior_img: Optional[np.ndarray] = None) -> Dict[str, float]:
        return evaluate_contract_metrics(source_img, erased_img, contract, prior_img)

