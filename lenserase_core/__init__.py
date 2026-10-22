"""
LensErase Core Package
"""

from .rope import (
    get_3d_rotary_frequencies,
    compute_3d_rope_embeddings,
    apply_rope,
    build_lenserase_token_coordinates
)
from .tracefusion import tracefusion, srgb_to_linear, linear_to_srgb
from .metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_preservation_score,
    evaluate_lenserase_bench_rubric
)
from .dataset import LensEraseDataset, create_mask_overlay
from .model import LensEraseModel

__all__ = [
    "get_3d_rotary_frequencies",
    "compute_3d_rope_embeddings",
    "apply_rope",
    "build_lenserase_token_coordinates",
    "tracefusion",
    "srgb_to_linear",
    "linear_to_srgb",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_preservation_score",
    "evaluate_lenserase_bench_rubric",
    "LensEraseDataset",
    "create_mask_overlay",
    "LensEraseModel"
]
