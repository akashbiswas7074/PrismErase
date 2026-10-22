"""
PrismErase: Continuous-Field Distance-Aware Foundation Model for Object Removal
(Better than LensErase)
"""

from .continuous_rope import (
    compute_continuous_distance_field,
    compute_boundary_curvature_field,
    get_4d_continuous_frequencies,
    compute_4d_continuous_rope,
    build_prismerase_token_coordinates
)
from .prismfusion import prismfusion
from .model import PrismEraseModel

__all__ = [
    "compute_continuous_distance_field",
    "compute_boundary_curvature_field",
    "get_4d_continuous_frequencies",
    "compute_4d_continuous_rope",
    "build_prismerase_token_coordinates",
    "prismfusion",
    "PrismEraseModel"
]
