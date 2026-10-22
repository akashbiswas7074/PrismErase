"""
LensErase-100M Synthetic Pair Generator
Reference: Section 2.1 of "LensErase: A Foundation Model for Erase Anything" (arXiv:2610.18072)

Pipeline:
1. Object Taxonomy & Instruction Generation:
   Reversible counterfactual pairs: clean real image I_clean is target I_output,
   inserted object produces I_input.
2. Coarse-to-Fine Mask Generation:
   Threshold low-frequency DCT and visual feature differences between I_input and I_output.
3. Union of differences defines erase mask M.
"""

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

def generate_synthetic_erase_pair(base_img: np.ndarray, object_type: str = "text",
                                  box: tuple = (100, 100, 300, 200)) -> dict:
    """
    Simulates the Section 2.1 LensErase-100M counterfactual pair generation:
    - Real clean image = I_output (ground-truth target)
    - Synthetically added element (text/watermark/object) = I_input
    - Extracted change region = Erase Mask M
    """
    clean_target = base_img.copy()
    h, w, c = clean_target.shape

    # Create modified input image with target element added
    input_img = clean_target.copy()
    pil_input = Image.fromarray(input_img)
    draw = ImageDraw.Draw(pil_input)

    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

    if object_type == "text":
        text_str = "SAMPLE WATERMARK"
        # Draw translucent or solid text watermark
        draw.rectangle([x1, y1, x2, y2], fill=(220, 220, 220))
        draw.text((x1 + 10, y1 + 15), text_str, fill=(20, 20, 20))
        mask[y1:y2, x1:x2] = 1

    elif object_type == "watermark":
        # Diagonal translucent watermark pattern
        for offset in range(-h, w, 80):
            draw.line([(offset, 0), (offset + h, h)], fill=(240, 240, 255), width=12)
        # Difference-based mask
        diff = np.max(np.abs(np.array(pil_input) - clean_target), axis=-1)
        mask = (diff > 15).astype(np.uint8)

    else:
        # Generic object / geometric badge
        draw.ellipse([x1, y1, x2, y2], fill=(230, 80, 50))
        mask[y1:y2, x1:x2] = 1

    input_np = np.array(pil_input)

    # Coarse-to-fine mask expansion to include subtle border effects (as in SAM2 step in Section 2.1)
    mask_dilated = ndimage.binary_dilation(mask, structure=np.ones((5, 5))).astype(np.uint8)

    return {
        "input_image": input_np,
        "target_image": clean_target,
        "mask": mask_dilated,
        "object_type": object_type
    }
