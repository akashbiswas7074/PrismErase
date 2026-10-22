"""
Foundation Inpainting Engine
============================
Integrates pretrained SOTA deep generative inpainting (LaMa ONNX)
as a high-fidelity foundational prior for PrismErase.
"""

import os
import numpy as np
from PIL import Image
from typing import Optional

_SESSION = None

def get_lama_session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'lama.onnx')
    if not os.path.exists(model_path):
        return None
        
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        _SESSION = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        return _SESSION
    except Exception as e:
        print(f"Warning: Failed to load LaMa ONNX session: {e}")
        return None

def foundation_inpaint(image: np.ndarray, mask: np.ndarray, target_res: int = 512) -> Optional[np.ndarray]:
    """
    Runs high-capacity foundation inpainting on image with mask.
    
    Args:
        image: uint8 RGB numpy array (H, W, 3)
        mask: uint8 or bool binary mask (H, W) where >0 represents the region to erase
        target_res: resolution for neural processing (default 512)
        
    Returns:
        uint8 RGB numpy array (H, W, 3) with erased region semantically reconstructed,
        or None if neural session unavailable.
    """
    session = get_lama_session()
    if session is None:
        return None

    orig_h, orig_w = image.shape[:2]
    
    # Preprocess
    img_pil = Image.fromarray(image).resize((target_res, target_res), Image.Resampling.BILINEAR)
    m_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((target_res, target_res), Image.Resampling.NEAREST)

    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    img_t = np.transpose(img_arr, (2, 0, 1))[None, ...] # (1, 3, target_res, target_res)

    mask_arr = (np.array(m_pil, dtype=np.float32) > 127.5).astype(np.float32)
    mask_t = mask_arr[None, None, ...] # (1, 1, target_res, target_res)

    # Run ONNX inference
    out = session.run(['clamp'], {'l_image_': img_t, 'l_mask_': mask_t})[0]
    out_img = np.clip(out[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    
    # Resize back to original resolution
    out_pil = Image.fromarray(out_img).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    return np.array(out_pil)
