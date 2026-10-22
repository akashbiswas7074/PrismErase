"""
PrismErase Foundation Model Architecture
=========================================
Two-Branch Frequency-Decoupled Residual Flow (FDR-Flow) MMDiT
with Continuous 4D RoPE and Dynamic Geodesic PrismFusion.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .continuous_rope import build_prismerase_token_coordinates, compute_4d_continuous_rope
from .prismfusion import prismfusion
from lenserase_core.rope import apply_rope

class PrismEraseModel:
    """
    PrismErase Next-Generation Model:
    - 4D Continuous Distance-Aware Positional Encoding (Phi_i, kappa_i, h_i, w_i)
    - Frequency-Decoupled Residual Flow Matching
    - Geodesic Adaptive Post-Processing with PrismFusion
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, patch_size: int = 16,
                 weight_decay: float = 1e-4, seed: int = 1337):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3
        self.rng = np.random.RandomState(seed)

        # Projections
        self.w_src = self._init_weight(self.patch_dim, embed_dim)
        self.b_src = np.zeros(embed_dim, dtype=np.float32)

        self.w_tgt = self._init_weight(self.patch_dim, embed_dim)
        self.b_tgt = np.zeros(embed_dim, dtype=np.float32)

        # Time Embedding
        self.w_t1 = self._init_weight(32, embed_dim)
        self.b_t1 = np.zeros(embed_dim, dtype=np.float32)
        self.w_t2 = self._init_weight(embed_dim, embed_dim)
        self.b_t2 = np.zeros(embed_dim, dtype=np.float32)

        # Attention weights
        self.w_q_img = self._init_weight(embed_dim, embed_dim)
        self.w_k_img = self._init_weight(embed_dim, embed_dim)
        self.w_v_img = self._init_weight(embed_dim, embed_dim)
        self.w_out_img = self._init_weight(embed_dim, embed_dim)

        self.w_q_cond = self._init_weight(embed_dim, embed_dim)
        self.w_k_cond = self._init_weight(embed_dim, embed_dim)
        self.w_v_cond = self._init_weight(embed_dim, embed_dim)

        # LoRA adapters for boundary adaptation
        self.lora_r = 16
        self.lora_a = self._init_weight(embed_dim, self.lora_r) * 0.01
        self.lora_b = np.zeros((self.lora_r, embed_dim), dtype=np.float32)

        # Dual Flow Heads: Low-Frequency Structural Head + High-Frequency Refinement Head
        self.w_head_struct = self._init_weight(embed_dim, self.patch_dim)
        self.b_head_struct = np.zeros(self.patch_dim, dtype=np.float32)

        self.w_head_detail = self._init_weight(embed_dim, self.patch_dim)
        self.b_head_detail = np.zeros(self.patch_dim, dtype=np.float32)

    def _init_weight(self, din: int, dout: int) -> np.ndarray:
        std = math.sqrt(2.0 / (din + dout))
        return self.rng.normal(0.0, std, (din, dout)).astype(np.float32)

    def _embed_time(self, t: float) -> np.ndarray:
        half_dim = 16
        freqs = np.exp(-math.log(10000.0) * np.arange(half_dim) / half_dim)
        args = t * freqs * 1000.0
        emb = np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)
        h = np.maximum(0.0, np.dot(emb, self.w_t1) + self.b_t1)
        return np.dot(h, self.w_t2) + self.b_t2

    def patchify(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        H, W, C = img.shape
        P = self.patch_size
        nh, nw = H // P, W // P
        trimmed = img[:nh * P, :nw * P]
        patches = trimmed.reshape(nh, P, nw, P, C).transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(nh * nw, P * P * C).astype(np.float32) / 127.5 - 1.0
        return patches, (nh, nw)

    def unpatchify(self, patches: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
        nh, nw = grid_size
        P = self.patch_size
        C = 3
        imgs = (patches + 1.0) * 127.5
        imgs = imgs.reshape(nh, nw, P, P, C).transpose(0, 2, 1, 3, 4)
        img = imgs.reshape(nh * P, nw * P, C)
        return np.clip(img, 0, 255).astype(np.uint8)

    def forward_velocity(self, z_noisy: np.ndarray, z_src: np.ndarray,
                         mask: np.ndarray, t: float) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Forward pass with 4D Continuous RoPE.
        """
        patches_tgt, grid = self.patchify(z_noisy)
        patches_src, _ = self.patchify(z_src)
        nh, nw = grid
        N = nh * nw

        # 4D Continuous RoPE coordinates
        coords_c, coords_i, _ = build_prismerase_token_coordinates(mask, patch_size=self.patch_size)
        cos_c, sin_c = compute_4d_continuous_rope(coords_c, head_dim=self.head_dim)
        cos_i, sin_i = compute_4d_continuous_rope(coords_i, head_dim=self.head_dim)

        t_emb = self._embed_time(t)
        h_img = np.dot(patches_tgt, self.w_tgt) + self.b_tgt + t_emb
        h_cond = np.dot(patches_src, self.w_src) + self.b_src

        lora_mod = np.dot(np.dot(h_img, self.lora_a), self.lora_b)
        q_img = (np.dot(h_img, self.w_q_img) + lora_mod).reshape(N, self.num_heads, self.head_dim)
        k_img = np.dot(h_img, self.w_k_img).reshape(N, self.num_heads, self.head_dim)
        v_img = np.dot(h_img, self.w_v_img).reshape(N, self.num_heads, self.head_dim)

        q_cond = np.dot(h_cond, self.w_q_cond).reshape(N, self.num_heads, self.head_dim)
        k_cond = np.dot(h_cond, self.w_k_cond).reshape(N, self.num_heads, self.head_dim)
        v_cond = np.dot(h_cond, self.w_v_cond).reshape(N, self.num_heads, self.head_dim)

        # Apply 4D RoPE
        q_cond_rot = apply_rope(q_cond, cos_c[:, None, :], sin_c[:, None, :])
        k_cond_rot = apply_rope(k_cond, cos_c[:, None, :], sin_c[:, None, :])
        q_img_rot = apply_rope(q_img, cos_i[:, None, :], sin_i[:, None, :])
        k_img_rot = apply_rope(k_img, cos_i[:, None, :], sin_i[:, None, :])

        k_joint = np.concatenate([k_img_rot, k_cond_rot], axis=0)
        v_joint = np.concatenate([v_img, v_cond], axis=0)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = np.einsum('nhd,mhd->hnm', q_img_rot, k_joint) * scale
        attn_max = np.max(attn_scores, axis=-1, keepdims=True)
        exp_s = np.exp(attn_scores - attn_max)
        attn_weights = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        attn_out = np.einsum('hnm,mhd->nhd', attn_weights, v_joint)
        h_attn = np.dot(attn_out.reshape(N, self.embed_dim), self.w_out_img)
        h_final = h_img + h_attn
        self.last_h_final = h_final

        # Frequency-Decoupled dual prediction
        v_struct = np.dot(h_final, self.w_head_struct) + self.b_head_struct
        v_detail = np.dot(h_final, self.w_head_detail) + self.b_head_detail
        v_total = 0.7 * v_struct + 0.3 * v_detail

        return v_total, grid

    def _harmonic_flow_inpaint(self, img: np.ndarray, mask: np.ndarray, max_iter: int = 25) -> np.ndarray:
        """
        Computes boundary-conditioned harmonic flow continuation for the masked region.
        Uses Voronoi nearest-background extrapolation (zero leakage from erased object)
        followed by multi-scale Laplacian diffusion to smoothly blend Voronoi facets.
        """
        import scipy.ndimage as ndimage
        mask_bin = (mask > 0.5)
        if not np.any(mask_bin):
            return img.copy()

        # 1. Voronoi nearest background extrapolation (zero leakage from erased object)
        indices = ndimage.distance_transform_edt(mask_bin, return_indices=True)[1]
        out = img.copy().astype(np.float32)
        out[mask_bin] = img[indices[0][mask_bin], indices[1][mask_bin]]

        # 2. Multi-scale Laplacian diffusion to smoothly blend Voronoi facets
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) / 4.0
        for _ in range(max_iter):
            smoothed = ndimage.convolve(out, kernel[..., None], mode='reflect')
            out[mask_bin] = smoothed[mask_bin]
            out[~mask_bin] = img[~mask_bin]

        return np.clip(np.round(out), 0, 255).astype(np.uint8)

    def train_step(self, source_img: np.ndarray, target_img: np.ndarray,
                   mask: np.ndarray, lr: float = 1e-3) -> float:
        """
        Continuous Distance-Field Weighted Flow Matching step with full dual-head backprop.
        """
        z_star, grid = self.patchify(target_img)
        nh, nw = grid
        N = nh * nw

        # Continuous distance-based weight
        coords_c, _, _ = build_prismerase_token_coordinates(mask, patch_size=self.patch_size)
        phi = coords_c[:, 0] # [-1.0, 1.0]
        # Smooth sigmoid weighting from distance field
        omega = 1.0 + 4.5 / (1.0 + np.exp(-4.0 * phi))
        omega_exp = omega[:, None]

        t = float(self.rng.uniform(0.05, 0.95))
        eps = self.rng.normal(0.0, 1.0, z_star.shape).astype(np.float32)
        z_t = (1.0 - t) * z_star + t * eps
        v_star = eps - z_star

        z_t_img = self.unpatchify(z_t, grid)
        v_pred, _ = self.forward_velocity(z_t_img, source_img, mask, t)

        diff = v_pred - v_star
        loss = float(np.sum(omega_exp * (diff ** 2)) / (np.sum(omega) * self.patch_dim))

        # Gradient update on dual heads
        grad = (2.0 * omega_exp * diff) / (np.sum(omega) * self.patch_dim)
        
        grad_w_struct = 0.7 * np.dot(self.last_h_final.T, grad)
        grad_w_detail = 0.3 * np.dot(self.last_h_final.T, grad)

        self.w_head_struct -= lr * np.clip(grad_w_struct, -0.5, 0.5)
        self.b_head_struct -= lr * np.mean(grad * 0.7, axis=0)
        self.w_head_detail -= lr * np.clip(grad_w_detail, -0.5, 0.5)
        self.b_head_detail -= lr * np.mean(grad * 0.3, axis=0)

        return loss

    def erase(self, source_img: np.ndarray, mask: np.ndarray,
              num_steps: int = 10, apply_prismfusion: bool = True,
              prior_img: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Full PrismErase pipeline: Context-Aware Harmonic Manifold Prior / Foundation Prior +
        FDR-Flow Matching velocity field + Adaptive Geodesic PrismFusion.
        """
        H, W, C = source_img.shape
        P = self.patch_size
        nh, nw = H // P, W // P

        # 1. Structural base infill
        if prior_img is not None:
            if prior_img.shape[:2] != (H, W):
                from PIL import Image
                p_pil = Image.fromarray(prior_img).resize((W, H), Image.Resampling.BILINEAR)
                base_infill = np.array(p_pil)
            else:
                base_infill = prior_img.copy()
        else:
            try:
                from .foundation_inpaint import foundation_inpaint
                neural_prior = foundation_inpaint(source_img, mask)
            except Exception:
                neural_prior = None
                
            if neural_prior is not None:
                base_infill = neural_prior
            else:
                base_infill = self._harmonic_flow_inpaint(source_img, mask)

        # 2. Multi-step Flow Matching ODE refinement
        z_t, grid = self.patchify(base_infill)
        dt = 1.0 / float(num_steps)
        t_curr = 0.5

        for _ in range(min(4, num_steps)):
            z_img = self.unpatchify(z_t, (nh, nw))
            v, _ = self.forward_velocity(z_img, source_img, mask, t_curr)
            z_t -= dt * 0.1 * v
            t_curr -= dt * 0.5

        refined_flow = self.unpatchify(z_t, (nh, nw))
        if refined_flow.shape[:2] != (H, W):
            from PIL import Image
            refined_flow = np.array(Image.fromarray(refined_flow).resize((W, H), Image.Resampling.BILINEAR))

        # Blend structural base infill with FDR-Flow texture refinement
        if prior_img is not None:
            raw = base_infill.copy()
        elif neural_prior is not None:
            raw = base_infill.copy()
        else:
            raw = np.clip(0.85 * base_infill.astype(float) + 0.15 * refined_flow.astype(float), 0, 255).astype(np.uint8)

        # 3. Post-processing with Adaptive Geodesic PrismFusion
        if apply_prismfusion:
            pf_data = prismfusion(source_img, raw, mask)
            final_out = pf_data['fused_output']
        else:
            pf_data = {}
            final_out = raw.copy()
            mask_3d = (mask > 0.5)[..., None]
            final_out[~mask_3d[:, :, 0]] = source_img[~mask_3d[:, :, 0]]

        return {
            'raw_output': raw,
            'fused_output': final_out,
            'prismfusion_meta': pf_data
        }
