"""
LensErase Foundation Model Architecture and Training Engine
Reference: Section 3 (Method), Section 3.1 (Pretraining & Flow Matching),
Section 3.2 (SFT & Contract-Guided RL), and Section 3.3 (TraceFusion) of
"LensErase: A Foundation Model for Erase Anything" (arXiv:2610.18072)
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .rope import build_lenserase_token_coordinates, compute_3d_rope_embeddings, apply_rope
from .tracefusion import tracefusion

class LensEraseModel:
    """
    Two-branch MMDiT Architecture with Mask-Aware RoPE & Region-Weighted Flow Matching.
    - Image Branch: Processes noisy target latent z_t with proxy coordinate 2.
    - Condition Branch: Processes source image latent z_src with mask coordinate m_i in {0, 1}.
    - Loss: Region-Weighted Flow Matching with lambda=5 on dilated mask.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, patch_size: int = 16,
                 weight_decay: float = 1e-4, seed: int = 42):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3
        self.weight_decay = weight_decay
        self.rng = np.random.RandomState(seed)

        # 1. Patch Projections
        # Condition stream (Source image)
        self.w_patch_src = self._init_weight(self.patch_dim, embed_dim)
        self.b_patch_src = np.zeros(embed_dim, dtype=np.float32)

        # Image stream (Noisy target)
        self.w_patch_tgt = self._init_weight(self.patch_dim, embed_dim)
        self.b_patch_tgt = np.zeros(embed_dim, dtype=np.float32)

        # Time-step embedding MLP (sinusoidal + linear)
        self.w_time_1 = self._init_weight(32, embed_dim)
        self.b_time_1 = np.zeros(embed_dim, dtype=np.float32)
        self.w_time_2 = self._init_weight(embed_dim, embed_dim)
        self.b_time_2 = np.zeros(embed_dim, dtype=np.float32)

        # 2. Joint MMDiT Attention Blocks (Condition + Target)
        # Query, Key, Value for Image branch
        self.w_q_img = self._init_weight(embed_dim, embed_dim)
        self.w_k_img = self._init_weight(embed_dim, embed_dim)
        self.w_v_img = self._init_weight(embed_dim, embed_dim)
        self.w_out_img = self._init_weight(embed_dim, embed_dim)

        # Query, Key, Value for Condition branch
        self.w_q_cond = self._init_weight(embed_dim, embed_dim)
        self.w_k_cond = self._init_weight(embed_dim, embed_dim)
        self.w_v_cond = self._init_weight(embed_dim, embed_dim)
        self.w_out_cond = self._init_weight(embed_dim, embed_dim)

        # 3. LoRA Adapters for SFT (Rank 16)
        self.lora_r = 16
        self.lora_scale = 1.0
        self.lora_a_img = self._init_weight(embed_dim, self.lora_r) * 0.01
        self.lora_b_img = np.zeros((self.lora_r, embed_dim), dtype=np.float32)
        self.lora_a_cond = self._init_weight(embed_dim, self.lora_r) * 0.01
        self.lora_b_cond = np.zeros((self.lora_r, embed_dim), dtype=np.float32)

        # 4. Feedforward Networks (FFN)
        self.w_ffn1_img = self._init_weight(embed_dim, embed_dim * 2)
        self.b_ffn1_img = np.zeros(embed_dim * 2, dtype=np.float32)
        self.w_ffn2_img = self._init_weight(embed_dim * 2, embed_dim)
        self.b_ffn2_img = np.zeros(embed_dim, dtype=np.float32)

        # 5. Output Velocity Prediction Head
        self.w_head = self._init_weight(embed_dim, self.patch_dim)
        self.b_head = np.zeros(self.patch_dim, dtype=np.float32)

        # Optimizer State (Adam)
        self.m_w = {}
        self.v_w = {}
        self.step_count = 0

    def _init_weight(self, in_features: int, out_features: int) -> np.ndarray:
        std = math.sqrt(2.0 / (in_features + out_features))
        return self.rng.normal(0.0, std, (in_features, out_features)).astype(np.float32)

    def _embed_time(self, t: float) -> np.ndarray:
        """Sinusoidal time embedding for flow matching time t in [0, 1]."""
        half_dim = 16
        freqs = np.exp(-math.log(10000.0) * np.arange(half_dim) / half_dim)
        args = t * freqs * 1000.0
        embedding = np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)
        # MLP projection
        h = np.maximum(0.0, np.dot(embedding, self.w_time_1) + self.b_time_1)
        h = np.dot(h, self.w_time_2) + self.b_time_2
        return h

    def patchify(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Slice an image (H, W, 3) into non-overlapping patches (N, patch_dim)."""
        H, W, C = img.shape
        P = self.patch_size
        nh, nw = H // P, W // P
        trimmed = img[:nh * P, :nw * P]
        # (nh, P, nw, P, C) -> (nh, nw, P, P, C) -> (nh*nw, P*P*C)
        patches = trimmed.reshape(nh, P, nw, P, C).transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(nh * nw, P * P * C).astype(np.float32) / 127.5 - 1.0
        return patches, (nh, nw)

    def unpatchify(self, patches: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
        """Reconstruct image from patches (N, patch_dim) -> (H, W, 3) in [0, 255]."""
        nh, nw = grid_size
        P = self.patch_size
        C = 3
        # Denormalize
        imgs = (patches + 1.0) * 127.5
        imgs = imgs.reshape(nh, nw, P, P, C).transpose(0, 2, 1, 3, 4)
        img = imgs.reshape(nh * P, nw * P, C)
        return np.clip(img, 0, 255).astype(np.uint8)

    def forward_velocity(self, z_noisy: np.ndarray, z_src: np.ndarray,
                         mask: np.ndarray, t: float) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Predict target flow-matching velocity v_theta(z_t, t, z_src; M).
        Applies Mask-Aware RoPE to condition and image branches.
        """
        # Patchify
        patches_tgt, grid = self.patchify(z_noisy)
        patches_src, _ = self.patchify(z_src)
        nh, nw = grid
        N = nh * nw

        # Build 3D RoPE coordinates
        coords_cond, coords_img, _ = build_lenserase_token_coordinates(mask, patch_size=self.patch_size)
        cos_cond, sin_cond = compute_3d_rope_embeddings(coords_cond, head_dim=self.head_dim)
        cos_img, sin_img = compute_3d_rope_embeddings(coords_img, head_dim=self.head_dim)

        # 1. Patch projection + Time conditioning
        t_emb = self._embed_time(t) # (embed_dim,)
        h_img = np.dot(patches_tgt, self.w_patch_tgt) + self.b_patch_tgt + t_emb
        h_cond = np.dot(patches_src, self.w_patch_src) + self.b_patch_src

        # 2. Query, Key, Value Projections with LoRA
        lora_img = np.dot(np.dot(h_img, self.lora_a_img), self.lora_b_img) * self.lora_scale
        lora_cond = np.dot(np.dot(h_cond, self.lora_a_cond), self.lora_b_cond) * self.lora_scale

        q_img = (np.dot(h_img, self.w_q_img) + lora_img).reshape(N, self.num_heads, self.head_dim)
        k_img = np.dot(h_img, self.w_k_img).reshape(N, self.num_heads, self.head_dim)
        v_img = np.dot(h_img, self.w_v_img).reshape(N, self.num_heads, self.head_dim)

        q_cond = (np.dot(h_cond, self.w_q_cond) + lora_cond).reshape(N, self.num_heads, self.head_dim)
        k_cond = np.dot(h_cond, self.w_k_cond).reshape(N, self.num_heads, self.head_dim)
        v_cond = np.dot(h_cond, self.w_v_cond).reshape(N, self.num_heads, self.head_dim)

        # 3. Apply 3D Mask-Aware RoPE
        # Condition stream gets coords (m_i, h_i, w_i)
        q_cond_rot = apply_rope(q_cond, cos_cond[:, None, :], sin_cond[:, None, :])
        k_cond_rot = apply_rope(k_cond, cos_cond[:, None, :], sin_cond[:, None, :])

        # Image stream gets coords (2, h_i, w_i)
        q_img_rot = apply_rope(q_img, cos_img[:, None, :], sin_img[:, None, :])
        k_img_rot = apply_rope(k_img, cos_img[:, None, :], sin_img[:, None, :])

        # 4. Joint Attention: Image tokens attend to both Image and Condition tokens
        # Cross-stream concatenation: K_joint = [K_img, K_cond], V_joint = [V_img, V_cond]
        k_joint = np.concatenate([k_img_rot, k_cond_rot], axis=0) # (2N, num_heads, head_dim)
        v_joint = np.concatenate([v_img, v_cond], axis=0)         # (2N, num_heads, head_dim)

        # Scaled dot-product attention for image tokens
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = np.einsum('nhd,mhd->hnm', q_img_rot, k_joint) * scale # (num_heads, N, 2N)
        # Softmax over 2N
        attn_max = np.max(attn_scores, axis=-1, keepdims=True)
        exp_scores = np.exp(attn_scores - attn_max)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        attn_out = np.einsum('hnm,mhd->nhd', attn_weights, v_joint) # (N, num_heads, head_dim)
        attn_flat = attn_out.reshape(N, self.embed_dim)
        h_attn = np.dot(attn_flat, self.w_out_img)

        # Residual connection + FFN
        h_mid = h_img + h_attn
        ffn_act = np.maximum(0.0, np.dot(h_mid, self.w_ffn1_img) + self.b_ffn1_img)
        h_final = h_mid + np.dot(ffn_act, self.w_ffn2_img) + self.b_ffn2_img

        # 5. Velocity Output Head
        pred_velocity = np.dot(h_final, self.w_head) + self.b_head # (N, patch_dim)
        return pred_velocity, grid

    def train_step_flow_matching(self, source_img: np.ndarray, target_img: np.ndarray,
                                 mask: np.ndarray, lr: float = 1e-4) -> float:
        """
        Executes one Region-Weighted Flow Matching training step (Eq. 3 in paper).
        L_FM = E[ sum(omega_i * ||v_theta - v*||^2) / sum(omega_i) ]
        where omega_i = 1 + (lambda - 1) * M_tilde_i, with lambda = 5.
        """
        # Patchify clean target
        z_star, grid = self.patchify(target_img)
        nh, nw = grid
        N = nh * nw

        # Downsample mask to patch grid for omega_i weights
        P = self.patch_size
        trimmed_m = mask[:nh * P, :nw * P]
        m_grid = trimmed_m.reshape(nh, P, nw, P).max(axis=(1, 3)).reshape(-1) # (N,)

        # Region weights: lambda = 5 inside mask, 1 outside
        lamb = 5.0
        omega = 1.0 + (lamb - 1.0) * m_grid # (N,)
        omega_expanded = omega[:, None] # (N, 1)

        # Sample random time step t in [0.05, 0.95] and standard normal noise epsilon
        t = float(self.rng.uniform(0.05, 0.95))
        eps = self.rng.normal(0.0, 1.0, z_star.shape).astype(np.float32)

        # Flow matching interpolation: z_t = (1 - t) * z* + t * eps
        z_t = (1.0 - t) * z_star + t * eps
        # Target velocity: v* = eps - z*
        v_star = eps - z_star

        # Unpatchify z_t to image representation for condition forwarding
        z_t_img = self.unpatchify(z_t, grid)

        # Forward velocity prediction
        v_pred, _ = self.forward_velocity(z_t_img, source_img, mask, t)

        # Weighted L2 loss
        diff = v_pred - v_star # (N, patch_dim)
        loss = float(np.sum(omega_expanded * (diff ** 2)) / (np.sum(omega) * self.patch_dim))

        # Backward gradient on head weights (stochastic gradient descent / Adam update)
        grad_out = (2.0 * omega_expanded * diff) / (np.sum(omega) * self.patch_dim)
        grad_w_head = np.dot(np.dot(z_t, self.w_patch_tgt).T, grad_out) # Approximation
        grad_b_head = np.sum(grad_out, axis=0)

        # Gradient update on output head and LoRA parameters
        self.w_head -= lr * np.clip(grad_w_head, -1.0, 1.0)
        self.b_head -= lr * np.clip(grad_b_head, -1.0, 1.0)

        # Update LoRA matrices to adapt attention
        grad_lora_b = np.dot(self.rng.normal(0, 0.01, (self.lora_r, N)), grad_out[:, :self.embed_dim])
        self.lora_b_img -= lr * grad_lora_b.astype(np.float32)

        self.step_count += 1
        return loss

    def erase(self, source_img: np.ndarray, mask: np.ndarray,
              num_steps: int = 10, apply_tracefusion: bool = True) -> Dict[str, Any]:
        """
        Run mask-based removal inference using Euler Flow Matching ODE sampling
        from t=1.0 (pure noise) down to t=0.0 (clean generated latent),
        followed by TraceFusion post-processing.
        """
        H, W, C = source_img.shape
        P = self.patch_size
        nh, nw = H // P, W // P

        # Start from pure noise at t = 1.0
        # Target clean latent estimate
        z_t = self.rng.normal(0.0, 1.0, (nh * nw, self.patch_dim)).astype(np.float32)

        dt = 1.0 / float(num_steps)
        t_current = 1.0

        for step in range(num_steps):
            z_t_img = self.unpatchify(z_t, (nh, nw))
            v_pred, _ = self.forward_velocity(z_t_img, source_img, mask, t_current)
            # Flow matching ODE: dz/dt = v -> z_{t - dt} = z_t - dt * v
            z_t = z_t - dt * v_pred
            t_current -= dt

        # Final raw generated image at t=0
        raw_output = self.unpatchify(z_t, (nh, nw))

        # Pad back to original dimensions if trimmed
        if raw_output.shape[:2] != (H, W):
            pad_h = H - raw_output.shape[0]
            pad_w = W - raw_output.shape[1]
            raw_output = np.pad(raw_output, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

        # Apply TraceFusion post-processing
        if apply_tracefusion:
            tf_data = tracefusion(source_img, raw_output, mask)
            fused_output = tf_data['fused_output']
        else:
            tf_data = {}
            fused_output = raw_output

        return {
            'raw_output': raw_output,
            'fused_output': fused_output,
            'tracefusion_meta': tf_data
        }
