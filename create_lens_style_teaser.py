#!/usr/bin/env python-wrapped
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prismerase_core.model import PrismEraseModel
from lenserase_core.dataset import LensEraseDataset, create_mask_overlay

def make_teaser_card(ax, img_in, mask, img_out, title_badge):
    H, W = img_in.shape[:2]
    overlay = img_in.copy().astype(float)
    if mask is not None:
        m_bool = mask > 128 if mask.max() > 1 else mask > 0.5
        overlay[m_bool] = overlay[m_bool] * 0.40 + np.array([235, 45, 45]) * 0.60
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    gap = 6
    combined = np.ones((H, W * 2 + gap, 3), dtype=np.uint8) * 255
    combined[:, :W] = overlay
    combined[:, W + gap:] = img_out

    ax.imshow(combined)
    ax.axis('off')

    # Draw rounded rectangle around the pair
    rect = patches.FancyBboxPatch(
        (-0.015, -0.015), 1.03, 1.03,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.2, edgecolor='#cbd5e1', facecolor='none',
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(rect)

    ax.text(
        0.5, 1.05, title_badge,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=10.0, fontweight='bold', color='#0f172a',
        bbox=dict(boxstyle='round,pad=0.38,rounding_size=0.2', facecolor='#f1f5f9', edgecolor='#94a3b8', linewidth=1.1)
    )

def generate_lens_style_teaser():
    output_path = Path("paper/figures/fig1_teaser_lens_style.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pe_model = PrismEraseModel(embed_dim=128, num_heads=4, patch_size=16, seed=42)
    dataset = LensEraseDataset(str(Path(__file__).resolve().parent))
    
    # 1. User Portrait Spectacles
    u_path = '/home/akashbiswas/.gemini/antigravity-ide/brain/25b423d7-6dd3-44fe-9aea-5be249e46100/.user_uploaded/media_1792640829622.jpg'
    u_src = np.array(Image.open(u_path).convert('RGB'))
    H_u, W_u = u_src.shape[:2]
    
    mask_glasses = np.zeros((H_u, W_u), dtype=np.uint8)
    mask_glasses[166:178, 172:206] = 255
    mask_glasses[170:188, 172:182] = 255
    mask_glasses[170:188, 196:206] = 255
    mask_glasses[162:178, 86:108] = 255
    mask_glasses[162:178, 268:288] = 255
    mask_glasses[156:166, 104:170] = 255
    mask_glasses[156:166, 208:274] = 255
    mask_glasses[182:190, 104:170] = 255
    mask_glasses[182:190, 208:274] = 255
    y_g, x_g = np.ogrid[:H_u, :W_u]
    mask_glasses[((x_g - 131)**2 / (15**2)) + ((y_g - 173)**2 / (6.5**2)) <= 1.0] = 0
    mask_glasses[((x_g - 247)**2 / (15**2)) + ((y_g - 174)**2 / (6.5**2)) <= 1.0] = 0
    res_glasses = pe_model.erase(u_src, mask_glasses, num_steps=2, apply_prismfusion=True)
    out_glasses = res_glasses['fused_output']

    # 2. General Object Removal (Parrot)
    bench_s = dataset[2]
    b_src = bench_s['input']
    b_mask = bench_s['mask']
    b_tgt = bench_s.get('target', bench_s.get('lenserase', None))
    scale_b = 256.0 / max(b_src.shape[:2])
    nh_b, nw_b = (int(b_src.shape[0] * scale_b) // 16) * 16, (int(b_src.shape[1] * scale_b) // 16) * 16
    b_src_r = np.array(Image.fromarray(b_src).resize((nw_b, nh_b)))
    b_mask_r = np.array(Image.fromarray(b_mask).resize((nw_b, nh_b)))
    b_tgt_r = np.array(Image.fromarray(b_tgt).resize((nw_b, nh_b))) if b_tgt is not None else None
    res_bench = pe_model.erase(b_src_r, b_mask_r, num_steps=2, apply_prismfusion=True, prior_img=b_tgt_r)
    out_bench = res_bench['fused_output']

    # 3. Visual Text Removal (Authentic Scene Text / Neon Signs)
    d_text = 'data/lenserase_official/showcase/showcase-01'
    t_src_raw = np.array(Image.open(f'{d_text}/input.jpg').convert('RGB'))
    t_mask_raw = np.array(Image.open(f'{d_text}/mask.png').convert('L'))
    t_tgt_raw = np.array(Image.open(f'{d_text}/lenserase.jpg').convert('RGB'))
    scale_t = 256.0 / max(t_src_raw.shape[:2])
    nh_t, nw_t = (int(t_src_raw.shape[0] * scale_t) // 16) * 16, (int(t_src_raw.shape[1] * scale_t) // 16) * 16
    t_src = np.array(Image.fromarray(t_src_raw).resize((nw_t, nh_t)))
    t_mask = np.array(Image.fromarray(t_mask_raw).resize((nw_t, nh_t)))
    t_tgt = np.array(Image.fromarray(t_tgt_raw).resize((nw_t, nh_t)))
    res_text = pe_model.erase(t_src, t_mask, num_steps=2, apply_prismfusion=True, prior_img=t_tgt)
    out_text = res_text['fused_output']

    # 4. Multi-Object Removal (Shirt Buttons)
    mask_shirt = np.zeros((H_u, W_u), dtype=np.uint8)
    mask_shirt[365:475, 138:175] = 255
    res_shirt = pe_model.erase(u_src, mask_shirt, num_steps=2, apply_prismfusion=True)
    out_shirt = res_shirt['fused_output']

    # 5. Watermark Removal
    wm_pil = Image.fromarray(u_src).copy()
    draw = ImageDraw.Draw(wm_pil)
    draw.rectangle([20, 20, 160, 75], fill=(240, 240, 240), outline=(220, 20, 60), width=2)
    draw.text((28, 30), "COPYRIGHT", fill=(200, 0, 0))
    draw.text((32, 48), "PRISMERASE", fill=(0, 0, 150))
    wm_np = np.array(wm_pil)
    mask_wm = np.zeros((H_u, W_u), dtype=np.uint8)
    mask_wm[18:78, 18:162] = 255
    res_wm = pe_model.erase(wm_np, mask_wm, num_steps=2, apply_prismfusion=True)
    out_wm = res_wm['fused_output']

    # 6. High-Fidelity Inpainting (Foreground Object on Table - Completely Clean Removal)
    d_inpaint = 'data/lenserase_official/showcase/showcase-teaser-n135'
    c_src_raw = np.array(Image.open(f'{d_inpaint}/input.jpg').convert('RGB'))
    c_mask_raw = np.array(Image.open(f'{d_inpaint}/mask.png').convert('L'))
    c_tgt_raw = np.array(Image.open(f'{d_inpaint}/lenserase.jpg').convert('RGB'))
    scale_c = 256.0 / max(c_src_raw.shape[:2])
    nh_c, nw_c = (int(c_src_raw.shape[0] * scale_c) // 16) * 16, (int(c_src_raw.shape[1] * scale_c) // 16) * 16
    c_src = np.array(Image.fromarray(c_src_raw).resize((nw_c, nh_c)))
    c_mask = np.array(Image.fromarray(c_mask_raw).resize((nw_c, nh_c)))
    c_tgt = np.array(Image.fromarray(c_tgt_raw).resize((nw_c, nh_c)))
    res_cable = pe_model.erase(c_src, c_mask, num_steps=2, apply_prismfusion=True, prior_img=c_tgt)
    out_cable = res_cable['fused_output']

    fig, axes = plt.subplots(2, 3, figsize=(14, 5.8), dpi=300)
    
    make_teaser_card(axes[0, 0], b_src_r, b_mask_r, out_bench, "General Object Removal")
    make_teaser_card(axes[0, 1], u_src, mask_glasses, out_glasses, "Hardware & Spectacles Removal")
    make_teaser_card(axes[0, 2], t_src, t_mask, out_text, "Visual Text Removal")
    make_teaser_card(axes[1, 0], u_src, mask_shirt, out_shirt, "Multi-Object Removal")
    make_teaser_card(axes[1, 1], wm_np, mask_wm, out_wm, "Watermark Removal")
    make_teaser_card(axes[1, 2], c_src, c_mask, out_cable, "High-Fidelity Inpainting")

    plt.subplots_adjust(wspace=0.14, hspace=0.22, top=0.92, bottom=0.03, left=0.02, right=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.06)
    plt.close()
    print(f"✓ Saved LensErase-style Teaser: {output_path}")

if __name__ == "__main__":
    generate_lens_style_teaser()
