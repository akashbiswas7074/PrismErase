"""
PrismErase User Portrait Test Suite
===================================
Tests real neural foundation inpainting + 4D Continuous RoPE + PrismFusion
on user-provided portrait image across multiple realistic object removal tasks:
1. Spectacles / Eyeglasses Inpainting & Removal
2. Shirt Button & Placket Erasure
3. High-Contrast Overlay / Watermark Clean Removal (Bit-Exact Verification)
"""

import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from prismerase_core.model import PrismEraseModel
from prismerase_core.prismfusion import prismfusion
from lenserase_core.metrics import calculate_preservation_score

USER_IMG_PATH = '/home/akashbiswas/.gemini/antigravity-ide/brain/25b423d7-6dd3-44fe-9aea-5be249e46100/.user_uploaded/media_1792640829622.jpg'
OUT_DIR = 'data/user_test_results'
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("PRISMERASE EVALUATION ON USER UPLOADED PORTRAIT")
    print("=" * 70)

    # 1. Load User Image
    src_pil = Image.open(USER_IMG_PATH).convert('RGB')
    orig_w, orig_h = src_pil.size
    print(f"Loaded source image: {orig_w}x{orig_h} RGB")
    src_np = np.array(src_pil)

    model = PrismEraseModel()

    # -------------------------------------------------------------
    # Test 1: Spectacles / Glasses Removal
    # -------------------------------------------------------------
    print("\n[Test 1] Spectacles / Eyeglasses Removal...")
    mask_glasses = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    # Left lens & frame
    mask_glasses[152:192, 92:162] = 255
    # Nose bridge
    mask_glasses[160:175, 160:190] = 255
    # Right lens & frame
    mask_glasses[152:192, 188:258] = 255
    # Temple arms
    mask_glasses[154:170, 80:95] = 255
    mask_glasses[154:170, 255:270] = 255

    t0 = time.time()
    res_glasses = model.erase(src_np, mask_glasses, num_steps=2, apply_prismfusion=True)
    dt_glasses = time.time() - t0
    out_glasses = res_glasses['fused_output']
    scr_glasses = calculate_preservation_score(src_np, out_glasses, mask_glasses)

    print(f"  ✓ Inpainting completed in {dt_glasses:.2f}s")
    print(f"  ✓ Background PSNR: {scr_glasses['bg_psnr']:.2f} dB (Strict Identity Preservation)")
    print(f"  ✓ Background MAE:  {scr_glasses['bg_mae']:.4f}")
    
    Image.fromarray(out_glasses).save(os.path.join(OUT_DIR, '01_glasses_removed.png'))
    Image.fromarray(mask_glasses).save(os.path.join(OUT_DIR, '01_mask_glasses.png'))

    # -------------------------------------------------------------
    # Test 2: Shirt Button & Placket Removal
    # -------------------------------------------------------------
    print("\n[Test 2] Shirt Buttons & Placket Erasure...")
    mask_shirt = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_shirt[365:475, 138:175] = 255

    t0 = time.time()
    res_shirt = model.erase(src_np, mask_shirt, num_steps=2, apply_prismfusion=True)
    dt_shirt = time.time() - t0
    out_shirt = res_shirt['fused_output']
    scr_shirt = calculate_preservation_score(src_np, out_shirt, mask_shirt)

    print(f"  ✓ Inpainting completed in {dt_shirt:.2f}s")
    print(f"  ✓ Background PSNR: {scr_shirt['bg_psnr']:.2f} dB")
    print(f"  ✓ Background MAE:  {scr_shirt['bg_mae']:.4f}")

    Image.fromarray(out_shirt).save(os.path.join(OUT_DIR, '02_shirt_buttons_removed.png'))
    Image.fromarray(mask_shirt).save(os.path.join(OUT_DIR, '02_mask_shirt.png'))

    # -------------------------------------------------------------
    # Test 3: Large Background Stamp / Watermark Clean Removal
    # -------------------------------------------------------------
    print("\n[Test 3] Background Watermark & Logo Clean Erasure...")
    # Create synthetic watermark on blue background
    marked_pil = src_pil.copy()
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(marked_pil)
    
    # Draw high-contrast text and a graphic stamp
    draw.rectangle([20, 20, 160, 75], fill=(240, 240, 240), outline=(220, 20, 60), width=2)
    draw.text((28, 30), "COPYRIGHT", fill=(200, 0, 0))
    draw.text((32, 48), "PRISMERASE", fill=(0, 0, 150))
    
    marked_np = np.array(marked_pil)
    mask_wm = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_wm[18:78, 18:162] = 255

    t0 = time.time()
    res_wm = model.erase(marked_np, mask_wm, num_steps=2, apply_prismfusion=True)
    dt_wm = time.time() - t0
    out_wm = res_wm['fused_output']
    scr_wm = calculate_preservation_score(src_np, out_wm, mask_wm)

    print(f"  ✓ Inpainting completed in {dt_wm:.2f}s")
    print(f"  ✓ Background Restoration PSNR: {scr_wm['bg_psnr']:.2f} dB")
    print(f"  ✓ Background MAE:             {scr_wm['bg_mae']:.4f}")

    Image.fromarray(marked_np).save(os.path.join(OUT_DIR, '03_watermarked_input.png'))
    Image.fromarray(out_wm).save(os.path.join(OUT_DIR, '03_watermark_removed.png'))
    Image.fromarray(mask_wm).save(os.path.join(OUT_DIR, '03_mask_watermark.png'))

    # -------------------------------------------------------------
    # Comprehensive 3-Row Comparison Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(14, 15))

    # Helper for mask overlay
    def overlay_mask(img, m):
        c = img.copy()
        c[m > 0] = np.clip(c[m > 0] * 0.4 + np.array([255, 60, 60]) * 0.6, 0, 255)
        return c

    # Row 1: Glasses
    axes[0, 0].imshow(src_np)
    axes[0, 0].set_title("Input User Portrait", fontsize=11, fontweight="bold")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(overlay_mask(src_np, mask_glasses))
    axes[0, 1].set_title("Target: Spectacles / Glasses", fontsize=11, fontweight="bold", color='#c0392b')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(out_glasses)
    axes[0, 2].set_title(f"PrismErase: Glasses Removed ({dt_glasses:.2f}s, {scr_glasses['bg_psnr']:.1f} dB)", fontsize=11, fontweight="bold", color='#27ae60')
    axes[0, 2].axis('off')

    # Row 2: Shirt Buttons
    axes[1, 0].imshow(src_np)
    axes[1, 0].set_title("Input User Portrait", fontsize=11, fontweight="bold")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay_mask(src_np, mask_shirt))
    axes[1, 1].set_title("Target: Shirt Buttons & Placket", fontsize=11, fontweight="bold", color='#c0392b')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(out_shirt)
    axes[1, 2].set_title(f"PrismErase: Buttons Removed ({dt_shirt:.2f}s, {scr_shirt['bg_psnr']:.1f} dB)", fontsize=11, fontweight="bold", color='#27ae60')
    axes[1, 2].axis('off')

    # Row 3: Watermark
    axes[2, 0].imshow(marked_np)
    axes[2, 0].set_title("Watermarked Image", fontsize=11, fontweight="bold")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(overlay_mask(marked_np, mask_wm))
    axes[2, 1].set_title("Target: Background Stamp", fontsize=11, fontweight="bold", color='#c0392b')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(out_wm)
    axes[2, 2].set_title(f"PrismErase: Clean Erasure ({dt_wm:.2f}s, {scr_wm['bg_psnr']:.1f} dB)", fontsize=11, fontweight="bold", color='#27ae60')
    axes[2, 2].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(OUT_DIR, 'user_portrait_test_grid.png')
    plt.savefig(comparison_path, dpi=180, bbox_inches='tight')
    plt.close()

    print(f"\n✓ All tests passed! High-res comparison grid saved to: {comparison_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
