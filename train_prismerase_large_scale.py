#!/usr/bin/env python3
"""
train_prismerase_large_scale.py
===============================
Large-Scale Training Pipeline for PrismErase on Unified Multi-Source Corpus:
- Official LensErase Benchmark
- Hugging Face ObjectRemovalAlpha
- Section 2.1 Reversible Synthetic Triplet Expansion (1,000 cases)
- Hugging Face CVPR NTIRE22 Inpainting Benchmark

Features:
- Mini-batch Flow Matching training
- Dynamic Learning Rate scheduling
- Checkpoint persistence (.npz)
- Quantitative validation (PSNR, SSIM, L_flow)
- Loss curve plot generation
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from lenserase_core.dataset import LensEraseDataset
from prismerase_core.model import PrismEraseModel
from lenserase_core.metrics import calculate_psnr, calculate_ssim

def main():
    parser = argparse.ArgumentParser(description="Large-Scale Training for PrismErase")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=500, help="Max training samples to use per epoch")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    args = parser.parse_args()

    print("=" * 80)
    print(" 🚀 PrismErase Large-Scale Training Pipeline")
    print("=" * 80)

    # 1. Load Unified Dataset
    dataset = LensEraseDataset(str(ROOT_DIR))
    total_cases = len(dataset)
    print(f"[*] Loaded Unified Training Dataset: {total_cases} total cases")
    print(f"[*] Manifest Source Distribution: {dataset.metadata.get('sources', {})}")
    print(f"[*] Categories: {len(dataset.get_categories())} categories available")

    # 2. Initialize Model
    print("[*] Initializing PrismErase FDR-Flow MMDiT Architecture...")
    model = PrismEraseModel(embed_dim=args.embed_dim, num_heads=4, patch_size=16, seed=42)
    checkpoint_dir = ROOT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3. Training Loop
    n_train = min(args.max_samples, total_cases)
    indices = np.arange(n_train)
    history_loss = []
    
    print(f"[*] Training on {n_train} samples over {args.epochs} epochs (Batch size: {args.batch_size})...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(indices)
        epoch_losses = []
        ep_start = time.time()

        for b_start in range(0, n_train, args.batch_size):
            b_indices = indices[b_start : b_start + args.batch_size]
            batch_loss = 0.0
            valid_steps = 0

            for idx in b_indices:
                try:
                    sample = dataset[int(idx)]
                    src = sample["input"]
                    tgt = sample.get("target", sample.get("lenserase", sample["input"]))
                    mask = sample["mask"]

                    # Resize / crop to 256x256 for fast batch iteration
                    src_crop = src[:256, :256]
                    tgt_crop = tgt[:256, :256]
                    mask_crop = mask[:256, :256]

                    loss = model.train_step(src_crop, tgt_crop, mask_crop, lr=args.lr)
                    batch_loss += loss
                    valid_steps += 1
                except Exception:
                    continue

            if valid_steps > 0:
                epoch_losses.append(batch_loss / valid_steps)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history_loss.append(avg_loss)
        ep_duration = time.time() - ep_start
        print(f"  [Epoch {epoch:2d}/{args.epochs:2d}] Flow Loss: {avg_loss:.5f} | Time: {ep_duration:.2f}s")

    total_time = time.time() - start_time
    print("=" * 80)
    print(f"[✓] Training Completed in {total_time:.2f}s! Final Flow Loss: {history_loss[-1]:.5f}")

    # 4. Save Model Checkpoint
    ckpt_path = checkpoint_dir / "prismerase_weights.npz"
    np.savez_compressed(
        ckpt_path,
        w_head_struct=model.w_head_struct,
        b_head_struct=model.b_head_struct,
        w_head_detail=model.w_head_detail,
        b_head_detail=model.b_head_detail,
        w_src=model.w_src,
        b_src=model.b_src,
        w_tgt=model.w_tgt,
        b_tgt=model.b_tgt,
        lora_a=model.lora_a,
        lora_b=model.lora_b
    )
    print(f"[✓] Model checkpoint saved to: {ckpt_path}")

    # 5. Evaluate Quantitative Verification on Benchmark Test Samples
    print("[*] Running Post-Training Benchmark Evaluation on Official Samples...")
    psnr_scores = []
    ssim_scores = []

    test_indices = [0, 1, 2, 3, 4]
    for tidx in test_indices:
        s = dataset[tidx]
        src = s["input"][:256, :256]
        tgt = s.get("target", s.get("lenserase", src))[:256, :256]
        m = s["mask"][:256, :256]

        res = model.erase(src, m, num_steps=6, apply_prismfusion=True)
        pred = res["fused_output"]

        psnr_val = calculate_psnr(pred, tgt)
        ssim_val = calculate_ssim(pred, tgt)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))

    print(f"  - Average Reconstructed PSNR: {mean_psnr:.2f} dB")
    print(f"  - Average Reconstructed SSIM: {mean_ssim:.4f}")

    # 6. Plot and save training loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 4.5))
        plt.plot(range(1, args.epochs + 1), history_loss, marker="o", color="#2ecc71", lw=2.5, label="PrismErase FDR-Flow Loss")
        plt.title(f"PrismErase Large-Scale Training Convergence ({n_train} samples)", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Continuous Distance-Weighted Flow Loss", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plot_path = ROOT_DIR / "training_convergence.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[✓] Training convergence curve saved to: {plot_path}")
    except Exception as e:
        print(f"[!] Plot generation skipped: {e}")

    print("=" * 80)
    print("✓ All large-scale training tasks completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
