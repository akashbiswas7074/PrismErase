#!/usr/bin/env python-wrapped
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_pipeline_flowchart():
    output_path = Path("paper/figures/fig_pipeline_flowchart.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Background canvas
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')

    # Color palette
    c_blue = '#2563eb'
    c_indigo = '#4f46e5'
    c_purple = '#7c3aed'
    c_emerald = '#059669'
    c_slate = '#475569'
    c_rose = '#e11d48'
    c_bg_card = '#ffffff'

    # Title header
    ax.text(8.0, 8.6, "PrismErase Foundation Architecture & Operational Dataflow",
            ha='center', va='center', fontsize=18, fontweight='bold', color='#0f172a')
    ax.text(8.0, 8.25, "Continuous 4D Distance-Aware RoPE  •  Two-Branch FDR-Flow MMDiT  •  Adaptive Geodesic PrismFusion  •  Erase Contract K=(T,E,O)",
            ha='center', va='center', fontsize=10.5, color='#64748b', style='italic')

    # Helper box drawer
    def draw_box(x, y, w, h, title, subtitle, items, border_color, accent_color, tag=""):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.8, edgecolor=border_color, facecolor=c_bg_card,
            zorder=2
        )
        ax.add_patch(rect)

        # Header bar
        header_h = 0.55
        header_rect = patches.FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle="round,pad=0.01,rounding_size=0.12",
            linewidth=0, facecolor=border_color,
            zorder=3
        )
        ax.add_patch(header_rect)

        ax.text(x + 0.25, y + h - header_h / 2.0, title,
                ha='left', va='center', fontsize=10.2, fontweight='bold', color='#ffffff', zorder=4)
        if tag:
            ax.text(x + w - 0.20, y + h - header_h / 2.0, tag,
                    ha='right', va='center', fontsize=8.0, fontweight='bold', color='#ffffff',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.25, linewidth=0), zorder=4)

        if subtitle:
            ax.text(x + 0.25, y + h - header_h - 0.25, subtitle,
                    ha='left', va='center', fontsize=9.5, fontweight='bold', color=accent_color, zorder=4)

        start_y = y + h - header_h - 0.55
        for i, item in enumerate(items):
            ax.text(x + 0.35, start_y - i * 0.32, f"• {item}",
                    ha='left', va='center', fontsize=8.5, color='#334155', zorder=4)

    # Helper arrow drawer
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color='#64748b', lw=2.0, mutation_scale=16,
                connectionstyle="arc3,rad=0.0"
            ),
            zorder=5
        )
        if label:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.text(mx, my + 0.16, label, ha='center', va='bottom',
                    fontsize=8.0, fontweight='bold', color='#475569',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#f1f5f9', edgecolor='none', alpha=0.9),
                    zorder=6)

    # 1. Inputs Block
    draw_box(0.5, 4.4, 2.7, 3.2,
             "Input Modality", "RGB Image & Mask",
             [
                 "Source Image: I ∈ R^(H×W×3)",
                 "Target Mask: M ∈ {0,1}^(H×W)",
                 "High-Res Dynamic Calibration",
                 "Multi-domain support (0-100%)",
                 "Preserve resolution scale s"
             ],
             c_slate, c_slate, "STAGE 0")

    # 2. Stage 1: Erase Contract
    draw_box(3.6, 4.4, 3.3, 3.2,
             "1. Erase Contract K=(T,E,O)", "Scope Decomposition",
             [
                 "T: Target Object (Masked zone)",
                 "E: Associated Effects (Shadows)",
                 "O: Protected Context (Apertures)",
                 "Luminance gradient halo check",
                 "Reward Gate: r = 0.5^(n_0) · b"
             ],
             c_blue, c_blue, "STAGE 1")

    # 3. Stage 2: Continuous 4D RoPE
    draw_box(7.3, 4.4, 3.8, 3.2,
             "2. Continuous 4D RoPE", "Riemannian Manifold",
             [
                 "Signed Distance Field: Φ(x)",
                 "Curvature Field: κ = div(∇Φ/|∇Φ|)",
                 "Coordinates: p_i = (Φ_i, κ_i, h_i, w_i)",
                 "Inner Product: ⟨R_Θ(p_i)q, R_Θ(p_j)k⟩",
                 "C^∞ Boundary Continuity (0 Gibbs)"
             ],
             c_purple, c_purple, "STAGE 2")

    # 4. Stage 3: Two-Branch FDR-Flow
    draw_box(11.5, 4.4, 4.0, 3.2,
             "3. Two-Branch FDR-Flow", "MMDiT Velocity Field",
             [
                 "Base Infill / Foundation Prior z_0",
                 "v_struct: Macro shape & luminance",
                 "v_detail: Micro-frequency texture",
                 "v_total = v_struct + λ_detail · v_detail",
                 "Flow Matching ODE: dt · v_total"
             ],
             c_indigo, c_indigo, "STAGE 3")

    # Lower Row: Stage 4 (PrismFusion) & Stage 5 (Evaluation)
    # 5. Stage 4: Geodesic PrismFusion
    draw_box(3.6, 0.6, 6.0, 3.2,
             "4. Adaptive Geodesic PrismFusion", "Dual-Branch Residual Blending",
             [
                 "Linear RGB Conversion: I_lin = I^(2.2) (Gamma artifact prevention)",
                 "High-Freq Laplacian Residuals: R_high = |I_lin - G_σ * I_lin|",
                 "Dark-Region Residuals (γ=0.75): R_dark = |I_gen^(0.75) - I_src^(0.75)|",
                 "Dynamic Noise Variance Thresholds: τ_low, τ_high = f(σ_local)",
                 "Curvature Feathering: α_fuse(x) along geodesic contour lines",
                 "Bit-Exact Lock: α_fuse ≡ 0 ⟹ I_fused ≡ I_src (>55 dB PSNR, 0.00 MAE)"
             ],
             c_emerald, c_emerald, "STAGE 4")

    # 6. Stage 5: Evaluation & Certification
    draw_box(10.1, 0.6, 5.4, 3.2,
             "5. Evaluation & HiFi Certification", "Contract-Aware Scoring",
             [
                 "Removal Score: Absence of target features (≥85%)",
                 "Reconstruction Score: Infill coherence & smooth gradient",
                 "Preservation Score: Non-target pixel lock (≥75%)",
                 "Outside PSNR: Background preservation (>58 dB)",
                 "Exponential Penalty: r = 0.5^(n_0) · b on ghost residue",
                 "Certification: is_hifi = True (HRR = 96.2% on benchmark)"
             ],
             c_rose, c_rose, "STAGE 5")

    # Connectors
    # Stage 0 -> Stage 1
    draw_arrow(3.2, 6.0, 3.6, 6.0, "I, M")

    # Stage 1 -> Stage 2
    draw_arrow(6.9, 6.0, 7.3, 6.0, "T, E, O")

    # Stage 2 -> Stage 3
    draw_arrow(11.1, 6.0, 11.5, 6.0, "p_i ∈ R^4")

    # Stage 3 -> Stage 4
    ax.annotate(
        "", xy=(8.0, 3.8), xytext=(13.2, 4.4),
        arrowprops=dict(
            arrowstyle="-|>", color='#64748b', lw=2.0, mutation_scale=16,
            connectionstyle="arc3,rad=-0.18"
        ),
        zorder=5
    )
    ax.text(10.8, 4.15, "I_gen (Flow Output)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#475569',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#f1f5f9', edgecolor='none', alpha=0.9),
            zorder=6)

    # Stage 0 -> Stage 4 (Direct source pass-through)
    ax.annotate(
        "", xy=(3.6, 2.2), xytext=(1.85, 4.4),
        arrowprops=dict(
            arrowstyle="-|>", color='#64748b', lw=2.0, mutation_scale=16,
            connectionstyle="arc3,rad=0.25"
        ),
        zorder=5
    )
    ax.text(2.0, 3.0, "Source I (Bit-Exact)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#475569',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#f1f5f9', edgecolor='none', alpha=0.9),
            zorder=6)

    # Stage 4 -> Stage 5
    draw_arrow(9.6, 2.2, 10.1, 2.2, "I_fused")

    # Final Output Badge
    ax.text(12.8, 0.22, "★ Certified Output: Seam-Free Infill + Bit-Exact Background (>55 dB PSNR)",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#065f46',
            bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.15', facecolor='#d1fae5', edgecolor='#10b981', lw=1.2),
            zorder=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f"✓ Pipeline flowchart generated at: {output_path}")

if __name__ == "__main__":
    create_pipeline_flowchart()
