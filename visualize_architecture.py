import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, ConnectionPatch
import numpy as np

os.environ["MPLBACKEND"] = "Agg"
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def draw_ms_kanconv_architecture(save_path="results/figures/architecture_diagram.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Color Palette - Modern Journal Aesthetics (Sensors/IEEE style)
    c_input = "#E0E7FF"      # Light Indigo
    c_input_b = "#4338CA"
    c_conv = "#FEF3C7"       # Soft Amber
    c_conv_b = "#D97706"
    c_kan = "#FCE7F3"        # Soft Pink / Rose
    c_kan_b = "#BE185D"
    c_se = "#D1FAE5"         # Soft Emerald
    c_se_b = "#059669"
    c_head = "#EDE9FE"       # Soft Purple
    c_head_b = "#6D28D9"
    c_bg_block = "#F8FAFC"   # Slate 50
    c_border_block = "#94A3B8"
    c_text = "#0F172A"

    def draw_box(x, y, w, h, text, subtext="", bg="#FFFFFF", border="#000000", radius=0.15, fontsize=9.5):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                             facecolor=bg, edgecolor=border, linewidth=1.5, zorder=3)
        ax.add_patch(box)
        if subtext:
            ax.text(x + w/2, y + h/2 + 0.12, text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=c_text, zorder=4)
            ax.text(x + w/2, y + h/2 - 0.15, subtext, ha='center', va='center',
                    fontsize=fontsize-2, color="#475569", zorder=4)
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=c_text, zorder=4)

    def draw_arrow(x1, y1, x2, y2, label="", color="#475569", width=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=width, mutation_scale=12),
                    zorder=2)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', va='bottom',
                    fontsize=8, color="#334155", fontweight='semibold')

    # Title
    ax.text(8, 8.5, "MS-KANConv: Multi-Scale KAN-Augmented Convolutional Network for Wearable HAR",
            ha='center', va='center', fontsize=14, fontweight='bold', color="#0F172A")

    # 1. INPUT
    draw_box(0.5, 3.8, 1.4, 1.4, "Input Sensor\nSignal", "(B, C, T)\n128 samples", bg=c_input, border=c_input_b)

    # Arrow Input to Block 1
    draw_arrow(2.0, 4.5, 2.7, 4.5)

    # 2. MS-KANConv BLOCK 1 (CONTAINER)
    b1_box = FancyBboxPatch((2.7, 1.8), 5.4, 5.4, boxstyle="round,pad=0.2",
                            facecolor=c_bg_block, edgecolor=c_border_block, linewidth=1.8, linestyle='--', zorder=1)
    ax.add_patch(b1_box)
    ax.text(5.4, 7.0, "MS-KANConv Block 1 (Multi-Scale Temporal Feature Extraction)",
            ha='center', va='center', fontsize=10, fontweight='bold', color="#1E293B")

    # Branches in Block 1
    y_branches = [5.7, 4.5, 3.3]
    ks_list = [3, 5, 7]
    d_list = [1, 2, 4]
    branch_names = ["Short-term Scale", "Mid-term Scale", "Long-term Scale"]

    for i, (yb, ks, d, bname) in enumerate(zip(y_branches, ks_list, d_list, branch_names)):
        # Branch Split arrow
        draw_arrow(2.9, 4.5, 3.1, yb)
        
        # Conv + KAN-Act Combo
        draw_box(3.1, yb - 0.35, 1.6, 0.7, f"Dilated Conv1D", f"k={ks}, d={d} (C=64)", bg=c_conv, border=c_conv_b, fontsize=8)
        draw_arrow(4.8, yb, 5.1, yb)
        draw_box(5.1, yb - 0.35, 1.4, 0.7, "KAN-Act", "B-Spline Act", bg=c_kan, border=c_kan_b, fontsize=8)
        
        # Merge arrow to Concat
        draw_arrow(6.6, yb, 7.0, 4.5)

    # Concat & SE & Residual
    draw_box(7.0, 3.9, 0.8, 1.2, "Concat\n(3×64)", "C=192", bg="#E2E8F0", border="#64748B", fontsize=7.5)
    draw_arrow(7.85, 4.5, 8.2, 4.5)

    # SE Block & LayerNorm
    draw_box(8.2, 3.8, 1.2, 1.4, "SE Channel\nAttention &\nLayerNorm", "Reduction=4", bg=c_se, border=c_se_b, fontsize=8)

    # Residual path for Block 1
    ax.annotate('', xy=(8.8, 3.8), xytext=(2.9, 3.8),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.2, linestyle=':',
                                connectionstyle="arc3,rad=-0.3", mutation_scale=10),
                zorder=2)
    ax.text(5.4, 2.0, "+ Residual Connection (1×1 Conv)", ha='center', va='center', fontsize=7.5, color="#64748B", style='italic')

    # Arrow Block 1 -> Block 2
    draw_arrow(9.5, 4.5, 10.0, 4.5)

    # 3. MS-KANConv BLOCK 2 (Condensed View)
    b2_box = FancyBboxPatch((10.0, 2.5), 1.8, 4.0, boxstyle="round,pad=0.15",
                            facecolor=c_bg_block, edgecolor=c_border_block, linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(b2_box)
    ax.text(10.9, 6.2, "MS-KANConv\nBlock 2", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#1E293B")
    draw_box(10.2, 4.0, 1.4, 1.8, "Multi-Scale\nKAN Branches\n+\nSE Attention", "(3×128 = 384)", bg=c_conv, border=c_conv_b, fontsize=8)
    
    # 4. GLOBAL AVERAGE POOLING
    draw_arrow(11.9, 4.5, 12.3, 4.5)
    draw_box(12.3, 3.9, 0.9, 1.2, "GAP", "Temporal\nPooled", bg="#E2E8F0", border="#64748B", fontsize=8)

    # 5. KAN CLASSIFICATION HEAD
    draw_arrow(13.25, 4.5, 13.6, 4.5)
    draw_box(13.6, 3.6, 1.1, 1.8, "KAN-Linear\nClassifier", "KAN(384→128)\nKAN(128→Classes)", bg=c_head, border=c_head_b, fontsize=7.5)

    # 6. OUTPUT PREDICTIONS
    draw_arrow(14.8, 4.5, 15.1, 4.5)
    draw_box(15.1, 3.8, 0.75, 1.4, "Activity\nClasses", "Softmax\nLogits", bg=c_input, border=c_input_b, fontsize=7.5)

    # 7. DETAIL CALLOUT: KAN-Activation B-Spline Mechanism (Bottom Left/Middle)
    callout_box = FancyBboxPatch((0.5, 0.35), 7.0, 1.25, boxstyle="round,pad=0.1",
                                 facecolor="#FFFBEB", edgecolor="#F59E0B", linewidth=1.2, zorder=2)
    ax.add_patch(callout_box)
    ax.text(0.7, 1.3, "Novelty Mechanism: Learnable KAN-Activation vs Standard Fixed Activation",
            fontsize=8.5, fontweight='bold', color="#92400E")
    ax.text(0.7, 0.95, r"$\mathbf{\phi(x) = w_{base} \cdot \mathrm{SiLU}(x) + w_{spline} \cdot \sum_{i} c_i B_i(\mathrm{tanh}(x))}$",
            fontsize=9.5, color="#1E293B")
    ax.text(0.7, 0.55, "• Replaces static ReLU with adaptive B-splines per channel to learn distinct temporal impact curves.",
            fontsize=7.5, color="#451A03")

    # 8. LEGEND / METADATA (Bottom Right)
    leg_box = FancyBboxPatch((8.0, 0.35), 7.5, 1.25, boxstyle="round,pad=0.1",
                             facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.2, zorder=2)
    ax.add_patch(leg_box)
    ax.text(8.2, 1.3, "Architectural Properties for Q2 Wearable HAR Submission:", fontsize=8.5, fontweight='bold', color="#334155")
    ax.text(8.2, 0.95, "✓ Multi-Scale Receptive Fields (d=1, 2, 4) capture micro-gestures & macro-postures simultaneously", fontsize=7.5, color="#1E293B")
    ax.text(8.2, 0.65, "✓ End-to-End learnable non-linear activations without deep stacking overhead (Low FLOPs / Edge-ready)", fontsize=7.5, color="#1E293B")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram successfully saved to {save_path}")

if __name__ == "__main__":
    draw_ms_kanconv_architecture()
