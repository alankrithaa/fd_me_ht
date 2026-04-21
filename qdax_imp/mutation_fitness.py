"""
Block 8: Pairwise Cohen's d across archive cells
=================================================
Computes Cohen's d between every pair of occupied cells:
    d_AB = (mean_A - mean_B) / sqrt((var_A + var_B) / 2)

High inter-cell Cohen's d = HT preserved genuinely different elites.
Low Cohen's d = cells are similar — archive lacks diversity in quality.

Add compute_pairwise_cohens_d() to diversity_metrics.py AND
run this file to generate the heatmap.

Run: python cohens_d.py
Output: results/cohens_d_heatmap.png
"""

import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from repertoire import DistributionalRepertoire
from evaluator import evaluate_via_pytorch, extract_features
from diversity_metrics import compute_pairwise_cohens_d
from main_pipeline import run_pipeline   # reuse the full pipeline

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE  — get a repertoire with occupancy
# ─────────────────────────────────────────────────────────────────────────────

print("Running pipeline to generate archive for Cohen's d analysis...")
rep_ht,    _, _ = run_pipeline(use_ht=True,  seed=42)
rep_naive, _, _ = run_pipeline(use_ht=False, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE + PLOT
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, rep, title in [
    (axes[0], rep_ht,    "Cohen's d — FD-ME-HT (with HT gate)"),
    (axes[1], rep_naive, "Cohen's d — Naive Baseline (no HT)"),
]:
    d_mat, labels, mean_d = compute_pairwise_cohens_d(rep)
    n = len(labels)

    # Diverging colormap centred at 0
    vmax = max(1.0, float(np.abs(d_mat).max()))
    im = ax.imshow(d_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_title(f"{title}\nMean |d| = {mean_d:.3f}", fontsize=11)
    ax.set_xlabel("Cell B", fontsize=10)
    ax.set_ylabel("Cell A", fontsize=10)

    # Annotate each cell with the d value
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{d_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

# Reference lines for effect size thresholds
fig.text(0.5, -0.02,
         "Cohen's d:  |d| < 0.2 negligible  |  0.5 medium  |  > 0.8 large",
         ha="center", fontsize=10, style="italic")

plt.suptitle("Pairwise Cohen's d Between Archive Cells\n"
             "High |d| = HT preserved genuinely different quality levels",
             fontsize=12)
plt.tight_layout()
plt.savefig("results/cohens_d_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
_, _, mean_d_ht    = compute_pairwise_cohens_d(rep_ht)
_, _, mean_d_naive = compute_pairwise_cohens_d(rep_naive)

print("\n" + "="*55)
print("  COHEN'S d SUMMARY  (screenshot this)")
print("="*55)
print(f"\n  HT archive:    mean |d| between cells = {mean_d_ht:.4f}")
print(f"  Naive archive: mean |d| between cells = {mean_d_naive:.4f}")
print("\n  Interpretation:")
print("  Higher mean |d| in HT = cells contain more differentiated elites.")
print("  HT guards against replacing with nearly-identical candidates,")
print("  preserving cells with genuinely distinct quality distributions.")
print("\nSaved: results/cohens_d_heatmap.png")