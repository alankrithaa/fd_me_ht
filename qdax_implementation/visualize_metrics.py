"""
Visualization for FD-ME-HT flat-array repertoire.
Reshapes flat (num_centroids,) arrays back to (N, N) for heatmap display.
"""

import matplotlib.pyplot as plt
import numpy as np


GRID_SIZE = 5  # must match main_pipeline.py


def plot_experiment_heatmaps(repertoire, grid_size=GRID_SIZE):
    """
    Plot two heatmaps:
      1. Acceptance Rate  — how often each cell accepted a new candidate
      2. Mean Effect Size — average CLES of accepted replacements per cell

    Works with flat (num_centroids,) arrays by reshaping to (N, N).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    acc = np.asarray(repertoire.acceptance_counts).reshape(grid_size, grid_size)
    att = np.asarray(repertoire.attempt_counts).reshape(grid_size, grid_size)
    es = np.asarray(repertoire.sum_effect_sizes).reshape(grid_size, grid_size)

    rate = acc / (att + 1e-8)
    im1 = axes[0].imshow(
        rate,
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Acceptance Rate per Cell")
    plt.colorbar(im1, ax=axes[0])

    mean_es = es / (acc + 1e-8)
    im2 = axes[1].imshow(
        mean_es,
        cmap="magma",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Mean Effect Size (CLES) per Cell")
    plt.colorbar(im2, ax=axes[1])

    for ax in axes:
        ax.set_xlabel("Entropy bin")
        ax.set_ylabel("Brightness bin")
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

    plt.tight_layout()
    plt.savefig("results/dynamics_heatmaps.png", dpi=150)
    plt.close()
