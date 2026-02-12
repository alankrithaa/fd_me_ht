import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_experiment_heatmaps(repertoire):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Acceptance Rate (How "active" is each cell?)
    # Calculate: (Successful Replacements) / (Total Attempts)
    rate = repertoire.acceptance_counts / (repertoire.attempt_counts + 1e-8)
    im1 = axes[0].imshow(rate, cmap="viridis", origin="lower")
    axes[0].set_title("Acceptance Rate Heatmap")
    plt.colorbar(im1, ax=axes[0])

    # 2. Mean Effect Size (Quality of improvements)
    # Calculate: sum_delta / successful_replacements
    mean_es = repertoire.sum_effect_sizes / (repertoire.acceptance_counts + 1e-8)
    im2 = axes[1].imshow(mean_es, cmap="magma", origin="lower")
    axes[1].set_title("Mean Effect Size (Delta)")
    plt.colorbar(im2, ax=axes[1])

    for ax in axes:
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Brightness")

    plt.tight_layout()
    plt.savefig("results/dynamics_heatmaps.png")
    plt.show()

# To use this, just call it at the end of main_pipeline.py:
# plot_experiment_heatmaps(repertoire)