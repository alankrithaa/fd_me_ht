"""
cohens_d_analysis.py
===================
Compute Cohen's d between all pairs of occupied cells to verify that
elites have genuinely distinct quality distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def compute_cohens_d(scores_A: np.ndarray, scores_B: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two score distributions.
    
    Args:
        scores_A: (M,) array of ratings from cell A
        scores_B: (M,) array of ratings from cell B
        
    Returns:
        d: Cohen's d effect size
    """
    mean_A = np.mean(scores_A)
    mean_B = np.mean(scores_B)
    
    var_A = np.var(scores_A, ddof=1)  # Sample variance
    var_B = np.var(scores_B, ddof=1)
    
    pooled_std = np.sqrt((var_A + var_B) / 2.0)
    
    if pooled_std < 1e-8:
        return 0.0
    
    d = (mean_A - mean_B) / pooled_std
    
    return float(np.abs(d))  # Return absolute value


def compute_cohens_d_matrix(repertoire) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Cohen's d between all pairs of occupied cells.
    
    Args:
        repertoire: DROME or naive repertoire with .scores attribute
        
    Returns:
        d_matrix: (num_occupied, num_occupied) matrix of Cohen's d values
        cell_indices: Array of occupied cell indices
    """
    # Get occupied cells
    occupied_mask = repertoire.fitnesses > -np.inf
    cell_indices = np.where(occupied_mask)[0]
    num_occupied = len(cell_indices)
    
    if num_occupied < 2:
        return np.zeros((0, 0)), cell_indices
    
    # Extract scores for occupied cells
    scores = repertoire.scores[cell_indices]  # (num_occupied, M)
    
    # Compute pairwise Cohen's d
    d_matrix = np.zeros((num_occupied, num_occupied))
    
    for i in range(num_occupied):
        for j in range(num_occupied):
            if i == j:
                d_matrix[i, j] = 0.0
            else:
                d_matrix[i, j] = compute_cohens_d(scores[i], scores[j])
    
    return d_matrix, cell_indices


def visualize_cohens_d_heatmap(
    d_matrix_ht: np.ndarray,
    d_matrix_naive: np.ndarray,
    save_path: str = "results/cohens_d_comparison.png"
):
    """
    Create side-by-side heatmaps of Cohen's d matrices.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # HT heatmap
    sns.heatmap(
        d_matrix_ht,
        ax=axes[0],
        cmap="YlOrRd",
        vmin=0,
        vmax=max(d_matrix_ht.max(), d_matrix_naive.max()) if len(d_matrix_ht) > 0 else 3,
        cbar_kws={"label": "Cohen's d"},
        square=True,
    )
    axes[0].set_title("DROME (HT Gate)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Elite Cell Index")
    axes[0].set_ylabel("Elite Cell Index")
    
    # Naive heatmap
    sns.heatmap(
        d_matrix_naive,
        ax=axes[1],
        cmap="YlOrRd",
        vmin=0,
        vmax=max(d_matrix_ht.max(), d_matrix_naive.max()) if len(d_matrix_naive) > 0 else 3,
        cbar_kws={"label": "Cohen's d"},
        square=True,
    )
    axes[1].set_title("Naive Baseline", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Elite Cell Index")
    axes[1].set_ylabel("Elite Cell Index")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved Cohen's d comparison to {save_path}")


def analyze_cohens_d_distribution(
    d_matrix_ht: np.ndarray,
    d_matrix_naive: np.ndarray
) -> dict:
    """
    Compute summary statistics for Cohen's d distributions.
    """
    # Extract upper triangle (exclude diagonal)
    if len(d_matrix_ht) > 1:
        ht_values = d_matrix_ht[np.triu_indices_from(d_matrix_ht, k=1)]
    else:
        ht_values = np.array([])
    
    if len(d_matrix_naive) > 1:
        naive_values = d_matrix_naive[np.triu_indices_from(d_matrix_naive, k=1)]
    else:
        naive_values = np.array([])
    
    results = {
        "ht_mean": float(np.mean(ht_values)) if len(ht_values) > 0 else 0.0,
        "ht_median": float(np.median(ht_values)) if len(ht_values) > 0 else 0.0,
        "ht_min": float(np.min(ht_values)) if len(ht_values) > 0 else 0.0,
        "ht_max": float(np.max(ht_values)) if len(ht_values) > 0 else 0.0,
        "naive_mean": float(np.mean(naive_values)) if len(naive_values) > 0 else 0.0,
        "naive_median": float(np.median(naive_values)) if len(naive_values) > 0 else 0.0,
        "naive_min": float(np.min(naive_values)) if len(naive_values) > 0 else 0.0,
        "naive_max": float(np.max(naive_values)) if len(naive_values) > 0 else 0.0,
    }
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    
    print("=" * 70)
    print("COHEN'S D INTER-ELITE ANALYSIS")
    print("=" * 70)
    
    print("\nUsage:")
    print("  Add to main_pipeline.py after running both HT and naive:")
    print()
    print("  from cohens_d_analysis import *")
    print()
    print("  # Compute Cohen's d matrices")
    print("  d_ht, cells_ht = compute_cohens_d_matrix(rep_ht)")
    print("  d_naive, cells_naive = compute_cohens_d_matrix(rep_naive)")
    print()
    print("  # Visualize")
    print("  visualize_cohens_d_heatmap(d_ht, d_naive)")
    print()
    print("  # Analyze")
    print("  stats = analyze_cohens_d_distribution(d_ht, d_naive)")
    print("  print(f'HT mean Cohen\\'s d: {stats[\"ht_mean\"]:.3f}')")
    print("  print(f'Naive mean Cohen\\'s d: {stats[\"naive_mean\"]:.3f}')")
    
    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print()
    print("Cohen's d measures effect size between elite distributions:")
    print("  • d < 0.2  : Negligible difference (same quality)")
    print("  • d = 0.5  : Medium difference (distinct qualities)")
    print("  • d > 0.8  : Large difference (very distinct)")
    print()
    print("DROME should show:")
    print("  ✅ Higher mean Cohen's d (elites are more distinct)")
    print("  ✅ Fewer d < 0.2 pairs (fewer redundant elites)")
    print("  ✅ More d > 0.8 pairs (more genuinely different elites)")
    print()
    print("This proves DROME archives genuinely diverse solutions,")
    print("not just noisy variants of the same genotype!")