"""
Diversity metrics for DistributionalRepertoire.
Uses QDax convention: empty cells have fitness == -jnp.inf.
"""
import jax.numpy as jnp
import numpy as np


def _occupancy_mask(repertoire):
    """Boolean mask of occupied cells — QDax convention: empty = -inf."""
    return np.asarray(repertoire.fitnesses > -jnp.inf)


def compute_behaviour_diversity(repertoire):
    """Coverage (%) and BD spread (mean distance from grid centre)."""
    mask = _occupancy_mask(repertoire)
    active_bds = np.asarray(repertoire.descriptors)[mask]
    coverage = float(np.mean(mask) * 100.0)
    if active_bds.shape[0] == 0:
        return {"bd_spread": 0.0, "coverage": coverage}
    center = np.array([0.5, 0.5], dtype=np.float32)
    avg_spread = float(np.mean(np.linalg.norm(active_bds - center, axis=1)))
    return {"coverage": coverage, "bd_spread": avg_spread}


def compute_pairwise_elite_distance(repertoire):
    """Mean pairwise Euclidean distance between occupied elite genotypes."""
    mask = _occupancy_mask(repertoire)
    occupied_genos = np.asarray(repertoire.genotypes)[mask]
    n = occupied_genos.shape[0]
    if n < 2:
        return 0.0
    sq_norms = np.sum(occupied_genos**2, axis=1)
    dot_product = occupied_genos @ occupied_genos.T
    dist_sq = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot_product, 0.0)
    dist_matrix = np.sqrt(dist_sq + 1e-6)
    return float(np.sum(dist_matrix) / (n * (n - 1)))


def compute_pairwise_cohens_d(repertoire):
    """
    Pairwise Cohen's d between every pair of occupied cells.

    d_AB = (mean_A - mean_B) / sqrt((var_A + var_B) / 2)

    Uses the VLM score distributions stored per cell (shape M,).

    Returns:
        d_matrix:   (N_occ, N_occ) array
        cell_labels: list of cell index strings
        mean_abs_d: scalar summary — mean |d| across all off-diagonal pairs
    """
    mask = _occupancy_mask(repertoire)
    occupied_idx = np.flatnonzero(mask)
    n_occ = occupied_idx.size

    if n_occ < 2:
        label = f"c{int(occupied_idx[0])}" if n_occ == 1 else "c0"
        return np.zeros((max(n_occ, 1), max(n_occ, 1)), dtype=np.float32), [label], 0.0

    scores = np.asarray(repertoire.scores)[occupied_idx]   # (n_occ, M)
    means = np.mean(scores, axis=1)
    varis = np.var(scores, axis=1)

    mean_diff = means[:, None] - means[None, :]
    pooled_std = np.sqrt((varis[:, None] + varis[None, :]) / 2.0 + 1e-10)
    d_matrix = np.divide(
        mean_diff,
        pooled_std,
        out=np.zeros_like(mean_diff, dtype=np.float32),
        where=pooled_std > 0.0,
    )
    np.fill_diagonal(d_matrix, 0.0)

    cell_labels = [f"c{idx}" for idx in occupied_idx]
    upper_tri = d_matrix[np.triu_indices(n_occ, k=1)]
    mean_abs_d = float(np.mean(np.abs(upper_tri))) if upper_tri.size else 0.0
    return d_matrix, cell_labels, mean_abs_d