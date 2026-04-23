"""
main_pipeline.py
================
DROME: sequential two-stage pipeline with multi-seed statistical validation.

Architecture:
  Stage 1 — A1 (T1 iterations, naive MAP-Elites):
      Fast, cheap exploration. Naive mean-based replacement.
      Builds broad behavioural coverage across the grid.
      Emitter samples from A1 throughout Stage 1.

  Stage 2 — A2 (T2 iterations, HT-gated MAP-Elites):
      A1 finishes fully. Its occupied elites seed A2.
      Emitter then samples from A2 and proposes new candidates.
      Each candidate is freshly evaluated and must pass the
      three-condition HT gate to replace an A2 incumbent.
      A1 is FROZEN after Stage 1 — not touched in Stage 2.

Cost intuition:
  A1 is intentionally cheap: many iterations, no statistical overhead.
  A2 is intentionally selective: fewer iterations, every update must be
  statistically justified. The total evaluation budget is T1 + T2.

Statistical validation (N=10 seeds, reduced for 8 GB RAM safety):
  Comparison: final A2 QD vs final A1 QD (same seed).
  - Paired t-test (H0: mean(QD_A2 - QD_A1) = 0, two-tailed).
  - Bootstrap 95% CI on the mean difference (10,000 resamples).
  - Mean +/- std for QD score, coverage, max fitness.

p-value in ht_logic.py: exact two-tailed Gaussian CDF via
    jax.scipy.stats.norm.cdf — fully JIT-compatible, no approximation.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

import jax
import jax.numpy as jnp
from jax import pure_callback, ShapeDtypeStruct

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids

from evaluator import (
    BD_BRIGHTNESS_IDX, BD_ENTROPY_IDX,
    evaluate_via_pytorch, extract_features,
)
from fdme_emitter import FDMEEmitter
from fdme_map_elites import DROMEMAPElites
from diversity_metrics import compute_behaviour_diversity, compute_pairwise_elite_distance


# ── Hyperparameters ────────────────────────────────────────────────────────────
GRID_SIZE       = 5
LATENT_DIM      = 8
NUM_RATERS      = 10        # M: all VLMs score each image
IMG_RES         = 64
SAMPLER_K       = 5

# Stage lengths
T1              = 20        # A1 (naive)    — more iterations, cheap exploration
T2              = 10       # A2 (HT-gated) — fewer iterations, statistically validated

BATCH_SIZE      = 4
INIT_BATCH_SIZE = 8

ALPHA           = 0.10      # HT gate significance threshold alpha
DELTA_MIN       = 0.55      # HT gate CLES floor delta_min

MUTATION_PROB   = 0.5
MUTATION_SIGMA  = 0.1

N_SEEDS         = 5        # reduced for 8 GB RAM safety
SEEDS           = list(range(N_SEEDS))   # 0 .. 9


# ── Descriptor callback ────────────────────────────────────────────────────────
def _numpy_descriptor_batch(images_np: np.ndarray) -> np.ndarray:
    feats = np.stack([extract_features(img) for img in images_np], axis=0)
    return feats[:, [BD_BRIGHTNESS_IDX, BD_ENTROPY_IDX]].astype(np.float32)


def descriptor_via_callback(images: jnp.ndarray) -> jnp.ndarray:
    out_shape = ShapeDtypeStruct((images.shape[0], 2), jnp.float32)
    return pure_callback(_numpy_descriptor_batch, out_shape, images)


# ── Simulated diffusion sampler ────────────────────────────────────────────────
def simulate_sampler_batch(genotypes: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """
    Lightweight fake diffusion sampler.
    Input:  (B, D) genotypes
    Output: (B, H, W, 3) images in [0, 1]

    The mean of each genotype controls base brightness.
    K rounds of decaying Gaussian noise simulate iterative diffusion refinement.
    In production this is replaced by Stable Diffusion + GNSO guidance.
    """
    B    = genotypes.shape[0]
    base = jnp.clip(jnp.mean(genotypes, axis=1) * 0.5 + 0.5, 0.05, 0.95)
    x    = jnp.broadcast_to(base[:, None, None, None], (B, IMG_RES, IMG_RES, 3))
    keys = jax.random.split(key, SAMPLER_K)
    for k in range(SAMPLER_K):
        noise = jax.random.normal(keys[k], shape=x.shape, dtype=jnp.float32)
        x     = x + noise / (k + 2)
    return jnp.clip(x, 0.0, 1.0)


# ── Scoring function ───────────────────────────────────────────────────────────
def scoring_function(genotypes: jnp.ndarray, key: jax.Array):
    """
    (genotypes, key) -> (fitnesses, descriptors, extra_scores)

    fitness = -mean(scores): QDAX maximises fitness, so higher fitness
    means lower mean VLM rating = higher perceived quality (Eq. 3).
    """
    images      = simulate_sampler_batch(genotypes, key)
    descriptors = descriptor_via_callback(images)
    scores      = evaluate_via_pytorch(images)
    fitnesses   = -jnp.mean(scores, axis=1)
    return fitnesses, descriptors, {"scores": scores}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(repertoire) -> dict:
    empty       = repertoire.fitnesses == -jnp.inf
    qd_score    = float(jnp.sum(repertoire.fitnesses, where=~empty))
    coverage    = float(100.0 * jnp.mean((~empty).astype(jnp.float32)))
    max_fitness = float(jnp.max(jnp.where(~empty, repertoire.fitnesses, -jnp.inf)))
    return {"qd_score": qd_score, "coverage": coverage, "max_fitness": max_fitness}


# ── Single-seed pipeline ───────────────────────────────────────────────────────
def run_one_seed(seed: int, t1: int = T1, t2: int = T2):
    """
    Run the full sequential two-stage DROME experiment for one seed.

    Stage 1: run A1 for t1 iterations (naive MAP-Elites).
    Stage 2: seed A2 from A1, run A2 for t2 iterations (HT-gated).

    Returns:
        a1              -- frozen A1 after Stage 1
        a2              -- final A2 after Stage 2
        qd_a1_history   -- list of A1 QD scores per Stage-1 iteration
        qd_a2_history   -- list of A2 QD scores per Stage-2 iteration
        metrics_a1_final -- final A1 metrics (end of Stage 1)
        metrics_a2_final -- final A2 metrics (end of Stage 2)
    """
    key = jax.random.key(seed)

    # Build initial population
    key, init_key = jax.random.split(key)
    init_genotypes = jax.random.normal(
        init_key, shape=(INIT_BATCH_SIZE, LATENT_DIM), dtype=jnp.float32
    )
    centroids = compute_euclidean_centroids(
        grid_shape=(GRID_SIZE, GRID_SIZE), minval=0.0, maxval=1.0
    )
    emitter = FDMEEmitter(
        batch_size     = BATCH_SIZE,
        latent_dim     = LATENT_DIM,
        mutation_prob  = MUTATION_PROB,
        mutation_sigma = MUTATION_SIGMA,
    )
    map_elites = DROMEMAPElites(
        scoring_function = scoring_function,
        emitter          = emitter,
        metrics_function = compute_metrics,
        alpha            = ALPHA,
        delta_min        = DELTA_MIN,
    )

    # ── Stage 1: run A1 ───────────────────────────────────────────────────────
    key, subkey = jax.random.split(key)
    a1, emitter_state, key = map_elites.init_a1(init_genotypes, centroids, subkey)

    qd_a1_history = []
    for _ in range(t1):
        key, subkey = jax.random.split(key)
        a1, emitter_state, metrics_a1 = map_elites.update_a1(a1, emitter_state, subkey)
        qd_a1_history.append(metrics_a1["qd_score"])

    metrics_a1_final = compute_metrics(a1)

    # ── Stage 2: seed A2 from A1 and run ─────────────────────────────────────
    key, subkey = jax.random.split(key)
    a2, emitter_state, key = map_elites.init_a2(a1, subkey)

    qd_a2_history = []
    for _ in range(t2):
        key, subkey = jax.random.split(key)
        a2, emitter_state, metrics_a2 = map_elites.update_a2(a2, emitter_state, subkey)
        qd_a2_history.append(metrics_a2["qd_score"])

    metrics_a2_final = compute_metrics(a2)

    return a1, a2, qd_a1_history, qd_a2_history, metrics_a1_final, metrics_a2_final


# ── Multi-seed runner ──────────────────────────────────────────────────────────
def run_multi_seed(seeds: list = SEEDS, t1: int = T1, t2: int = T2,
                   verbose: bool = True):
    """
    Run the sequential two-stage pipeline across N seeds.

    Returns:
        all_a1            -- list of frozen A1 archives (one per seed)
        all_a2            -- list of final A2 archives (one per seed)
        qd_a1_matrix      -- (N, T1) A1 QD scores
        qd_a2_matrix      -- (N, T2) A2 QD scores
        final_metrics_a1  -- list of final A1 metric dicts
        final_metrics_a2  -- list of final A2 metric dicts
    """
    all_a1, all_a2 = [], []
    qd_a1_rows, qd_a2_rows = [], []
    final_metrics_a1, final_metrics_a2 = [], []

    if verbose:
        print(f"\nRunning {len(seeds)} seeds  "
              f"(Stage 1: {t1} iters, Stage 2: {t2} iters) ...")

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"  seed {seed:2d}  ({i+1:2d}/{len(seeds)})", end="", flush=True)

        a1, a2, h_a1, h_a2, m_a1, m_a2 = run_one_seed(seed, t1, t2)

        all_a1.append(a1)
        all_a2.append(a2)
        qd_a1_rows.append(h_a1)
        qd_a2_rows.append(h_a2)
        final_metrics_a1.append(m_a1)
        final_metrics_a2.append(m_a2)

        if verbose:
            print(f"  |  A1 QD={m_a1['qd_score']:.4f}  "
                  f"A2 QD={m_a2['qd_score']:.4f}  "
                  f"A1 cov={m_a1['coverage']:.0f}%  "
                  f"A2 cov={m_a2['coverage']:.0f}%")

    return (all_a1, all_a2,
            np.array(qd_a1_rows), np.array(qd_a2_rows),
            final_metrics_a1, final_metrics_a2)


# ── Statistical comparison ────────────────────────────────────────────────────
def statistical_comparison(final_metrics_a1: list,
                            final_metrics_a2: list,
                            n_boot: int = 10_000) -> dict:
    """
    Paired t-test + bootstrap CI: A2 (DROME) vs A1 (naive) final QD scores.

    Pairing: same seed => same random initialisation.
    H0: mean(QD_A2 - QD_A1) = 0  (two-tailed).
    Bootstrap: 10,000 resamples of per-seed differences.
    """
    qd_a1 = np.array([m["qd_score"] for m in final_metrics_a1])
    qd_a2 = np.array([m["qd_score"] for m in final_metrics_a2])
    diffs  = qd_a2 - qd_a1       # positive => A2 improved over A1

    t_stat, t_pval = scipy_stats.ttest_1samp(diffs, popmean=0.0)

    rng        = np.random.default_rng(0)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    cov_a1 = [m["coverage"]    for m in final_metrics_a1]
    cov_a2 = [m["coverage"]    for m in final_metrics_a2]
    mf_a2  = [m["max_fitness"] for m in final_metrics_a2]

    result = {
        "n_seeds":          len(diffs),
        "a1_qd_mean":       float(np.mean(qd_a1)),
        "a1_qd_std":        float(np.std(qd_a1)),
        "a2_qd_mean":       float(np.mean(qd_a2)),
        "a2_qd_std":        float(np.std(qd_a2)),
        "mean_diff":        float(np.mean(diffs)),
        "std_diff":         float(np.std(diffs)),
        "t_stat":           float(t_stat),
        "t_pval":           float(t_pval),
        "boot_ci_lo":       float(ci_lo),
        "boot_ci_hi":       float(ci_hi),
        "a1_coverage_mean": float(np.mean(cov_a1)),
        "a1_coverage_std":  float(np.std(cov_a1)),
        "a2_coverage_mean": float(np.mean(cov_a2)),
        "a2_coverage_std":  float(np.std(cov_a2)),
        "a2_maxfit_mean":   float(np.mean(mf_a2)),
        "a2_maxfit_std":    float(np.std(mf_a2)),
    }

    sig               = result["t_pval"] < 0.05
    ci_excludes_zero  = result["boot_ci_lo"] > 0 or result["boot_ci_hi"] < 0

    print("\n" + "=" * 65)
    print(f"  STATISTICAL COMPARISON  (N={result['n_seeds']} paired seeds)")
    print(f"  A1: naive MAP-Elites ({T1} iters)  ->  A2: HT-gated ({T2} iters)")
    print(f"  Paired t-test: H0: mean(QD_A2 - QD_A1) = 0  (two-tailed)")
    print("=" * 65)
    print(f"  A1 final QD:     {result['a1_qd_mean']:+.4f} ± {result['a1_qd_std']:.4f}")
    print(f"  A2 final QD:     {result['a2_qd_mean']:+.4f} ± {result['a2_qd_std']:.4f}")
    print(f"  Mean diff A2-A1: {result['mean_diff']:+.4f} ± {result['std_diff']:.4f}")
    print(f"  t-statistic:     {result['t_stat']:.3f}")
    print(f"  p-value:         {result['t_pval']:.4f}  "
          f"{'** (p < 0.05)' if sig else '(n.s.)'}")
    print(f"  Bootstrap 95% CI: [{result['boot_ci_lo']:+.4f}, "
          f"{result['boot_ci_hi']:+.4f}]")
    print(f"  CI excludes 0:   {'YES' if ci_excludes_zero else 'NO'}")
    print()
    print(f"  A1 Coverage: {result['a1_coverage_mean']:.1f}% ± "
          f"{result['a1_coverage_std']:.1f}%")
    print(f"  A2 Coverage: {result['a2_coverage_mean']:.1f}% ± "
          f"{result['a2_coverage_std']:.1f}%")
    print(f"  A2 Max Fitness: {result['a2_maxfit_mean']:.4f} ± "
          f"{result['a2_maxfit_std']:.4f}")
    print("=" * 65)

    return result


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_qd_curves(qd_a1_matrix: np.ndarray,
                   qd_a2_matrix: np.ndarray,
                   t1: int = T1,
                   t2: int = T2,
                   save_path: str = "results/qd_curve_sequential.png"):
    """
    Two-panel plot: Stage 1 (A1) on left, Stage 2 (A2) on right.
    Mean +/- 1 SD across seeds shown as shaded bands.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Stage 1: A1
    iters_a1 = np.arange(t1)
    mean_a1  = qd_a1_matrix.mean(axis=0)
    std_a1   = qd_a1_matrix.std(axis=0)
    ax1.plot(iters_a1, mean_a1, color="darkorange", linewidth=2,
             label="A1 (naive MAP-Elites)")
    ax1.fill_between(iters_a1, mean_a1 - std_a1, mean_a1 + std_a1,
                     alpha=0.20, color="darkorange")
    ax1.set_xlabel("Iteration (Stage 1)")
    ax1.set_ylabel("QD Score")
    ax1.set_title(f"Stage 1: A1 Naive Exploration\n"
                  f"({t1} iterations, N={qd_a1_matrix.shape[0]} seeds)")
    ax1.legend()

    # Stage 2: A2
    iters_a2 = np.arange(t2)
    mean_a2  = qd_a2_matrix.mean(axis=0)
    std_a2   = qd_a2_matrix.std(axis=0)
    ax2.plot(iters_a2, mean_a2, color="steelblue", linewidth=2,
             label="A2 (HT-gated DROME)")
    ax2.fill_between(iters_a2, mean_a2 - std_a2, mean_a2 + std_a2,
                     alpha=0.20, color="steelblue")
    # Mark the starting QD (inherited from A1) as a horizontal dashed line
    ax2.axhline(y=qd_a1_matrix[:, -1].mean(), color="darkorange",
                linestyle="--", linewidth=1.5, alpha=0.8,
                label=f"A1 final QD (mean = {qd_a1_matrix[:, -1].mean():.4f})")
    ax2.set_xlabel("Iteration (Stage 2)")
    ax2.set_ylabel("QD Score")
    ax2.set_title(f"Stage 2: A2 HT-Gated Refinement\n"
                  f"({t2} iterations, seeded from A1)")
    ax2.legend()

    fig.suptitle(
        f"Sequential Two-Stage DROME  "
        f"(Mean \u00b1 1 SD, N={qd_a1_matrix.shape[0]} seeds)",
        fontsize=13
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_diff_distribution(final_metrics_a1: list,
                            final_metrics_a2: list,
                            stats: dict,
                            save_path: str = "results/qd_diff_distribution.png"):
    """Per-seed (A2 - A1) final QD difference histogram with CI overlay."""
    qd_a1 = np.array([m["qd_score"] for m in final_metrics_a1])
    qd_a2 = np.array([m["qd_score"] for m in final_metrics_a2])
    diffs  = qd_a2 - qd_a1

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diffs, bins=10, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(stats["mean_diff"],  color="navy",   linestyle="-",  linewidth=2,
               label=f"Mean = {stats['mean_diff']:+.4f}")
    ax.axvline(stats["boot_ci_lo"], color="tomato", linestyle="--", linewidth=1.5,
               label=f"95% CI  [{stats['boot_ci_lo']:+.4f}, "
                     f"{stats['boot_ci_hi']:+.4f}]")
    ax.axvline(stats["boot_ci_hi"], color="tomato", linestyle="--", linewidth=1.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, label="No effect (0)")
    ax.set_xlabel("Final QD Difference  (A2 \u2212 A1)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Per-Seed QD: A2 \u2212 A1\n"
        f"t = {stats['t_stat']:.3f},  p = {stats['t_pval']:.4f}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_rejection_breakdown(all_a2: list,
                              save_path: str = "results/rejection_breakdown.png"):
    """Total A2 rejections by reason, aggregated across all seeds."""
    total_p_rej  = sum(int(jnp.sum(a2.rejection_p_count))  for a2 in all_a2)
    total_es_rej = sum(int(jnp.sum(a2.rejection_es_count)) for a2 in all_a2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["p-value fail\n(\u0070\u0302 \u2265 \u03b1)",
         "Effect size fail\n(\u03b4\u0302 \u2264 \u03b4_min)"],
        [total_p_rej, total_es_rej],
        color=["steelblue", "tomato"],
    )
    ax.set_title(f"A2 HT Gate — Rejection Breakdown\n"
                 f"(N={len(all_a2)} seeds, Stage 2 total)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 65)
    print("  DROME: Sequential Two-Stage Pipeline")
    print(f"  Stage 1 — A1 (naive):    T1={T1} iterations")
    print(f"  Stage 2 — A2 (HT gate):  T2={T2} iterations  "
          f"(\u03b1={ALPHA}, \u03b4_min={DELTA_MIN})")
    print(f"  M={NUM_RATERS}, \u03c3=0.05, Grid={GRID_SIZE}\u00d7{GRID_SIZE}")
    print(f"  N={N_SEEDS} seeds | paired t-test + bootstrap CI")
    print("=" * 65)

    (all_a1, all_a2,
     qd_a1_mat, qd_a2_mat,
     final_m_a1, final_m_a2) = run_multi_seed(
        seeds=SEEDS, t1=T1, t2=T2, verbose=True
    )

    # Statistical comparison
    stats = statistical_comparison(final_m_a1, final_m_a2)

    # A2 HT gate activity (Stage 2 only, aggregated)
    total_att = sum(int(jnp.sum(a2.attempt_counts))     for a2 in all_a2)
    total_acc = sum(int(jnp.sum(a2.acceptance_counts))  for a2 in all_a2)
    total_p   = sum(int(jnp.sum(a2.rejection_p_count))  for a2 in all_a2)
    total_es  = sum(int(jnp.sum(a2.rejection_es_count)) for a2 in all_a2)
    print(f"\n  A2 HT gate (Stage 2, all {N_SEEDS} seeds combined):")
    print(f"    Attempts:          {total_att}")
    print(f"    Acceptances:       {total_acc}  "
          f"({100.*total_acc/(total_att+1e-8):.1f}%)")
    print(f"    Rejected (p fail): {total_p}")
    print(f"    Rejected (ES fail):{total_es}")

    # Diversity on A2 seed 0
    div   = compute_behaviour_diversity(all_a2[0])
    pdist = compute_pairwise_elite_distance(all_a2[0])
    print(f"\n  A2 diversity (seed 0): coverage={div['coverage']:.1f}%  "
          f"BD_spread={div['bd_spread']:.4f}  pairwise_dist={pdist:.4f}")

    # Plots
    plot_qd_curves(qd_a1_mat, qd_a2_mat)
    plot_diff_distribution(final_m_a1, final_m_a2, stats)
    plot_rejection_breakdown(all_a2)

    print("\n\u2713 All results saved to results/")