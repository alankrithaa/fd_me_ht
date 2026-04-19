"""
main_pipeline.py
================
FD-ME-HT pipeline using the ACTUAL QDAX MAPElites class for the update loop.

Key point:
- We instantiate FDMEHTMAPElites, which subclasses QDAX MAPElites.
- We override only init() in the subclass.
- The iteration loop uses QDAX MAPElites.update() unchanged.
"""

from __future__ import annotations

import os
import functools
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import pure_callback, ShapeDtypeStruct

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.utils.metrics import default_qd_metrics

from evaluator import (
    BD_BRIGHTNESS_IDX,
    BD_ENTROPY_IDX,
    evaluate_via_pytorch,
    extract_features,
)
from fdme_emitter import FDMEEmitter
from fdme_map_elites import FDMEHTMAPElites
from diversity_metrics import compute_behaviour_diversity, compute_pairwise_elite_distance
from visualize_metrics import plot_experiment_heatmaps


# -------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------
GRID_SIZE = 5
LATENT_DIM = 8
NUM_RATERS = 5
ITERATIONS = 100
BATCH_SIZE = 4
INIT_BATCH_SIZE = 8
IMG_RES = 64
SAMPLER_K = 5

ALPHA = 0.05
DELTA_MIN = 0.6
MUTATION_PROB = 0.5
MUTATION_SIGMA = 0.1
SEED = 42

#all algorithm inputs are above. 
# -------------------------------------------------------------------
# Descriptor callback: brightness + entropy
# -------------------------------------------------------------------
def _numpy_descriptor_batch(images_np: np.ndarray) -> np.ndarray:
    feats = np.stack([extract_features(img) for img in images_np], axis=0)  # (B, 5)
    return feats[:, [BD_BRIGHTNESS_IDX, BD_ENTROPY_IDX]].astype(np.float32)  # brightness, entropy


def descriptor_via_callback(images: jnp.ndarray) -> jnp.ndarray:
    out_shape = ShapeDtypeStruct((images.shape[0], 2), jnp.float32)
    return pure_callback(_numpy_descriptor_batch, out_shape, images)

#extracts brightness and entropy from a batch of images using extract_features() from evaluator.py, which is a numpy function. We use pure_callback to safely call it from JAX code. The resulting descriptors are (B, 2) arrays of brightness and entropy for each image in the batch.
# -------------------------------------------------------------------
# JAX-native simulated sampler
# -------------------------------------------------------------------
def simulate_sampler_batch(genotypes: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """
    JAX-friendly fake diffusion sampler.
    Input: (B, D)
    Output: (B, H, W, 3)
    """
    #takes a batch of genotypes (latent vectors), produces a batch of images. The mean of each genotype controls base brightness. K=5 rounds of decaying noise added on top. 
    #This is the simulated diffusion sampler. in a real sustem this would be stable diffusion + GNSO guidance. 


    batch_size = genotypes.shape[0]

    base = jnp.clip(jnp.mean(genotypes, axis=1) * 0.5 + 0.5, 0.05, 0.95)
    x = jnp.broadcast_to(base[:, None, None, None], (batch_size, IMG_RES, IMG_RES, 3))

    keys = jax.random.split(key, SAMPLER_K)
    for k in range(SAMPLER_K):
        noise = jax.random.normal(keys[k], shape=x.shape, dtype=jnp.float32)
        x = x + noise / (k + 2)

    return jnp.clip(x, 0.0, 1.0)


# -------------------------------------------------------------------
# QDAX scoring function signature:
#   (genotypes, key) -> (fitnesses, descriptors, extra_scores, key)
# -------------------------------------------------------------------
def scoring_function(genotypes: jnp.ndarray, key: jax.Array):
    key_sampler, key_out = jax.random.split(key)

    images = simulate_sampler_batch(genotypes, key_sampler)              # (B, H, W, 3)
    descriptors = descriptor_via_callback(images)                        # (B, 2)
    scores = evaluate_via_pytorch(images)                                # (B, M)

    fitnesses = -jnp.mean(scores, axis=1)  # QDAX maximizes fitness

    extra_scores = {
        "scores": scores,
    }

    return fitnesses, descriptors, extra_scores


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def compute_metrics(repertoire):
    empty = repertoire.fitnesses == -jnp.inf
    qd_score = float(jnp.sum(repertoire.fitnesses, where=~empty))
    coverage = float(100.0 * jnp.mean((~empty).astype(jnp.float32)))
    max_fitness = float(jnp.max(jnp.where(~empty, repertoire.fitnesses, -jnp.inf)))
    return {
        "qd_score": qd_score,
        "coverage": coverage,
        "max_fitness": max_fitness,
    }


# -------------------------------------------------------------------
# Run one experiment
# -------------------------------------------------------------------
def run_pipeline(use_ht: bool, seed: int = SEED, iterations: int = ITERATIONS):
    key = jax.random.key(seed)

    # Initial population
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.normal(
        subkey,
        shape=(INIT_BATCH_SIZE, LATENT_DIM),
        dtype=jnp.float32,
    )

    # Centroids
    centroids = compute_euclidean_centroids(
        grid_shape=(GRID_SIZE, GRID_SIZE),
        minval=0.0,
        maxval=1.0,
    )

    # Emitter
    emitter = FDMEEmitter(
        batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM,
        mutation_prob=MUTATION_PROB,
        mutation_sigma=MUTATION_SIGMA,
    )

    # Metrics fn
    metrics_fn = compute_metrics

    # ACTUAL MAPElites object
    map_elites = FDMEHTMAPElites(
        scoring_function=scoring_function,
        emitter=emitter,
        metrics_function=metrics_fn,
        use_ht=use_ht,
        alpha=ALPHA,
        delta_min=DELTA_MIN,
    )

    # Init repertoire via our subclass
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, key = map_elites.init(
        init_genotypes,
        centroids,
        subkey,
    )

    qd_history = []
    metrics_history = []

    print(f"\n{'='*60}")
    print(f"  FD-ME-HT with QDAX MAPElites.update() | use_ht={use_ht}")
    print(f"{'='*60}")

    for t in range(iterations):
        repertoire, emitter_state, metrics = map_elites.update(
            repertoire,
            emitter_state,
            key,
        )

        metrics_py = {k: float(v) for k, v in metrics.items()}
        qd_history.append(metrics_py["qd_score"])
        metrics_history.append(metrics_py)

        if t % 20 == 0 or t == iterations - 1:
            print(
                f"iter {t:3d} | coverage {metrics_py['coverage']:.1f}% | "
                f"QD {metrics_py['qd_score']:.4f} | max_fit {metrics_py['max_fitness']:.4f}"
            )

    return repertoire, qd_history, metrics_history


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    rep_ht, qd_ht, metrics_ht = run_pipeline(use_ht=True, seed=SEED)
    rep_naive, qd_naive, metrics_naive = run_pipeline(use_ht=False, seed=SEED)

    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)

    for label, rep, qd_hist in [
        ("HT", rep_ht, qd_ht),
        ("Naive", rep_naive, qd_naive),
    ]:
        div = compute_behaviour_diversity(rep)
        pdist = compute_pairwise_elite_distance(rep)
        m = compute_metrics(rep)

        print(f"\n[{label}]")
        print(f"  QD Score:               {qd_hist[-1]:.4f}")
        print(f"  Coverage:               {m['coverage']:.1f}%")
        print(f"  Max Fitness:            {m['max_fitness']:.4f}")
        print(f"  BD Spread:              {div['bd_spread']:.4f}")
        print(f"  Pairwise Elite Dist:    {pdist:.4f}")

        if label == "HT":
            print(f"  Rejected (p-value):     {int(jnp.sum(rep.rejection_p_count))}")
            print(f"  Rejected (effect size): {int(jnp.sum(rep.rejection_es_count))}")

    # Plot 1: QD score curve
    plt.figure(figsize=(9, 5))
    plt.plot(qd_ht, label="FD-ME-HT (HT gate)", linewidth=2)
    plt.plot(qd_naive, label="Naive baseline", linewidth=2, linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("QD Score")
    plt.title("QD Score vs Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/qd_curve_comparison.png", dpi=150)
    plt.close()

    # Plot 2: rejection breakdown
    p_rej = int(jnp.sum(rep_ht.rejection_p_count))
    es_rej = int(jnp.sum(rep_ht.rejection_es_count))

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["p-value fail", "effect size fail"],
        [p_rej, es_rej],
    )
    plt.title("HT Rejection Breakdown")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/rejection_breakdown.png", dpi=150)
    plt.close()

    # Plot 3: heatmaps
    plot_experiment_heatmaps(rep_ht, grid_size=GRID_SIZE)

    print("\nSaved all results in results/")