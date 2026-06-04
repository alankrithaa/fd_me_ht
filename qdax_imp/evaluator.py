"""
evaluator.py  (ALIGNED with ht_proof.py)
========================================
Structured synthetic VLM evaluator matching paper Eq. 1 EXACTLY.

KEY DESIGN (this is what makes score-inflation measurable):
  - genotype -> features is DETERMINISTIC: psi(g) = sigmoid(g @ GENO_TO_FEAT)
  - the ONLY stochastic source is rater noise eps ~ N(0, sigma^2)
  - this means every genotype has a well-defined TRUE quality,
    enabling analytical ground-truth re-evaluation.

Rating model (paper Eq. 1):
    r_v(x) = -w_v^T psi(x) + eps,   eps ~ N(0, sigma^2)
Lower score = better. Fitness = -mean(scores), higher = better.

VLM pool: NV = 10, M = NUM_RATERS = 10 (all VLMs rate every candidate).
Weights:  w_v = |N(0,I)|, L2-normalised per row (fixed, seed 42).
"""

import numpy as np
import jax
import jax.numpy as jnp

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_VLMS    = 10
NUM_RATERS  = 10          # M
NOISE_SIGMA = 0.05        # baseline sigma (swept in experiments)
FEATURE_DIM = 5
LATENT_DIM  = 8

# Behaviour descriptor indices into the feature vector
BD_BRIGHTNESS_IDX = 0
BD_ENTROPY_IDX    = 2

# ── Fixed random components (seed 42) ──────────────────────────────────────────
_rng = np.random.RandomState(42)

# VLM preference weights: |N(0,I)| then L2-normalised per row
_raw         = np.abs(_rng.randn(NUM_VLMS, FEATURE_DIM)).astype(np.float32)
_norms       = np.linalg.norm(_raw, axis=1, keepdims=True)
VLM_WEIGHTS  = _raw / (_norms + 1e-10)               # (10, 5), unit rows

# Deterministic genotype -> feature projection (fixed)
GENO_TO_FEAT = (_rng.randn(LATENT_DIM, FEATURE_DIM).astype(np.float32) * 0.5)

# JAX copies
VLM_WEIGHTS_J  = jnp.array(VLM_WEIGHTS)
GENO_TO_FEAT_J = jnp.array(GENO_TO_FEAT)


# ── Deterministic feature map ──────────────────────────────────────────────────
def genotype_to_features(genotypes: jnp.ndarray) -> jnp.ndarray:
    """psi(g) in [0,1]^5. Deterministic. Shape (B, 5)."""
    return jax.nn.sigmoid(genotypes @ GENO_TO_FEAT_J)


# ── Analytical ground truth ────────────────────────────────────────────────────
def true_fitness(genotypes: jnp.ndarray) -> jnp.ndarray:
    """Exact true fitness (no noise). Higher = better. Shape (B,)."""
    feats = genotype_to_features(genotypes)
    return jnp.mean(feats @ VLM_WEIGHTS_J.T, axis=1)


# ── Stochastic scoring (the noisy evaluator) ───────────────────────────────────
def score_genotypes(genotypes: jnp.ndarray, key, sigma: float = NOISE_SIGMA):
    """
    Stochastic evaluation matching Eq. 1.
    Returns (fitnesses, descriptors, {"scores": scores}).
      scores  : (B, M)  lower = better
      fitness : (B,)    -mean(scores), higher = better
    """
    feats = genotype_to_features(genotypes)               # (B, 5)
    clean = -(feats @ VLM_WEIGHTS_J.T)                    # (B, 10), lower = better
    noise = jax.random.normal(key, clean.shape) * sigma
    scores = clean + noise
    fitnesses = -jnp.mean(scores, axis=1)
    descriptors = feats[:, jnp.array([BD_BRIGHTNESS_IDX, BD_ENTROPY_IDX])]
    return fitnesses, descriptors, {"scores": scores}


# ── Backwards-compat: extract_features (used by image-based diversity helpers) ──
def extract_features(image_np: np.ndarray) -> np.ndarray:
    """Retained for any image-based analysis. NOT used by the scoring path."""
    img  = np.asarray(image_np, dtype=np.float32)
    gray = np.mean(img, axis=2)
    brightness = float(np.mean(gray))
    contrast = float(np.clip(np.std(gray) / 0.289, 0.0, 1.0))
    hist, _ = np.histogram(gray.ravel(), bins=32, range=(0.0, 1.0))
    p = hist / (hist.sum() + 1e-10)
    entropy = float(np.clip(-np.sum(p * np.log(p + 1e-10)) / np.log(32), 0.0, 1.0))
    dx = np.abs(np.diff(gray, axis=1)); dy = np.abs(np.diff(gray, axis=0))
    edge = float(np.clip((np.mean(dx) + np.mean(dy)) / 2.0 / 0.5, 0.0, 1.0))
    sym = float(np.clip(1.0 - np.mean(np.abs(gray - np.fliplr(gray))), 0.0, 1.0))
    return np.array([brightness, contrast, entropy, edge, sym], dtype=np.float32)


# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Evaluator sanity check (deterministic) ===")
    norms = np.linalg.norm(VLM_WEIGHTS, axis=1)
    print(f"Weight row norms (all ~1.0): {norms.round(4)}")
    key = jax.random.key(0)
    g = jax.random.normal(key, (4, LATENT_DIM))
    f, d, s = score_genotypes(g, key)
    tf = true_fitness(g)
    print(f"apparent fitness: {np.array(f).round(3)}")
    print(f"true     fitness: {np.array(tf).round(3)}")
    print(f"scores shape: {s['scores'].shape}  desc shape: {d.shape}")
    print("=== OK ===")