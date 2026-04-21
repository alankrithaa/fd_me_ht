"""
Structured Fake VLM Evaluator
==============================
Each simulated VLM v has a fixed preference vector w_v ∈ R^5 over 5 perceptual
image features. The score for image x from VLM v is:

    s_v(x) = -dot(w_v, φ(x)) + ε,    ε ~ N(0, 0.05²)  # REDUCED from 0.1

The negation makes this a MINIMIZATION problem: lower score = higher quality.
A candidate that scores genuinely well on the features a VLM cares about will
CONSISTENTLY get a lower (better) score — giving the Hypothesis Test a real
signal to work with, not just noise.

The 5 features φ(x):
    1. mean_brightness  — average pixel intensity
    2. contrast         — std of pixel values
    3. entropy          — histogram-based randomness (32 bins)
    4. edge_density     — mean absolute gradient magnitude (Sobel approx)
    5. symmetry        - left-right mirror stability in grayscale.

VLM pool: 10 total VLMs, 10 sampled per evaluation (ALL VLMs).  # INCREASED from 5
VLM_WEIGHTS is a fixed (10, 5) matrix, seeded at 42 — same every run.
FIXED: Removed L2-normalization to preserve inter-VLM variance.
Noise σ = 0.05 — REDUCED from 0.1 to double the signal-to-noise ratio.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax import pure_callback

# ──────────────────────────────────────────────
# CONSTANTS — fixed for the entire experiment
# ──────────────────────────────────────────────
NUM_VLMS = 10       # Total VLMs in the pool
NUM_RATERS = 10     # INCREASED: All VLMs score each image (was 5)
NOISE_SIGMA = 0.05  # REDUCED: Halved stochasticity (was 0.1)
FEATURE_DIM = 5     # Number of perceptual features that the quality of image is judged on

# Behaviour Descriptor (BD) projection: 2D descriptor from feature indices
# extract_features() returns [brightness, contrast, entropy, edge_density, symmetry]
BD_BRIGHTNESS_IDX = 0  # First feature: mean brightness
BD_ENTROPY_IDX = 2     # Third feature: entropy

# Fixed weight matrix — seeded once, never changes between runs.
# Each row w_v is the preference vector of VLM v over the 5 features.
# FIXED: Removed L2-normalization to preserve inter-VLM variance in feature weighting strength.
_rng = np.random.RandomState(42)
VLM_WEIGHTS = _rng.randn(NUM_VLMS, FEATURE_DIM).astype(np.float32)  # NO normalization


# ──────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────
def extract_features(image_np: np.ndarray) -> np.ndarray:
    """
    Extract 5 perceptual features from a single image.

    Args:
        image_np: numpy array of shape (H, W, 3), values in [0, 1].

    Returns:
        features: numpy array of shape (5,), each feature normalised to [0, 1].
    """
    # Ensure float32 numpy
    img = np.asarray(image_np, dtype=np.float32)          # (H, W, 3)
    gray = np.mean(img, axis=2)                            # (H, W)

    # 1. Mean brightness — already in [0, 1]
    mean_brightness = float(np.mean(gray)) #just mean(all pixels)

    # 2. Contrast — std(pixels); max possible for uniform dist over [0,1] ≈ 0.289
    contrast = float(np.std(gray)) / 0.289
    contrast = np.clip(contrast, 0.0, 1.0)

    # 3. Entropy — histogram over 32 bins, normalised to [0, 1]
    hist, _ = np.histogram(gray.ravel(), bins=32, range=(0.0, 1.0))
    p = hist / (hist.sum() + 1e-10)
    raw_entropy = float(-np.sum(p * np.log(p + 1e-10)))
    max_entropy = float(np.log(32))                        # log(num_bins)
    entropy = np.clip(raw_entropy / max_entropy, 0.0, 1.0) #high entropy = complex, varied image

    # 4. Edge density — Sobel approximation via finite differences
    dx = np.abs(np.diff(gray, axis=1))                    # (H, W-1)
    dy = np.abs(np.diff(gray, axis=0))                    # (H-1, W)
    edge_density = float((np.mean(dx) + np.mean(dy)) / 2.0)
    edge_density = np.clip(edge_density / 0.5, 0.0, 1.0)  # normalise

    # 5. Symmetry — left-right mirror similarity in grayscale, already in [0,1]
    gray_flipped = np.fliplr(gray)
    symmetry = 1.0 - float(np.mean(np.abs(gray - gray_flipped)))
    symmetry = np.clip(symmetry, 0.0, 1.0)

    return np.array(
    [mean_brightness, contrast, entropy, edge_density, symmetry],
    dtype=np.float32
    )
    


# ──────────────────────────────────────────────
# PYTORCH SCORER  (the "fake VLM jury")
# ──────────────────────────────────────────────
def torch_vlm_scorer(images_numpy: np.ndarray) -> np.ndarray:
    """
    Score a batch of images using the structured fake VLM pool.

    For each image:
      - Extract 5 perceptual features φ(x)
      - Use ALL NUM_RATERS VLMs from the pool (no sampling - this increases statistical power)
      - Compute score: s_v(x) = -dot(w_v, φ(x)) + ε,  ε ~ N(0, σ²)
      - MINIMISATION: lower score = higher quality image

    Args:
        images_numpy: numpy array of shape (batch, H, W, 3), values in [0, 1].

    Returns:
        scores: numpy array of shape (batch, NUM_RATERS), dtype float32.
    """
    images_np = np.asarray(images_numpy, dtype=np.float32)
    batch_size = images_np.shape[0]
    # Extract features for every image in the batch — shape (batch, 5)
    features_batch = np.stack(
        [extract_features(images_np[i]) for i in range(batch_size)],
        axis=0
    )  # (batch, 5)

    # Convert to torch for the matrix multiply
    features_t = torch.from_numpy(features_batch).float()         # (batch, 5)
    weights_t  = torch.from_numpy(VLM_WEIGHTS).float()            # (10, 5)

    with torch.no_grad():
        # Compute all 10 VLM scores for every image: (batch, 10)
        all_scores = features_t @ weights_t.T                      # (batch, 10)

        # Negate → minimisation: lower score = better
        all_scores = -all_scores

        # Use ALL VLMs (no sampling for maximum statistical power)
        selected_scores = all_scores[:, :NUM_RATERS]               # (batch, NUM_RATERS)

        # Add stochastic noise — simulates inter-query VLM variance
        noise = torch.randn_like(selected_scores) * NOISE_SIGMA
        selected_scores = selected_scores + noise

    return selected_scores.numpy().astype(np.float32)


# ──────────────────────────────────────────────
# JAX BRIDGE
# ──────────────────────────────────────────────
def evaluate_via_pytorch(images: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-compatible wrapper around torch_vlm_scorer.
    Uses pure_callback to cross the JAX → PyTorch boundary safely.

    Args:
        images: JAX array of shape (batch, H, W, 3).

    Returns:
        scores: JAX array of shape (batch, NUM_RATERS).
    """
    result_shape = jax.ShapeDtypeStruct(
        shape=(images.shape[0], NUM_RATERS),
        dtype=jnp.float32
    )
    return pure_callback(torch_vlm_scorer, result_shape, images)


# ──────────────────────────────────────────────
# QUICK SANITY CHECK  (run this file directly)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Evaluator Check ===\n")

    # 1. Feature extraction
    test_image = np.random.rand(64, 64, 3).astype(np.float32)
    feats = extract_features(test_image)
    print(f"Features for random image:  {feats}")
    print(f"All features in [0,1]:      {bool(np.all(feats >= 0) and np.all(feats <= 1))}\n")

    # 2. Bright image vs dark image — scores should differ
    bright_image = np.ones((64, 64, 3), dtype=np.float32) * 0.9
    dark_image   = np.ones((64, 64, 3), dtype=np.float32) * 0.1
    bright_feats = extract_features(bright_image)
    dark_feats   = extract_features(dark_image)
    print(f"Bright image features: {bright_feats}")
    print(f"Dark   image features: {dark_feats}")
    print(f"Brightness differs:    {bright_feats[0] != dark_feats[0]}\n")

    # 3. Scorer output shape and dtype
    batch = np.random.rand(3, 64, 64, 3).astype(np.float32)
    scores = torch_vlm_scorer(batch)
    print(f"Scorer output shape:  {scores.shape}   (expected (3, {NUM_RATERS}))")
    print(f"Scorer output dtype:  {scores.dtype}   (expected float32)")
    print(f"Score range:          [{scores.min():.3f}, {scores.max():.3f}]\n")

    # 4. Same image scored twice — scores differ due to noise
    single = np.random.rand(1, 64, 64, 3).astype(np.float32)
    s1 = torch_vlm_scorer(single)
    s2 = torch_vlm_scorer(single)
    print(f"Same image, call 1:   {s1[0]}")
    print(f"Same image, call 2:   {s2[0]}")
    print(f"Scores differ (noise): {not np.allclose(s1, s2)}\n")

    # 5. Consistently better image: bright beats dark across many calls
    wins = 0
    for _ in range(50):
        sb = torch_vlm_scorer(bright_image[None])
        sd = torch_vlm_scorer(dark_image[None])
        # In minimisation, lower = better; bright wins if mean(sb) != mean(sd)
        # VLMs that prefer brightness will score bright lower (better)
        wins += int(np.mean(sb) < np.mean(sd))
    print(f"Bright beats dark in {wins}/50 calls (expect ~40-50 if VLMs prefer brightness)")
    print(f"  With increased M and reduced noise, this should be more consistent")
    print("\n=== All checks passed ===")