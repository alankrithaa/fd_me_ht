"""
evaluator.py
============
Structured Fake VLM Evaluator.

Each simulated VLM v has a fixed preference vector w_v in R^5 over 5 perceptual
image features. The score for image x from VLM v is:

    s_v(x) = -dot(w_v, psi(x)) + eps,    eps ~ N(0, 0.05^2)

The negation makes this a MINIMIZATION problem: lower score = higher quality.

The 5 features psi(x):
    0. mean_brightness  -- average pixel intensity
    1. contrast         -- std of pixel values (normalised)
    2. entropy          -- histogram Shannon entropy (32 bins, normalised)
    3. edge_density     -- mean absolute gradient magnitude (Sobel approx)
    4. symmetry         -- left-right mirror similarity

VLM pool: NV=10 VLMs, all 10 used per evaluation (M=NUM_RATERS=10).
Noise: sigma=0.05.

VLM WEIGHTS: w_v = |N(0,I)|, then L2-normalised per row.
  - Absolute value: all preferences are non-negative (each VLM 'cares about'
    each feature to some positive degree, varying only in magnitude).
  - L2-normalisation: all VLMs contribute on the same score scale.
  Matches paper Sections 1 and 6.1 exactly.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax import pure_callback

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_VLMS    = 10
NUM_RATERS  = 10       # M: all VLMs score every image
NOISE_SIGMA = 0.05
FEATURE_DIM = 5

# Behaviour descriptor indices into extract_features() output
BD_BRIGHTNESS_IDX = 0   # mean brightness
BD_ENTROPY_IDX    = 2   # entropy

# ── VLM weight matrix (fixed, seed 42) ────────────────────────────────────────
# w_v = |N(0,I)| then L2-normalised per row.
_rng         = np.random.RandomState(42)
_raw         = np.abs(_rng.randn(NUM_VLMS, FEATURE_DIM)).astype(np.float32)
_norms       = np.linalg.norm(_raw, axis=1, keepdims=True)
VLM_WEIGHTS  = _raw / (_norms + 1e-10)          # shape (10, 5), rows are unit vectors


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_features(image_np: np.ndarray) -> np.ndarray:
    """
    Extract 5 perceptual features from a single (H, W, 3) float32 image in [0,1].
    Returns an (5,) float32 array, each value in [0,1].
    """
    img  = np.asarray(image_np, dtype=np.float32)
    gray = np.mean(img, axis=2)                              # (H, W)

    # 0. Brightness
    brightness = float(np.mean(gray))

    # 1. Contrast  (max std of uniform on [0,1] ≈ 0.289)
    contrast = float(np.clip(np.std(gray) / 0.289, 0.0, 1.0))

    # 2. Entropy  (32-bin histogram, normalised by log(32))
    hist, _ = np.histogram(gray.ravel(), bins=32, range=(0.0, 1.0))
    p       = hist / (hist.sum() + 1e-10)
    entropy = float(np.clip(-np.sum(p * np.log(p + 1e-10)) / np.log(32), 0.0, 1.0))

    # 3. Edge density  (Sobel approx, normalised by 0.5)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge = float(np.clip((np.mean(dx) + np.mean(dy)) / 2.0 / 0.5, 0.0, 1.0))

    # 4. Symmetry  (left-right mirror)
    sym = float(np.clip(1.0 - np.mean(np.abs(gray - np.fliplr(gray))), 0.0, 1.0))

    return np.array([brightness, contrast, entropy, edge, sym], dtype=np.float32)


# ── PyTorch scorer ────────────────────────────────────────────────────────────
def torch_vlm_scorer(images_numpy: np.ndarray) -> np.ndarray:
    """
    Score a batch of images using the full VLM pool.

    Args:
        images_numpy: (B, H, W, 3) float32 array in [0,1].

    Returns:
        scores: (B, NUM_RATERS) float32. Lower = better quality.
    """
    imgs  = np.asarray(images_numpy, dtype=np.float32)
    B     = imgs.shape[0]
    feats = np.stack([extract_features(imgs[i]) for i in range(B)], axis=0)  # (B, 5)

    ft = torch.from_numpy(feats).float()                 # (B, 5)
    wt = torch.from_numpy(VLM_WEIGHTS).float()           # (10, 5)

    with torch.no_grad():
        scores = -(ft @ wt.T)                            # (B, 10), negated -> minimise
        scores = scores + torch.randn_like(scores) * NOISE_SIGMA

    return scores[:, :NUM_RATERS].numpy().astype(np.float32)


# ── JAX bridge ────────────────────────────────────────────────────────────────
def evaluate_via_pytorch(images: jnp.ndarray) -> jnp.ndarray:
    """JAX pure_callback wrapper around torch_vlm_scorer."""
    shape = jax.ShapeDtypeStruct((images.shape[0], NUM_RATERS), jnp.float32)
    return pure_callback(torch_vlm_scorer, shape, images)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Evaluator sanity check ===")
    norms = np.linalg.norm(VLM_WEIGHTS, axis=1)
    print(f"Weight row norms (all ~1.0): {norms.round(4)}")
    print(f"All weights >= 0: {bool(np.all(VLM_WEIGHTS >= 0))}")

    batch  = np.random.rand(4, 64, 64, 3).astype(np.float32)
    scores = torch_vlm_scorer(batch)
    print(f"Scores shape: {scores.shape}  (expected (4, {NUM_RATERS}))")
    print(f"Score range:  [{scores.min():.3f}, {scores.max():.3f}]")
    print("=== OK ===")