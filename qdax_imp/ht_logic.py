"""
ht_logic.py
===========
JAX-native Hypothesis Testing gate for DROME archive replacement.

Implements Algorithm 2 (HTGate) from the paper exactly.

P-VALUE FIX:
    Uses the exact two-tailed Gaussian CDF via jax.scipy.stats.norm.cdf,
    which is fully JIT-compatible. Replaces the previous exponential approximation.

        p = 2 * (1 - Phi(z))

Z-test uses the correct Standard Error of the difference between two means (Eq. 12):

        SE = sqrt(var_new/M + var_old/M)
        z  = |mean_new - mean_old| / SE

Acceptance rule (Eq. 15):
    accept  iff  (mean_new < mean_old)   [directional improvement]
              AND (p < alpha)             [statistically significant, p exact]
              AND (cles > delta_min)      [practically meaningful]
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm


@jax.jit
def calculate_ht_replacement(
    new_scores: jnp.ndarray,
    old_scores: jnp.ndarray,
    alpha: float = 0.10,
    delta_min: float = 0.55,
):
    """
    Hypothesis-testing replacement gate for DROME's robust archive (A2).

    Args:
        new_scores:  (M,) ratings for the challenger image. Lower = better.
        old_scores:  (M,) ratings for the current archive elite.
        alpha:       Significance threshold (default 0.10 per paper §6.3).
        delta_min:   CLES floor for practical significance (default 0.55).

    Returns:
        should_replace: bool — True if challenger should replace the elite.
        p_val:          Exact two-tailed Gaussian p-value.
        cles:           Common Language Effect Size (Eq. 5).
    """

    # ── 1. CLES — Common Language Effect Size (Eq. 5) ────────────────────────
    # diffs[i,j] = new_i - old_j.  Success (new is better) means diffs < 0.
    diffs   = new_scores[:, None] - old_scores[None, :]     # (M, M)
    success = jnp.where(diffs < 0,  1.0, 0.0)
    ties    = jnp.where(diffs == 0, 0.5, 0.0)
    cles    = jnp.mean(success + ties)                       # (1/M²)·ΣΣ[...]

    # ── 2. Two-sample Z-test with correct SE (Eq. 12) ────────────────────────
    mean_new = jnp.mean(new_scores)
    mean_old = jnp.mean(old_scores)
    var_new  = jnp.var(new_scores)                           # σ̂²_new
    var_old  = jnp.var(old_scores)                           # σ̂²_old

    M  = new_scores.shape[0]
    # Standard error of the difference between two sample means:
    #   SE = sqrt(σ²_new/M + σ²_old/M)
    se = jnp.sqrt(var_new / M + var_old / M + 1e-10)

    z_score = jnp.abs(mean_new - mean_old) / se

    # Exact two-tailed p-value via the Gaussian CDF (fully JIT-compatible).
    #   p = 2·(1 − Φ(z))
    # jax.scipy.stats.norm.cdf is equivalent to scipy.stats.norm.cdf
    # and compiles through XLA without any Python-side callbacks.
    p_val = 2.0 * (1.0 - jax_norm.cdf(z_score))

    # ── 3. Three-condition gate (Eq. 15) ─────────────────────────────────────
    directional = mean_new < mean_old                        # condition 1
    significant = p_val    < alpha                           # condition 2
    meaningful  = cles     > delta_min                       # condition 3

    should_replace = directional & significant & meaningful

    return should_replace, p_val, cles