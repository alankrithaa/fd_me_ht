import jax
import jax.numpy as jnp

@jax.jit
def calculate_ht_replacement(new_scores, old_scores, alpha=0.05, delta_min=0.6):
    """
    JAX-native replacement logic using a simplified Rank-Sum approach.
    new_scores: (M,) array of ratings for candidate
    old_scores: (M,) array of ratings for current elite
    """
    # 1. Calculate Effect Size (CLES / Probability of Superiority)
    # We compare every rater against every rater
    diffs = new_scores[:, None] - old_scores[None, :]
    # Probability that a random new score is 'better' (lower in your case) than old
    # Using < because you mentioned minimizing in your FD setup
    success_matrix = jnp.where(diffs < 0, 1.0, 0.0)
    ties_matrix = jnp.where(diffs == 0, 0.5, 0.0)
    
    cles = jnp.mean(success_matrix + ties_matrix)
    
    # 2. Simplified Significance (Z-test approximation for Rank-Sum)
    # For small M, we can use a simpler mean comparison or a t-test
    mean_new = jnp.mean(new_scores)
    mean_old = jnp.mean(old_scores)
    
    # Replacement Rule: Significant improvement AND meaningful effect size
    # In JAX, we return a boolean mask
    should_replace = (mean_new < mean_old) & (cles > delta_min)
    
    return should_replace, cles