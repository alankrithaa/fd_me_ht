import jax
import jax.numpy as jnp

@jax.jit
def calculate_ht_replacement(new_scores, old_scores, alpha=0.05, delta_min=0.6):
    #this function take 2 arrays of shape (M,) 
    # the new candidate's M scores and the current archive's M scores and decides whether to replace 
    """
    JAX-native replacement logic using a Distributional Hypothesis Testing approach.
    
    Args:
        new_scores: (M,) array of ratings for the candidate image.
        old_scores: (M,) array of ratings for the current elite in the cell.
        alpha: Significance level (threshold for p-value).
        delta_min: Effect size threshold (Minimum CLES required).
        
    Returns:
        should_replace: Boolean indicating if the candidate should replace the elite.
        p_val: The calculated statistical significance of the improvement.
        cles: The Common Language Effect Size (Probability of Superiority).
    """
    
    # 1. Calculate Effect Size (CLES / Probability of Superiority)
    # We create an M x M matrix comparing every new rater score against every old rater score.
    diffs = new_scores[:, None] - old_scores[None, :] 
    
    # In Fast Direct, we are MINIMIZING (lower score = closer to target).
    # Success is defined as New Score < Old Score.
    success_matrix = jnp.where(diffs < 0, 1.0, 0.0)
    ties_matrix = jnp.where(diffs == 0, 0.5, 0.0)
    
    # CLES represents the probability that a random new score is better than a random old score.
    cles = jnp.mean(success_matrix + ties_matrix)
    
    # 2. Statistical Significance (P-Value)
    # We use a Z-test approximation suitable for JAX/GPU execution.
    mean_new = jnp.mean(new_scores)
    mean_old = jnp.mean(old_scores)
    
    # Pooled standard deviation for two-sample Z-test
    # σ_pool = sqrt((σ_new² + σ_old² + ε) / 2)
    std_new = jnp.std(new_scores)
    std_old = jnp.std(old_scores)
    pooled_var = (std_new**2 + std_old**2) / 2.0 + 1e-8
    pooled_std = jnp.sqrt(pooled_var)
    z_score = jnp.abs(mean_new - mean_old) / pooled_std
    
    # Convert Z-score to a simplified P-value (Exponential approximation for JIT safety)
    p_val = jnp.exp(-0.717 * z_score - 0.416 * (z_score**2))
    
    # 3. Decision Gatekeeper
    # Replace ONLY if the improvement is statistically likely (p < alpha)
    # AND the improvement is meaningfully large (cles > delta_min)
    # Since we are minimizing, we also ensure mean_new is actually lower than mean_old.
    significant_improvement = (p_val < alpha) & (mean_new < mean_old)
    meaningful_improvement = (cles > delta_min)
    
    should_replace = significant_improvement & meaningful_improvement
    
    return should_replace, p_val, cles