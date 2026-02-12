import jax
import jax.numpy as jnp
from typing import NamedTuple

class DistributionalRepertoire(NamedTuple):
    """A QDAX-style repertoire that stores a distribution of scores per cell."""
    # Main Data
    genotypes: jnp.ndarray      # Latent vectors (Grid_H, Grid_W, D)
    phenotypes: jnp.ndarray     # Image pixels (Grid_H, Grid_W, Res, Res, C)
    scores: jnp.ndarray         # M-rater score history (Grid_H, Grid_W, M)
    fitnesses: jnp.ndarray      # Mean score for quick reference (Grid_H, Grid_W)
    occupancy: jnp.ndarray      # 1 if cell is filled, 0 otherwise (Grid_H, Grid_W)
    
    # BD tracking for distance metrics
    descriptors: jnp.ndarray    # BD coordinates (Grid_H, Grid_W, 2)

    # Experiment Logging (Replacement Dynamics)
    acceptance_counts: jnp.ndarray # How many times cell was successfully replaced
    attempt_counts: jnp.ndarray    # How many times a candidate landed in this cell
    sum_effect_sizes: jnp.ndarray  # Cumulative CLES/Delta for mean calculation
    
    @classmethod
    def init(cls, grid_size, latent_dim, num_raters, img_res=64):
        """Initializes an empty archive."""
        shape = (grid_size, grid_size)
        return cls(
            genotypes=jnp.zeros(shape + (latent_dim,)),
            phenotypes=jnp.zeros(shape + (img_res, img_res, 3)),
            scores=jnp.zeros(shape + (num_raters,)),
            fitnesses=jnp.full(shape, jnp.inf), # Assuming minimization
            occupancy=jnp.zeros(shape, dtype=jnp.int32),
            descriptors=jnp.zeros(shape + (2,)),
            acceptance_counts=jnp.zeros(shape, dtype=jnp.int32),
            attempt_counts=jnp.zeros(shape, dtype=jnp.int32),
            sum_effect_sizes=jnp.zeros(shape)
        )

# Metric: Cohen's d for your tech report conclusion
@jax.jit
def compute_cohens_d(new_scores, old_scores):
    """Calculates standardized difference between distributions."""
    n1, n2 = len(new_scores), len(old_scores)
    var1, var2 = jnp.var(new_scores), jnp.var(old_scores)
    pooled_std = jnp.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (jnp.mean(new_scores) - jnp.mean(old_scores)) / (pooled_std + 1e-8)