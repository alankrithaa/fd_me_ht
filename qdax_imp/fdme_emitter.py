from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.emitters.emitter import Emitter
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class FDMEEmitter(Emitter):
    """
    Simple MAP-Elites-style emitter for FD-ME-HT.

    Behaviour:
    - If the repertoire already contains elites, sample parents from occupied cells
      and mutate them with Gaussian noise.
    - Otherwise, or when mutation is not selected, sample fresh genotypes from
      a standard normal prior.

    IMPORTANT:
    - QDAX expects emit() to return exactly:
          (genotypes, extra_info)
      not (genotypes, extra_info, key).
    """

    def __init__(
        self,
        batch_size: int,
        latent_dim: int,
        mutation_prob: float = 0.5,
        mutation_sigma: float = 0.1,
    ) -> None:
        self._batch_size = batch_size
        self._latent_dim = latent_dim
        self._mutation_prob = mutation_prob
        self._mutation_sigma = mutation_sigma

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def use_all_data(self) -> bool:
        return False

    def init(
        self,
        key: RNGKey,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> Tuple[None, RNGKey]:
        return None, key

    def emit(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[None],
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """
        Emit a batch of offspring genotypes.

        Returns:
            offspring: shape (batch_size, latent_dim)
            extra_info: dict
        """
        key_sel, key_noise, key_fresh, key_mode = jax.random.split(key, 4)

        occupied = repertoire.fitnesses > -jnp.inf
        has_elites = jnp.any(occupied)
        num_cells = repertoire.fitnesses.shape[0]

        probs = occupied.astype(jnp.float32)
        probs = jnp.where(
            has_elites,
            probs / (jnp.sum(probs) + 1e-8),
            jnp.ones_like(probs, dtype=jnp.float32) / num_cells,
        )

        elite_indices = jax.random.choice(
            key_sel,
            a=num_cells,
            shape=(self._batch_size,),
            replace=True,
            p=probs,
        )
        elite_genotypes = repertoire.genotypes[elite_indices]

        mutated = elite_genotypes + self._mutation_sigma * jax.random.normal(
            key_noise,
            shape=elite_genotypes.shape,
            dtype=jnp.float32,
        )

        fresh = jax.random.normal(
            key_fresh,
            shape=(self._batch_size, self._latent_dim),
            dtype=jnp.float32,
        )

        mutate_mask = jax.random.bernoulli(
            key_mode,
            p=self._mutation_prob,
            shape=(self._batch_size, 1),
        )

        offspring = jnp.where(has_elites & mutate_mask, mutated, fresh)

        extra_info = {}
        return offspring, extra_info

    def state_update(
        self,
        emitter_state: Optional[None],
        repertoire: MapElitesRepertoire,
        genotypes: Optional[Genotype],
        fitnesses: Fitness,
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> Optional[None]:
        return emitter_state