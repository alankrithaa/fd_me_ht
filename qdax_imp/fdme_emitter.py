"""
fdme_emitter.py
===============
FDMEEmitter — offspring generator for the DROME two-archive pipeline.

Always samples parents from A1 (the NaiveRepertoire), which is the more
populated and rapidly-updated archive. This ensures the emitter has rich
coverage to draw from, independent of how conservative A2's HT gate is.

Behaviour:
  - If A1 contains any elites, sample from occupied cells and apply
    Gaussian mutation with probability mutation_prob.
  - Otherwise (A1 empty at init), sample fresh genotypes from N(0, I).
"""

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.emitters.emitter import Emitter
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class FDMEEmitter(Emitter):

    def __init__(
        self,
        batch_size:     int,
        latent_dim:     int,
        mutation_prob:  float = 0.5,
        mutation_sigma: float = 0.1,
    ) -> None:
        self._batch_size     = batch_size
        self._latent_dim     = latent_dim
        self._mutation_prob  = mutation_prob
        self._mutation_sigma = mutation_sigma

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def use_all_data(self) -> bool:
        return False

    def init(
        self,
        key:          RNGKey,
        repertoire:   MapElitesRepertoire,
        genotypes:    Genotype,
        fitnesses:    Fitness,
        descriptors:  Descriptor,
        extra_scores: ExtraScores,
    ) -> Tuple[None, RNGKey]:
        return None, key

    def emit(
        self,
        repertoire:    MapElitesRepertoire,   # should be A1
        emitter_state: Optional[None],
        key:           RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """
        Emit a batch of offspring.
        Parents sampled from A1; mutated or fresh based on mutation_prob.
        """
        key_sel, key_noise, key_fresh, key_mode = jax.random.split(key, 4)

        occupied   = repertoire.fitnesses > -jnp.inf
        has_elites = jnp.any(occupied)
        num_cells  = repertoire.fitnesses.shape[0]

        # Sampling distribution over cells
        probs = jnp.where(
            has_elites,
            occupied.astype(jnp.float32) / (jnp.sum(occupied.astype(jnp.float32)) + 1e-8),
            jnp.ones(num_cells, dtype=jnp.float32) / num_cells,
        )

        # Select parents
        elite_indices  = jax.random.choice(
            key_sel, a=num_cells, shape=(self._batch_size,), replace=True, p=probs
        )
        elite_genotypes = repertoire.genotypes[elite_indices]

        # Mutated candidates
        mutated = elite_genotypes + self._mutation_sigma * jax.random.normal(
            key_noise, shape=elite_genotypes.shape, dtype=jnp.float32
        )

        # Fresh candidates from prior
        fresh = jax.random.normal(
            key_fresh, shape=(self._batch_size, self._latent_dim), dtype=jnp.float32
        )

        # Choose mutation vs fresh per candidate
        mutate_mask = jax.random.bernoulli(
            key_mode, p=self._mutation_prob, shape=(self._batch_size, 1)
        )
        offspring = jnp.where(has_elites & mutate_mask, mutated, fresh)

        return offspring, {}

    def state_update(
        self,
        emitter_state: Optional[None],
        repertoire:    MapElitesRepertoire,
        genotypes:     Optional[Genotype],
        fitnesses:     Fitness,
        descriptors:   Optional[Descriptor],
        extra_scores:  ExtraScores,
    ) -> Optional[None]:
        return emitter_state