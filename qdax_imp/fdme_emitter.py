from __future__ import annotations
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from qdax.core.emitters.emitter import Emitter
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class FDMEEmitter(Emitter):
    def __init__(self, batch_size, latent_dim, mutation_prob=0.5, mutation_sigma=0.3):
        self._batch_size = batch_size
        self._latent_dim = latent_dim
        self._mutation_prob = mutation_prob
        self._mutation_sigma = mutation_sigma

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def use_all_data(self):
        return False

    def init(self, key, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        return None, key

    def emit(self, repertoire, emitter_state, key):
        key_sel, key_noise, key_fresh, key_mode = jax.random.split(key, 4)
        occupied = repertoire.fitnesses > -jnp.inf
        has_elites = jnp.any(occupied)
        num_cells = repertoire.fitnesses.shape[0]
        probs = jnp.where(
            has_elites,
            occupied.astype(jnp.float32) / (jnp.sum(occupied.astype(jnp.float32)) + 1e-8),
            jnp.ones(num_cells, dtype=jnp.float32) / num_cells)
        elite_indices = jax.random.choice(
            key_sel, a=num_cells, shape=(self._batch_size,), replace=True, p=probs)
        elite_genotypes = repertoire.genotypes[elite_indices]
        mutated = elite_genotypes + self._mutation_sigma * jax.random.normal(
            key_noise, shape=elite_genotypes.shape, dtype=jnp.float32)
        fresh = jax.random.normal(
            key_fresh, shape=(self._batch_size, self._latent_dim), dtype=jnp.float32)
        mutate_mask = jax.random.bernoulli(
            key_mode, p=self._mutation_prob, shape=(self._batch_size, 1))
        offspring = jnp.where(has_elites & mutate_mask, mutated, fresh)
        return offspring, {}

    def state_update(self, emitter_state, repertoire, genotypes, fitnesses,
                     descriptors, extra_scores):
        return emitter_state