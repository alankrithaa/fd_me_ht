"""
fdme_map_elites.py
==================
DROMEMAPElites: sequential two-stage pipeline.

Stage 1 — A1 (exploratory, T1 iterations):
    Standard MAP-Elites with naive mean-based replacement.
    Fast, cheap, high-turnover. Runs for T1 iterations to build
    broad coverage of the behavioural space.
    The emitter samples from A1 throughout this stage.

Stage 2 — A2 (robust, T2 iterations):
    A2 is seeded with ALL of A1's occupied elites at the start.
    The emitter then samples from A2 and proposes new candidates.
    Each candidate is evaluated fresh and gated by the three-condition
    HT rule before it can replace an A2 incumbent.
    A1 is NOT updated during Stage 2 — it is frozen after Stage 1.

This is a strictly sequential pipeline:
    A1 runs fully → A1 elites seed A2 → A2 runs fully.
No simultaneous updating. The two archives never run in parallel.

Typical settings  (configurable in main_pipeline.py):
    T1 = 100   cheap exploratory iterations on A1
    T2 = 50    expensive HT-validated iterations on A2
"""

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import Genotype, RNGKey

from repertoire import DistributionalRepertoire


class DROMEMAPElites:
    """
    Sequential two-stage DROME optimiser.

    Public API:

        map_elites = DROMEMAPElites(scoring_fn, emitter, metrics_fn,
                                    alpha=0.10, delta_min=0.55)

        # Stage 1 init — build A1
        a1, emitter_state, key = map_elites.init_a1(init_genotypes, centroids, key)

        # Stage 1 loop — run T1 iterations on A1
        for t in range(T1):
            a1, emitter_state, metrics = map_elites.update_a1(a1, emitter_state, key)
            key, _ = jax.random.split(key)

        # Stage 2 init — seed A2 from A1's elites, reset emitter
        a2, emitter_state, key = map_elites.init_a2(a1, key)

        # Stage 2 loop — run T2 iterations on A2 with HT gate
        for t in range(T2):
            a2, emitter_state, metrics = map_elites.update_a2(a2, emitter_state, key)
            key, _ = jax.random.split(key)

        # Final robust results live in a2
    """

    def __init__(
        self,
        scoring_function,
        emitter:          Emitter,
        metrics_function,
        alpha:            float = 0.10,
        delta_min:        float = 0.55,
    ) -> None:
        self._scoring_fn  = scoring_function
        self._emitter     = emitter
        self._metrics_fn  = metrics_function
        self._alpha       = alpha
        self._delta_min   = delta_min

    # ── Stage 1: init A1 ──────────────────────────────────────────────────────
    def init_a1(
        self,
        genotypes: Genotype,
        centroids: jnp.ndarray,
        key:       RNGKey,
    ) -> Tuple[DistributionalRepertoire, Optional[EmitterState], RNGKey]:
        """
        Score the initial population and build A1 (naive archive).
        Initialise the emitter to sample from A1.

        Returns: (a1, emitter_state, key)
        """
        key, score_key = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_fn(genotypes, score_key)

        a1 = DistributionalRepertoire.init(
            genotypes    = genotypes,
            fitnesses    = fitnesses,
            descriptors  = descriptors,
            centroids    = centroids,
            extra_scores = extra_scores,
            use_ht       = False,           # A1: naive replacement
            alpha        = self._alpha,
            delta_min    = self._delta_min,
        )

        key, emitter_key = jax.random.split(key)
        emitter_state, key = self._emitter.init(
            key          = emitter_key,
            repertoire   = a1,
            genotypes    = genotypes,
            fitnesses    = fitnesses,
            descriptors  = descriptors,
            extra_scores = extra_scores,
        )

        return a1, emitter_state, key

    # ── Stage 1: update A1 ────────────────────────────────────────────────────
    def update_a1(
        self,
        a1:            DistributionalRepertoire,
        emitter_state: Optional[EmitterState],
        key:           RNGKey,
    ) -> Tuple[DistributionalRepertoire, Optional[EmitterState], dict]:
        """
        One iteration of Stage 1.

        Emit offspring from A1, score them, update A1 naively, update emitter.

        Returns: (a1, emitter_state, metrics_on_a1)
        """
        offspring, _ = self._emitter.emit(a1, emitter_state, key)

        key, score_key = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_fn(offspring, score_key)

        a1 = a1.add(
            batch_of_genotypes    = offspring,
            batch_of_descriptors  = descriptors,
            batch_of_fitnesses    = fitnesses,
            batch_of_extra_scores = extra_scores,
        )

        emitter_state = self._emitter.state_update(
            emitter_state = emitter_state,
            repertoire    = a1,
            genotypes     = offspring,
            fitnesses     = fitnesses,
            descriptors   = descriptors,
            extra_scores  = extra_scores,
        )

        metrics = self._metrics_fn(a1)
        return a1, emitter_state, metrics

    # ── Stage 2: init A2 from A1 ──────────────────────────────────────────────
    def init_a2(
        self,
        a1:  DistributionalRepertoire,
        key: RNGKey,
    ) -> Tuple[DistributionalRepertoire, Optional[EmitterState], RNGKey]:
        """
        Seed A2 from all occupied elites in A1.

        Every cell that A1 has filled is transferred directly into A2.
        A2 is then the starting point for Stage 2 HT-gated refinement.
        The emitter is re-initialised to sample from A2.

        A1 is not modified and not used again after this point.

        Returns: (a2, emitter_state, key)
        """
        # Pull all occupied entries from A1
        occupied_mask = a1.fitnesses > -jnp.inf           # (C,)
        occupied_idx  = jnp.where(
            occupied_mask,
            jnp.arange(a1.fitnesses.shape[0]),
            -1,
        )

        # Extract occupied genotypes, descriptors, fitnesses, scores
        # We pass ALL cells and let A2.add() ignore empty ones (fitness == -inf)
        a2 = DistributionalRepertoire.init(
            genotypes    = a1.genotypes,          # (C, D) — all cells
            fitnesses    = a1.fitnesses,           # (C,)   — empty = -inf
            descriptors  = a1.descriptors,         # (C, BD)
            centroids    = a1.centroids,
            extra_scores = {"scores": a1.scores},  # (C, M)
            use_ht       = True,                   # A2: HT-gated replacement
            alpha        = self._alpha,
            delta_min    = self._delta_min,
        )

        # Re-initialise emitter to sample from A2
        key, emitter_key = jax.random.split(key)
        emitter_state, key = self._emitter.init(
            key          = emitter_key,
            repertoire   = a2,
            genotypes    = a1.genotypes,
            fitnesses    = a1.fitnesses,
            descriptors  = a1.descriptors,
            extra_scores = {"scores": a1.scores},
        )

        return a2, emitter_state, key

    # ── Stage 2: update A2 ────────────────────────────────────────────────────
    def update_a2(
        self,
        a2:            DistributionalRepertoire,
        emitter_state: Optional[EmitterState],
        key:           RNGKey,
    ) -> Tuple[DistributionalRepertoire, Optional[EmitterState], dict]:
        """
        One iteration of Stage 2.

        Emit offspring from A2, score them fresh, attempt HT-gated
        replacement in A2, update emitter.

        Returns: (a2, emitter_state, metrics_on_a2)
        """
        offspring, _ = self._emitter.emit(a2, emitter_state, key)

        key, score_key = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_fn(offspring, score_key)

        a2 = a2.add(
            batch_of_genotypes    = offspring,
            batch_of_descriptors  = descriptors,
            batch_of_fitnesses    = fitnesses,
            batch_of_extra_scores = extra_scores,
        )

        emitter_state = self._emitter.state_update(
            emitter_state = emitter_state,
            repertoire    = a2,
            genotypes     = offspring,
            fitnesses     = fitnesses,
            descriptors   = descriptors,
            extra_scores  = extra_scores,
        )

        metrics = self._metrics_fn(a2)
        return a2, emitter_state, metrics