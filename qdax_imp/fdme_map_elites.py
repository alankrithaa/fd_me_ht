from __future__ import annotations

from typing import Optional, Tuple

from qdax.core.map_elites import MAPElites
from qdax.core.emitters.emitter import EmitterState
from qdax.custom_types import Centroid, Genotype, RNGKey

from repertoire import DistributionalRepertoire


class FDMEHTMAPElites(MAPElites):
    """
    Thin subclass of the real QDAX MAPElites class.

    QDAX's MAPElites.update() is used unchanged, but we override init() to use
    our custom DistributionalRepertoire instead of the default QDAX repertoire.
    
    UPDATED: Now tracks iteration count to support exploration window in repertoire.
    """

    def __init__(
        self,
        scoring_function,
        emitter,
        metrics_function,
        use_ht: bool = True,
        alpha: float = 0.05,
        delta_min: float = 0.6,
    ) -> None:
        super().__init__(
            scoring_function=scoring_function,
            emitter=emitter,
            metrics_function=metrics_function,
        )
        self._use_ht = use_ht
        self._alpha = alpha
        self._delta_min = delta_min
        self._current_iteration = 0  # NEW: Track iteration for exploration window

    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        key: RNGKey,
    ) -> Tuple[DistributionalRepertoire, Optional[EmitterState], RNGKey]:
        fitnesses, descriptors, extra_scores = self._scoring_function(genotypes, key)

        repertoire = DistributionalRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
            use_ht=self._use_ht,
            alpha=self._alpha,
            delta_min=self._delta_min,
        )

        emitter_state, key = self._emitter.init(
            key=key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, key

    def update(
        self,
        repertoire: DistributionalRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ):
        """
        Override update() to inject iteration count into repertoire updates.
        This enables the exploration window bypass in repertoire.add().
        """
        # Emit new offspring
        offspring, extra_info = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # Score offspring
        fitnesses, descriptors, extra_scores = self._scoring_function(
            offspring, random_key
        )

        # Add to repertoire WITH ITERATION COUNT for exploration window
        repertoire = repertoire.add(
            batch_of_genotypes=offspring,
            batch_of_descriptors=descriptors,
            batch_of_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            exploration_iter=self._current_iteration,  # NEW: Pass iteration count
        )

        # Update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=offspring,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # Compute metrics
        metrics = self._metrics_function(repertoire)

        # Increment iteration counter
        self._current_iteration += 1

        return repertoire, emitter_state, metrics