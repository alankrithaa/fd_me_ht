from __future__ import annotations

from typing import Optional, Tuple

from qdax.core.map_elites import MAPElites
from qdax.core.emitters.emitter import EmitterState
from qdax.custom_types import Centroid, Genotype, RNGKey

from repertoire import DistributionalRepertoire


class FDMEHTMAPElites(MAPElites):
    """
    Thin subclass of the real QDAX MAPElites class.

    QDAX's MAPElites.update() is used unchanged.
    We override only init() so that the algorithm uses our custom
    DistributionalRepertoire instead of the default QDAX repertoire.
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