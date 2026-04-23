from __future__ import annotations
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import Genotype, RNGKey
from repertoire import DistributionalRepertoire

class DROMEMAPElites:
    def __init__(
        self,
        scoring_function,
        emitter: Emitter,
        metrics_function,
        alpha: float = 0.10,
        delta_min: float = 0.55,
        refine_frequency: int = 5,
    ) -> None:
        self._scoring_fn = scoring_function
        self._emitter = emitter
        self._metrics_fn = metrics_function
        self._alpha = alpha
        self._delta_min = delta_min
        self._refine_freq = refine_frequency

    def init(
        self,
        genotypes: Genotype,
        centroids: jnp.ndarray,
        key: RNGKey,
    ) -> Tuple[DistributionalRepertoire, DistributionalRepertoire, Optional[EmitterState], RNGKey]:
        key, score_key = jax.random.split(key)
        f, d, s = self._scoring_fn(genotypes, score_key)

        # A1: Scout (Naive)
        a1 = DistributionalRepertoire.init(
            genotypes=genotypes, fitnesses=f, descriptors=d,
            centroids=centroids, extra_scores=s, use_ht=False
        )
        # A2: Judge (HT-Gated)
        a2 = DistributionalRepertoire.init(
            genotypes=genotypes, fitnesses=f, descriptors=d,
            centroids=centroids, extra_scores=s, use_ht=True,
            alpha=self._alpha, delta_min=self._delta_min
        )

        key, emitter_key = jax.random.split(key)
        emitter_state, key = self._emitter.init(
            key=emitter_key, repertoire=a1, genotypes=genotypes, 
            fitnesses=f, descriptors=d, extra_scores=s
        )
        return a1, a2, emitter_state, key

    def update(
        self,
        a1: DistributionalRepertoire,
        a2: DistributionalRepertoire,
        emitter_state: Optional[EmitterState],
        iteration: int,
        key: RNGKey,
    ) -> Tuple[DistributionalRepertoire, DistributionalRepertoire, Optional[EmitterState], dict]:
        
        # 1. A1 Scout Update (Always exploring)
        offspring, _ = self._emitter.emit(a1, emitter_state, key)
        key, score_key = jax.random.split(key)
        fitnesses, descriptors, extra_scores = self._scoring_fn(offspring, score_key)
        a1 = a1.add(offspring, descriptors, fitnesses, extra_scores)
        
        # 2. A2 Judge Update (Intermittent Pull)
        def _do_refine(carry):
            _a2, _a1 = carry
            return _a2.add(_a1.genotypes, _a1.descriptors, _a1.fitnesses, {"scores": _a1.scores}), _a1

        a2, _ = jax.lax.cond(
            (iteration % self._refine_freq == 0) & (iteration > 0),
            _do_refine,
            lambda carry: carry,
            (a2, a1)
        )

        emitter_state = self._emitter.state_update(emitter_state, a1, offspring, fitnesses, descriptors, extra_scores)
        return a1, a2, emitter_state, {"a1": self._metrics_fn(a1), "a2": self._metrics_fn(a2)}