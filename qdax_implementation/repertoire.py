"""
repertoire.py
=============
Distributional repertoire built on top of QDAX MapElitesRepertoire.

Main idea:
- Keep QDAX storage layout and API
- Replace scalar archive update with hypothesis-testing gate
- Store score distributions and tracking counters in extra_scores
"""

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
#directly import MapElitesRepertoire and related functions from QDAX, since we subclass it
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
    get_cells_indices,
)
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype

from ht_logic import calculate_ht_replacement


class DistributionalRepertoire(MapElitesRepertoire):
    SCORES_KEY = "scores"
    ACC_KEY = "acceptance_counts"
    ATT_KEY = "attempt_counts"
    ES_KEY = "sum_effect_sizes"
    REJ_P_KEY = "rejection_p_count"
    REJ_ES_KEY = "rejection_es_count"

    USE_HT_KEY = "use_ht_flag"
    ALPHA_KEY = "alpha_value"
    DELTA_KEY = "delta_min_value"

    ALL_EXTRA_KEYS: Tuple[str, ...] = (
        SCORES_KEY,
        ACC_KEY,
        ATT_KEY,
        ES_KEY,
        REJ_P_KEY,
        REJ_ES_KEY,
        USE_HT_KEY,
        ALPHA_KEY,
        DELTA_KEY,
    )

    @property
    def scores(self) -> jnp.ndarray:
        return self.extra_scores[self.SCORES_KEY]

    @property
    def acceptance_counts(self) -> jnp.ndarray:
        return self.extra_scores[self.ACC_KEY]

    @property
    def attempt_counts(self) -> jnp.ndarray:
        return self.extra_scores[self.ATT_KEY]

    @property
    def sum_effect_sizes(self) -> jnp.ndarray:
        return self.extra_scores[self.ES_KEY]

    @property
    def rejection_p_count(self) -> jnp.ndarray:
        return self.extra_scores[self.REJ_P_KEY]

    @property
    def rejection_es_count(self) -> jnp.ndarray:
        return self.extra_scores[self.REJ_ES_KEY]

    @property
    def use_ht_flag(self) -> bool:
        return bool(self.extra_scores[self.USE_HT_KEY])

    @property
    def alpha_value(self) -> float:
        return float(self.extra_scores[self.ALPHA_KEY])

    @property
    def delta_min_value(self) -> float:
        return float(self.extra_scores[self.DELTA_KEY])

    @classmethod #all from QDAX
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: jnp.ndarray,
        extra_scores: ExtraScores,
        use_ht: bool = True,
        alpha: float = 0.05,
        delta_min: float = 0.6,
    ) -> "DistributionalRepertoire":
        """
        Build a distributional repertoire from an initial scored population.
        Mirrors QDAX MapElitesRepertoire.init(...), but adds tracking arrays.
        """
        num_centroids = centroids.shape[0]
        latent_dim = genotypes.shape[1]
        descriptor_dim = descriptors.shape[1]
        num_raters = extra_scores["scores"].shape[1]

        default_genotypes = jnp.zeros((num_centroids, latent_dim), dtype=genotypes.dtype)
        default_fitnesses = jnp.full((num_centroids,), -jnp.inf, dtype=fitnesses.dtype)
        default_descriptors = jnp.zeros((num_centroids, descriptor_dim), dtype=descriptors.dtype)

        default_extra_scores = {
            cls.SCORES_KEY: jnp.zeros((num_centroids, num_raters), dtype=extra_scores["scores"].dtype),
            cls.ACC_KEY: jnp.zeros((num_centroids,), dtype=jnp.int32),
            cls.ATT_KEY: jnp.zeros((num_centroids,), dtype=jnp.int32),
            cls.ES_KEY: jnp.zeros((num_centroids,), dtype=jnp.float32),
            cls.REJ_P_KEY: jnp.zeros((num_centroids,), dtype=jnp.int32),
            cls.REJ_ES_KEY: jnp.zeros((num_centroids,), dtype=jnp.int32),
            cls.USE_HT_KEY: jnp.array(use_ht),
            cls.ALPHA_KEY: jnp.array(alpha, dtype=jnp.float32),
            cls.DELTA_KEY: jnp.array(delta_min, dtype=jnp.float32),
        }

        repertoire = cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            extra_scores=default_extra_scores,
            keys_extra_scores=cls.ALL_EXTRA_KEYS,
        )

        return repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_descriptors=descriptors,
            batch_of_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
        )

    @classmethod
    def init_empty(
        cls,
        grid_shape: Tuple[int, int],
        latent_dim: int,
        num_raters: int,
        min_bd: float = 0.0,
        max_bd: float = 1.0,
        use_ht: bool = True,
        alpha: float = 0.05,
        delta_min: float = 0.6,
    ) -> "DistributionalRepertoire":
        centroids = compute_euclidean_centroids(
            grid_shape=grid_shape,
            minval=min_bd,
            maxval=max_bd,
        )
        num_centroids = centroids.shape[0]

        return cls(
            genotypes=jnp.zeros((num_centroids, latent_dim), dtype=jnp.float32),
            fitnesses=jnp.full((num_centroids,), -jnp.inf, dtype=jnp.float32),
            descriptors=jnp.zeros((num_centroids, 2), dtype=jnp.float32),
            centroids=centroids,
            extra_scores={
                cls.SCORES_KEY: jnp.zeros((num_centroids, num_raters), dtype=jnp.float32),
                cls.ACC_KEY: jnp.zeros((num_centroids,), dtype=jnp.int32),
                cls.ATT_KEY: jnp.zeros((num_centroids,), dtype=jnp.int32),
                cls.ES_KEY: jnp.zeros((num_centroids,), dtype=jnp.float32),
                cls.REJ_P_KEY: jnp.zeros((num_centroids,), dtype=jnp.int32),
                cls.REJ_ES_KEY: jnp.zeros((num_centroids,), dtype=jnp.int32),
                cls.USE_HT_KEY: jnp.array(use_ht),
                cls.ALPHA_KEY: jnp.array(alpha, dtype=jnp.float32),
                cls.DELTA_KEY: jnp.array(delta_min, dtype=jnp.float32),
            },
            keys_extra_scores=cls.ALL_EXTRA_KEYS,
        )

    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
        use_ht: Optional[bool] = None,
        alpha: Optional[float] = None,
        delta_min: Optional[float] = None,
    ) -> "DistributionalRepertoire":
        """
        Same API shape as QDAX add(), but if MAPElites.update() calls it without
        HT args, we fall back to config stored in the repertoire itself.
        """
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        batch_of_scores = batch_of_extra_scores.get(
            self.SCORES_KEY,
            jnp.zeros((batch_of_genotypes.shape[0], self.scores.shape[-1]), dtype=jnp.float32),
        )

        use_ht = self.use_ht_flag if use_ht is None else use_ht
        alpha = self.alpha_value if alpha is None else alpha
        delta_min = self.delta_min_value if delta_min is None else delta_min

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        num_centroids = self.centroids.shape[0]
        batch_size = batch_of_genotypes.shape[0]

        best_fitness_in_batch = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices,
            num_segments=num_centroids,
        )

        candidate_is_batch_best = (
            batch_of_fitnesses == best_fitness_in_batch[batch_of_indices]
        )

        repertoire = self

        for b in range(batch_size):
            if not bool(candidate_is_batch_best[b]):
                continue

            c = int(batch_of_indices[b])
            u_new = batch_of_genotypes[b]
            bd_new = batch_of_descriptors[b]
            f_new = batch_of_fitnesses[b]
            S_new = batch_of_scores[b]

            new_att = repertoire.extra_scores[self.ATT_KEY].at[c].add(1)
            repertoire = repertoire.replace(
                extra_scores={**repertoire.extra_scores, self.ATT_KEY: new_att}
            )

            is_empty = bool(repertoire.fitnesses[c] == -jnp.inf)

            if is_empty:
                repertoire = repertoire._insert(c, u_new, bd_new, f_new, S_new)
            else:
                S_old = repertoire.scores[c]

                if use_ht:
                    should_replace, p_val, cles = calculate_ht_replacement(
                        S_new, S_old, alpha=alpha, delta_min=delta_min
                    )
                    p_val_f = float(p_val)
                    cles_f = float(cles)

                    # Rejection tracking: mutually exclusive failure modes
                    # p_reject: statistical significance failed (p >= alpha)
                    # es_reject: significance passed (p < alpha) but effect size failed (cles <= delta_min)
                    is_p_reject = (not bool(should_replace)) and (p_val_f >= alpha)
                    is_es_reject = (not bool(should_replace)) and (p_val_f < alpha) and (cles_f <= delta_min)

                    new_rp = repertoire.extra_scores[self.REJ_P_KEY].at[c].add(int(is_p_reject))
                    new_res = repertoire.extra_scores[self.REJ_ES_KEY].at[c].add(int(is_es_reject))
                    repertoire = repertoire.replace(
                        extra_scores={
                            **repertoire.extra_scores,
                            self.REJ_P_KEY: new_rp,
                            self.REJ_ES_KEY: new_res,
                        }
                    )
                else:
                    should_replace = bool(jnp.mean(S_new) < jnp.mean(S_old))
                    cles_f = 1.0

                if bool(should_replace):
                    repertoire = repertoire._insert(
                        c, u_new, bd_new, f_new, S_new, cles=cles_f
                    )

        return repertoire

    def _insert(
        self,
        c: int,
        u: jnp.ndarray,
        bd: jnp.ndarray,
        fitness: jnp.ndarray,
        S: jnp.ndarray,
        cles: float = 1.0,
    ) -> "DistributionalRepertoire":
        new_genotypes = jax.tree.map(lambda g: g.at[c].set(u), self.genotypes)
        new_fitnesses = self.fitnesses.at[c].set(fitness)
        new_descriptors = self.descriptors.at[c].set(bd)

        new_extra = {
            **self.extra_scores,
            self.SCORES_KEY: self.extra_scores[self.SCORES_KEY].at[c].set(S),
            self.ACC_KEY: self.extra_scores[self.ACC_KEY].at[c].add(1),
            self.ES_KEY: self.extra_scores[self.ES_KEY].at[c].add(cles),
        }

        return self.replace(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            extra_scores=new_extra,
        )