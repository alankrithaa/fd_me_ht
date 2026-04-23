from __future__ import annotations
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire, get_cells_indices
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

    @property
    def scores(self) -> jnp.ndarray: 
        return self.extra_scores[self.SCORES_KEY]

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: jnp.ndarray,
        extra_scores: ExtraScores,
        use_ht: bool = True,
        alpha: float = 0.10,
        delta_min: float = 0.55,
    ) -> "DistributionalRepertoire":
        C = centroids.shape[0]
        D = genotypes.shape[1]
        BD = descriptors.shape[1]
        M = extra_scores["scores"].shape[1]

        empty = cls(
            genotypes=jnp.zeros((C, D), dtype=genotypes.dtype),
            fitnesses=jnp.full((C,), -jnp.inf, dtype=fitnesses.dtype),
            descriptors=jnp.zeros((C, BD), dtype=descriptors.dtype),
            centroids=centroids,
            extra_scores={
                cls.SCORES_KEY: jnp.zeros((C, M), dtype=jnp.float32),
                cls.ACC_KEY: jnp.zeros((C,), dtype=jnp.int32),
                cls.ATT_KEY: jnp.zeros((C,), dtype=jnp.int32),
                cls.ES_KEY: jnp.zeros((C,), dtype=jnp.float32),
                cls.REJ_P_KEY: jnp.zeros((C,), dtype=jnp.int32),
                cls.REJ_ES_KEY: jnp.zeros((C,), dtype=jnp.int32),
                cls.USE_HT_KEY: jnp.array(use_ht),
                cls.ALPHA_KEY: jnp.array(alpha, dtype=jnp.float32),
                cls.DELTA_KEY: jnp.array(delta_min, dtype=jnp.float32),
            },
            keys_extra_scores=(),
        )
        return empty.add(genotypes, descriptors, fitnesses, extra_scores)

    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> "DistributionalRepertoire":
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        batch_of_scores = batch_of_extra_scores.get(
            self.SCORES_KEY,
            jnp.zeros((batch_of_genotypes.shape[0], self.scores.shape[-1]), dtype=jnp.float32)
        )

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        def scan_body(rep, inputs):
            u_new, bd_new, f_new, S_new, c_idx = inputs
            c = jnp.asarray(c_idx, dtype=jnp.int32)
            
            # Record Attempt
            rep = rep.replace(extra_scores={
                **rep.extra_scores,
                self.ATT_KEY: rep.extra_scores[self.ATT_KEY].at[c].add(1)
            })

            # Gate Logic
            should_replace, p_val, cles = calculate_ht_replacement(
                S_new, rep.scores[c], 
                rep.extra_scores[self.ALPHA_KEY], 
                rep.extra_scores[self.DELTA_KEY]
            )
            
            # Standard MAP-Elites mean comparison
            naive_replace = jnp.mean(S_new) < jnp.mean(rep.scores[c])
            
            # Use HT if flag is set, otherwise use naive mean
            cond = jnp.where(rep.extra_scores[self.USE_HT_KEY], should_replace, naive_replace)
            
            # Always allow if cell was empty
            cond = jnp.where(rep.fitnesses[c] == -jnp.inf, True, cond)

            def _perform_insert(_rep):
                return _rep._insert(c, u_new, bd_new, f_new, S_new, jnp.where(cond, cles, 1.0))

            return jax.lax.cond(cond, _perform_insert, lambda x: x, rep), None

        final_rep, _ = jax.lax.scan(
            scan_body, 
            self, 
            (batch_of_genotypes, batch_of_descriptors, batch_of_fitnesses, batch_of_scores, batch_of_indices)
        )
        return final_rep

    def _insert(self, c, u, bd, fitness, S, cles=1.0):
        new_extra = {
            **self.extra_scores,
            self.SCORES_KEY: self.extra_scores[self.SCORES_KEY].at[c].set(S),
            self.ACC_KEY: self.extra_scores[self.ACC_KEY].at[c].add(1),
            self.ES_KEY: self.extra_scores[self.ES_KEY].at[c].add(cles),
        }
        return self.replace(
            genotypes=jax.tree.map(lambda g: g.at[c].set(u), self.genotypes),
            fitnesses=self.fitnesses.at[c].set(fitness),
            descriptors=self.descriptors.at[c].set(bd),
            extra_scores=new_extra,
        )