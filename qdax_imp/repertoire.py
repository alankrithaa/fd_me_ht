"""
repertoire.py
=============
DistributionalRepertoire: one class, two modes.

  A1 = DistributionalRepertoire(..., use_ht=False)  -- naive, runs all T iters
  A2 = DistributionalRepertoire(..., use_ht=True)   -- HT-gated, runs all T iters

Both archives receive the same scored candidates every iteration.
Empty cells are always filled unconditionally in both archives.
There is no exploration-window phase-switch: the two-archive design makes it
unnecessary because A1 naturally fulfils the exploratory role at all times.
"""

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
    get_cells_indices,
)
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype

from ht_logic import calculate_ht_replacement


class DistributionalRepertoire(MapElitesRepertoire):
    """
    MAP-Elites archive that stores the full score vector per cell.

    Extra state (all in extra_scores dict):
        scores            (C, M) float32  -- current elite's score vector
        acceptance_counts (C,)   int32    -- successful replacements per cell
        attempt_counts    (C,)   int32    -- total attempts per cell
        sum_effect_sizes  (C,)   float32  -- cumulative CLES on acceptance
        rejection_p_count (C,)   int32    -- rejections from p >= alpha
        rejection_es_count(C,)   int32    -- rejections from cles <= delta_min
        use_ht_flag       scalar bool
        alpha_value       scalar float
        delta_min_value   scalar float
    """

    SCORES_KEY  = "scores"
    ACC_KEY     = "acceptance_counts"
    ATT_KEY     = "attempt_counts"
    ES_KEY      = "sum_effect_sizes"
    REJ_P_KEY   = "rejection_p_count"
    REJ_ES_KEY  = "rejection_es_count"
    USE_HT_KEY  = "use_ht_flag"
    ALPHA_KEY   = "alpha_value"
    DELTA_KEY   = "delta_min_value"

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def scores(self)              -> jnp.ndarray: return self.extra_scores[self.SCORES_KEY]
    @property
    def acceptance_counts(self)   -> jnp.ndarray: return self.extra_scores[self.ACC_KEY]
    @property
    def attempt_counts(self)      -> jnp.ndarray: return self.extra_scores[self.ATT_KEY]
    @property
    def sum_effect_sizes(self)    -> jnp.ndarray: return self.extra_scores[self.ES_KEY]
    @property
    def rejection_p_count(self)   -> jnp.ndarray: return self.extra_scores[self.REJ_P_KEY]
    @property
    def rejection_es_count(self)  -> jnp.ndarray: return self.extra_scores[self.REJ_ES_KEY]
    @property
    def use_ht_flag(self)         -> jnp.ndarray: return self.extra_scores[self.USE_HT_KEY]
    @property
    def alpha_value(self)         -> jnp.ndarray: return self.extra_scores[self.ALPHA_KEY]
    @property
    def delta_min_value(self)     -> jnp.ndarray: return self.extra_scores[self.DELTA_KEY]

    # ── init() ────────────────────────────────────────────────────────────────
    @classmethod
    def init(
        cls,
        genotypes:    Genotype,
        fitnesses:    Fitness,
        descriptors:  Descriptor,
        centroids:    jnp.ndarray,
        extra_scores: ExtraScores,
        use_ht:       bool  = True,
        alpha:        float = 0.10,
        delta_min:    float = 0.55,
    ) -> "DistributionalRepertoire":
        """Create an empty repertoire then populate with the initial population."""
        C   = centroids.shape[0]
        D   = genotypes.shape[1]
        BD  = descriptors.shape[1]
        M   = extra_scores["scores"].shape[1]

        empty = cls(
            genotypes   = jnp.zeros((C, D),  dtype=genotypes.dtype),
            fitnesses   = jnp.full ((C,), -jnp.inf, dtype=fitnesses.dtype),
            descriptors = jnp.zeros((C, BD), dtype=descriptors.dtype),
            centroids   = centroids,
            extra_scores= {
                cls.SCORES_KEY:  jnp.zeros((C, M), dtype=jnp.float32),
                cls.ACC_KEY:     jnp.zeros((C,),   dtype=jnp.int32),
                cls.ATT_KEY:     jnp.zeros((C,),   dtype=jnp.int32),
                cls.ES_KEY:      jnp.zeros((C,),   dtype=jnp.float32),
                cls.REJ_P_KEY:   jnp.zeros((C,),   dtype=jnp.int32),
                cls.REJ_ES_KEY:  jnp.zeros((C,),   dtype=jnp.int32),
                cls.USE_HT_KEY:  jnp.array(use_ht),
                cls.ALPHA_KEY:   jnp.array(alpha,     dtype=jnp.float32),
                cls.DELTA_KEY:   jnp.array(delta_min, dtype=jnp.float32),
            },
            keys_extra_scores=(),
        )
        return empty.add(
            batch_of_genotypes   = genotypes,
            batch_of_descriptors = descriptors,
            batch_of_fitnesses   = fitnesses,
            batch_of_extra_scores= extra_scores,
        )

    # ── add() ─────────────────────────────────────────────────────────────────
    def add(
        self,
        batch_of_genotypes:    Genotype,
        batch_of_descriptors:  Descriptor,
        batch_of_fitnesses:    Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> "DistributionalRepertoire":
        """
        Insert candidates into the archive.

        - Empty cell   -> unconditional insert (both A1 and A2).
        - A1 (use_ht=False) -> replace if mean(S_new) < mean(S_old).
        - A2 (use_ht=True)  -> replace only if HT gate passes (3 conditions).
        """
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        batch_of_scores = batch_of_extra_scores.get(
            self.SCORES_KEY,
            jnp.zeros(
                (batch_of_genotypes.shape[0], self.scores.shape[-1]),
                dtype=jnp.float32,
            ),
        )

        use_ht    = self.use_ht_flag
        alpha     = self.alpha_value
        delta_min = self.delta_min_value

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        C                = self.centroids.shape[0]

        # Per-cell winner: highest fitness in the batch for each cell
        best_f_per_cell     = jax.ops.segment_max(batch_of_fitnesses, batch_of_indices, C)
        candidate_is_winner = (batch_of_fitnesses == best_f_per_cell[batch_of_indices])

        def _process_one(rep, i_inputs):
            is_winner, c_idx, u_new, bd_new, f_new, S_new = i_inputs

            if not bool(is_winner):
                return rep

            c = jnp.asarray(c_idx, dtype=jnp.int32)

            # Count attempt
            rep = rep.replace(extra_scores={
                **rep.extra_scores,
                self.ATT_KEY: rep.extra_scores[self.ATT_KEY].at[c].add(1),
            })

            # Empty cell: always insert
            if bool(rep.fitnesses[c] == -jnp.inf):
                return rep._insert(c, u_new, bd_new, f_new, S_new)

            S_old = rep.scores[c]

            # A1 — naive mean comparison
            if not bool(use_ht):
                if bool(jnp.mean(S_new) < jnp.mean(S_old)):
                    return rep._insert(c, u_new, bd_new, f_new, S_new,
                                       cles=jnp.asarray(1.0, dtype=jnp.float32))
                return rep

            # A2 — hypothesis-testing gate
            should_replace, p_val, cles = calculate_ht_replacement(
                S_new, S_old, alpha=alpha, delta_min=delta_min
            )
            accepted = bool(should_replace)

            # Track rejection reason
            p_rej  = int((not accepted) and bool(p_val >= alpha))
            es_rej = int((not accepted) and bool(p_val < alpha) and bool(cles <= delta_min))
            rep = rep.replace(extra_scores={
                **rep.extra_scores,
                self.REJ_P_KEY:  rep.extra_scores[self.REJ_P_KEY].at[c].add(p_rej),
                self.REJ_ES_KEY: rep.extra_scores[self.REJ_ES_KEY].at[c].add(es_rej),
            })

            if accepted:
                return rep._insert(c, u_new, bd_new, f_new, S_new, cles=cles)
            return rep

        rep        = self
        batch_size = int(batch_of_genotypes.shape[0])
        inputs     = (candidate_is_winner, batch_of_indices,
                      batch_of_genotypes, batch_of_descriptors,
                      batch_of_fitnesses, batch_of_scores)
        for i in range(batch_size):
            rep = _process_one(rep, tuple(x[i] for x in inputs))

        return rep

    # ── _insert() ─────────────────────────────────────────────────────────────
    def _insert(
        self,
        c:       jnp.ndarray,
        u:       jnp.ndarray,
        bd:      jnp.ndarray,
        fitness: jnp.ndarray,
        S:       jnp.ndarray,
        cles:    float = 1.0,
    ) -> "DistributionalRepertoire":
        new_extra = {
            **self.extra_scores,
            self.SCORES_KEY: self.extra_scores[self.SCORES_KEY].at[c].set(S),
            self.ACC_KEY:    self.extra_scores[self.ACC_KEY].at[c].add(1),
            self.ES_KEY:     self.extra_scores[self.ES_KEY].at[c].add(cles),
        }
        return self.replace(
            genotypes   = jax.tree.map(lambda g: g.at[c].set(u), self.genotypes),
            fitnesses   = self.fitnesses.at[c].set(fitness),
            descriptors = self.descriptors.at[c].set(bd),
            extra_scores= new_extra,
        )