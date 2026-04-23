import os
import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats
import matplotlib.pyplot as plt
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids

# Import sub-modules
from evaluator import evaluate_via_pytorch, extract_features, BD_BRIGHTNESS_IDX, BD_ENTROPY_IDX
from fdme_emitter import FDMEEmitter
from fdme_map_elites import DROMEMAPElites
from repertoire import DistributionalRepertoire

# --- Config ---
N_SEEDS         = 15
TOTAL_ITERS     = 100
REFINE_FREQ     = 25 
GRID_SIZE       = 5
LATENT_DIM      = 8
BATCH_SIZE      = 4
INIT_BATCH_SIZE = 4
ALPHA           = 0.10
DELTA_MIN       = 0.55

def scoring_function(genotypes, key):
    B = genotypes.shape[0]
    base = jnp.clip(jnp.mean(genotypes, axis=1) * 0.5 + 0.5, 0.05, 0.95)
    images = jnp.broadcast_to(base[:, None, None, None], (B, 64, 64, 3))
    keys = jax.random.split(key, 5)
    for k in range(5):
        images = images + jax.random.normal(keys[k], images.shape) / (k + 2)
    images = jnp.clip(images, 0.0, 1.0)
    
    def _numpy_bd(x):
        return np.stack([extract_features(img) for img in x])[:, [BD_BRIGHTNESS_IDX, BD_ENTROPY_IDX]].astype(np.float32)
    
    descriptors = jax.pure_callback(_numpy_bd, jax.ShapeDtypeStruct((B, 2), jnp.float32), images)
    scores = evaluate_via_pytorch(images)
    fitnesses = -jnp.mean(scores, axis=1)
    return fitnesses, descriptors, {"scores": scores}

def run_one_seed(seed: int):
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    init_genotypes = jax.random.normal(init_key, (INIT_BATCH_SIZE, LATENT_DIM))
    centroids = compute_euclidean_centroids((GRID_SIZE, GRID_SIZE), 0, 1)
    
    emitter = FDMEEmitter(BATCH_SIZE, LATENT_DIM)
    map_elites = DROMEMAPElites(
        scoring_function=scoring_function,
        emitter=emitter,
        metrics_function=lambda r: {
            "qd": float(jnp.sum(r.fitnesses, where=r.fitnesses > -jnp.inf)),
            "cov": float(100 * jnp.mean(r.fitnesses > -jnp.inf)),
            "max_fit": float(jnp.max(r.fitnesses))
        },
        alpha=ALPHA,
        delta_min=DELTA_MIN,
        refine_frequency=REFINE_FREQ
    )
    
    a1, a2, emitter_state, key = map_elites.init(init_genotypes, centroids, key)
    
    for t in range(TOTAL_ITERS):
        key, subkey = jax.random.split(key)
        a1, a2, emitter_state, m = map_elites.update(a1, a2, emitter_state, t, subkey)
        
    return a1, a2, m

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print(f"Running {N_SEEDS} seeds (Total Iters: {TOTAL_ITERS}) ...")
    
    qds_a1, qds_a2 = [], []
    cov_a1, cov_a2 = [], []
    max_fits = []
    
    total_att, total_rej_p, total_rej_es = 0, 0, 0

    for s in range(N_SEEDS):
        a1, a2, m = run_one_seed(s)
        
        qds_a1.append(m['a1']['qd'])
        qds_a2.append(m['a2']['qd'])
        cov_a1.append(m['a1']['cov'])
        cov_a2.append(m['a2']['cov'])
        max_fits.append(m['a2']['max_fit'])
        
        # Aggregate A2 Rejections
        total_att += int(jnp.sum(a2.extra_scores["attempt_counts"]))
        total_rej_p += int(jnp.sum(a2.extra_scores["rejection_p_count"]))
        total_rej_es += int(jnp.sum(a2.extra_scores["rejection_es_count"]))
        
        print(f"  seed {s:2d} ({s+1:2d}/{N_SEEDS}) | A1 QD={m['a1']['qd']:.4f}  A2 QD={m['a2']['qd']:.4f}  A1 cov={m['a1']['cov']:.0f}%  A2 cov={m['a2']['cov']:.0f}%")

    # --- Statistical Summary ---
    qds_a1, qds_a2 = np.array(qds_a1), np.array(qds_a2)
    diffs = qds_a2 - qds_a1
    t_stat, p_val = stats.ttest_rel(qds_a2, qds_a1)
    
    # Simple Bootstrap for CI
    boot_means = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(1000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print("\n" + "="*65)
    print("  STATISTICAL COMPARISON  (N=20 paired seeds)")
    print(f"  A1: naive (Baseline) -> A2: DROME (HT-gated)")
    print("="*65)
    print(f"  A1 final QD:     {np.mean(qds_a1):+7.4f} ± {np.std(qds_a1):.4f}")
    print(f"  A2 final QD:     {np.mean(qds_a2):+7.4f} ± {np.std(qds_a2):.4f}")
    print(f"  Mean diff A2-A1: {np.mean(diffs):+7.4f} ± {np.std(diffs):.4f}")
    print(f"  t-statistic:     {t_stat:.4f}")
    print(f"  p-value:         {p_val:.4f} {'(n.s.)' if p_val > 0.05 else '*'}")
    print(f"  Bootstrap 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  CI excludes 0:   {'YES' if ci_low > 0 or ci_high < 0 else 'NO'}")
    print(f"\n  A1 Coverage: {np.mean(cov_a1):.1f}% ± {np.std(cov_a1):.1f}%")
    print(f"  A2 Coverage: {np.mean(cov_a2):.1f}% ± {np.std(cov_a2):.1f}%")
    print(f"  A2 Max Fitness: {np.mean(max_fits):.4f} ± {np.std(max_fits):.4f}")
    print("="*65)
    print(f"\n  A2 HT gate breakdown (all seeds combined):")
    print(f"    Attempts:           {total_att}")
    print(f"    Rejected (p fail):  {total_rej_p}")
    print(f"    Rejected (ES fail): {total_rej_es}")
    print(f"    Acceptance Rate:    {((total_att - total_rej_p - total_rej_es)/max(1, total_att))*100:.1f}%")
    print("="*65)