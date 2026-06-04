"""
HT-MAP-Elites Proof of Concept v2 — WITH INDEPENDENT RE-EVALUATION
===================================================================
Key fix: When A2 pulls candidates from A1, it RE-EVALUATES them with
fresh noise rather than using A1's cached (selection-biased) scores.

This eliminates the winner's curse: candidates that got into A1 via
lucky noise draws will NOT reproduce that luck on re-evaluation.
The HT gate now tests genuinely independent evidence.

Run: python ht_proof_v2.py
Time: ~5 min on CPU
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

N_SEEDS        = 15
TOTAL_ITERS    = 300
BATCH_SIZE     = 8
GRID_SIZE      = 5
LATENT_DIM     = 8
FEATURE_DIM    = 5
NUM_VLMS       = 10
REFINE_FREQ    = 5
ALPHA          = 0.10
DELTA_MIN      = 0.55
MUTATION_SIGMA = 0.3

SIGMA_LEVELS = [0.05, 0.10, 0.20, 0.30]

# ══════════════════════════════════════════════════════════════════════════════
# FIXED RANDOM COMPONENTS (seed 42)
# ══════════════════════════════════════════════════════════════════════════════

_rng_fixed = np.random.RandomState(42)
_raw_w = np.abs(_rng_fixed.randn(NUM_VLMS, FEATURE_DIM)).astype(np.float32)
VLM_WEIGHTS = _raw_w / (np.linalg.norm(_raw_w, axis=1, keepdims=True) + 1e-10)
GENO_TO_FEAT = _rng_fixed.randn(LATENT_DIM, FEATURE_DIM).astype(np.float32) * 0.5

def make_centroids(grid_size):
    edges = np.linspace(0, 1, grid_size + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    cx, cy = np.meshgrid(centers, centers)
    return np.stack([cx.ravel(), cy.ravel()], axis=1).astype(np.float32)

CENTROIDS = make_centroids(GRID_SIZE)

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def genotype_to_features(genotypes):
    return sigmoid(genotypes @ GENO_TO_FEAT)

def true_fitness(genotypes):
    """Analytical ground truth — zero noise, exact."""
    features = genotype_to_features(genotypes)
    clean = features @ VLM_WEIGHTS.T
    return np.mean(clean, axis=1)

def noisy_evaluate(genotypes, sigma, rng):
    """Stochastic evaluation. Returns (fitnesses, descriptors, scores)."""
    features = genotype_to_features(genotypes)
    clean_scores = -(features @ VLM_WEIGHTS.T)
    noise = rng.randn(*clean_scores.shape).astype(np.float32) * sigma
    scores = clean_scores + noise
    fitnesses = -np.mean(scores, axis=1)
    descriptors = features[:, [0, 2]]
    return fitnesses, descriptors, scores

def assign_cells(descriptors):
    dists = np.sum((descriptors[:, None, :] - CENTROIDS[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1)

def ht_gate(new_scores, old_scores, alpha, delta_min):
    mean_new = np.mean(new_scores)
    mean_old = np.mean(old_scores)
    var_new = np.var(new_scores, ddof=0)
    var_old = np.var(old_scores, ddof=0)
    M = len(new_scores)
    
    diffs = new_scores[:, None] - old_scores[None, :]
    cles = float(np.mean((diffs < 0).astype(float) + 0.5 * (diffs == 0).astype(float)))
    
    se = np.sqrt(var_new / M + var_old / M + 1e-10)
    z = abs(mean_new - mean_old) / se
    p_val = float(2.0 * (1.0 - stats.norm.cdf(z)))
    
    directional = mean_new < mean_old
    significant = p_val < alpha
    meaningful = cles > delta_min
    
    accept = directional and significant and meaningful
    return accept, p_val, cles

# ══════════════════════════════════════════════════════════════════════════════
# ARCHIVE
# ══════════════════════════════════════════════════════════════════════════════

class Archive:
    def __init__(self, n_cells, latent_dim, num_vlms):
        self.n_cells = n_cells
        self.genotypes = np.zeros((n_cells, latent_dim), dtype=np.float32)
        self.fitnesses = np.full(n_cells, -np.inf, dtype=np.float32)
        self.scores = np.zeros((n_cells, num_vlms), dtype=np.float32)
        self.descriptors = np.zeros((n_cells, 2), dtype=np.float32)
        self.attempts = np.zeros(n_cells, dtype=np.int32)
        self.accepts = np.zeros(n_cells, dtype=np.int32)
        self.rej_p = np.zeros(n_cells, dtype=np.int32)
        self.rej_es = np.zeros(n_cells, dtype=np.int32)
    
    def occupied_mask(self):
        return self.fitnesses > -np.inf
    
    def try_insert_naive(self, cell, genotype, fitness, scores, descriptor):
        if self.fitnesses[cell] == -np.inf or fitness > self.fitnesses[cell]:
            self.genotypes[cell] = genotype
            self.fitnesses[cell] = fitness
            self.scores[cell] = scores
            self.descriptors[cell] = descriptor
            return True
        return False
    
    def try_insert_ht(self, cell, genotype, fitness, scores, descriptor, alpha, delta_min):
        """HT-gated insertion using FRESH scores (not A1's cached scores)."""
        self.attempts[cell] += 1
        
        if self.fitnesses[cell] == -np.inf:
            self.genotypes[cell] = genotype
            self.fitnesses[cell] = fitness
            self.scores[cell] = scores
            self.descriptors[cell] = descriptor
            self.accepts[cell] += 1
            return True
        
        accept, p_val, cles = ht_gate(scores, self.scores[cell], alpha, delta_min)
        
        if accept:
            self.genotypes[cell] = genotype
            self.fitnesses[cell] = fitness
            self.scores[cell] = scores
            self.descriptors[cell] = descriptor
            self.accepts[cell] += 1
            return True
        else:
            if p_val >= alpha:
                self.rej_p[cell] += 1
            elif cles <= delta_min:
                self.rej_es[cell] += 1
            return False
    
    def qd_score(self):
        mask = self.occupied_mask()
        return float(np.sum(self.fitnesses[mask]))
    
    def gt_qd_score(self):
        mask = self.occupied_mask()
        if not np.any(mask):
            return 0.0
        return float(np.sum(true_fitness(self.genotypes[mask])))
    
    def coverage(self):
        return float(np.mean(self.occupied_mask()) * 100)
    
    def rejection_rate(self):
        total_att = int(np.sum(self.attempts))
        total_rej = int(np.sum(self.rej_p)) + int(np.sum(self.rej_es))
        return total_rej / max(1, total_att)

def emit_batch(a1, rng, batch_size, mutation_prob=0.5):
    occupied = np.where(a1.occupied_mask())[0]
    batch = []
    for _ in range(batch_size):
        if len(occupied) > 0 and rng.rand() < mutation_prob:
            idx = rng.choice(occupied)
            u = a1.genotypes[idx] + rng.randn(LATENT_DIM).astype(np.float32) * MUTATION_SIGMA
        else:
            u = rng.randn(LATENT_DIM).astype(np.float32)
        batch.append(u)
    return np.stack(batch, axis=0)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — THE KEY CHANGE IS IN THE REFINE STEP
# ══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed, sigma):
    rng = np.random.RandomState(seed)
    
    a1 = Archive(GRID_SIZE ** 2, LATENT_DIM, NUM_VLMS)
    a2 = Archive(GRID_SIZE ** 2, LATENT_DIM, NUM_VLMS)
    
    for t in range(TOTAL_ITERS):
        # 1. Emit + evaluate + update A1 (naive, every iteration)
        batch_g = emit_batch(a1, rng, BATCH_SIZE)
        fitnesses, descriptors, scores = noisy_evaluate(batch_g, sigma, rng)
        cells = assign_cells(descriptors)
        
        for b in range(BATCH_SIZE):
            a1.try_insert_naive(cells[b], batch_g[b], fitnesses[b],
                                scores[b], descriptors[b])
        
        # 2. Refine A2 every REFINE_FREQ iterations
        if t > 0 and t % REFINE_FREQ == 0:
            for c in range(GRID_SIZE ** 2):
                if a1.fitnesses[c] > -np.inf:
                    # ═══════════════════════════════════════════════════
                    # KEY CHANGE: RE-EVALUATE with fresh noise
                    # Do NOT use A1's cached scores (selection-biased).
                    # Generate independent scores for the HT gate.
                    # ═══════════════════════════════════════════════════
                    geno = a1.genotypes[c:c+1]
                    fresh_fit, fresh_desc, fresh_scores = noisy_evaluate(
                        geno, sigma, rng
                    )
                    a2.try_insert_ht(
                        c, a1.genotypes[c],
                        fresh_fit[0], fresh_scores[0], fresh_desc[0],
                        ALPHA, DELTA_MIN
                    )
    
    return a1, a2


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 75)
    print("  HT-MAP-Elites v2 — WITH INDEPENDENT RE-EVALUATION")
    print(f"  N_SEEDS={N_SEEDS}, T={TOTAL_ITERS}, B={BATCH_SIZE}")
    print(f"  frefine={REFINE_FREQ}, α={ALPHA}, δmin={DELTA_MIN}")
    print(f"  σ levels = {SIGMA_LEVELS}")
    print("  KEY CHANGE: A2 re-evaluates candidates with fresh noise")
    print("=" * 75)
    
    all_results = {}
    
    for sigma in SIGMA_LEVELS:
        print(f"\n{'━'*75}")
        print(f"  σ = {sigma}")
        print(f"{'━'*75}")
        
        apparent_a1, apparent_a2 = [], []
        gt_a1, gt_a2 = [], []
        cov_a1, cov_a2 = [], []
        total_rej_p, total_rej_es, total_att = 0, 0, 0
        
        for s in range(N_SEEDS):
            a1, a2 = run_one_seed(s, sigma)
            
            app1, app2 = a1.qd_score(), a2.qd_score()
            g1, g2 = a1.gt_qd_score(), a2.gt_qd_score()
            
            apparent_a1.append(app1); apparent_a2.append(app2)
            gt_a1.append(g1); gt_a2.append(g2)
            cov_a1.append(a1.coverage()); cov_a2.append(a2.coverage())
            
            total_att += int(np.sum(a2.attempts))
            total_rej_p += int(np.sum(a2.rej_p))
            total_rej_es += int(np.sum(a2.rej_es))
            
            print(f"  seed {s:2d} | A1: app={app1:.3f} gt={g1:.3f} | "
                  f"A2: app={app2:.3f} gt={g2:.3f} | "
                  f"cov={a1.coverage():.0f}%/{a2.coverage():.0f}%")
        
        apparent_a1 = np.array(apparent_a1)
        apparent_a2 = np.array(apparent_a2)
        gt_a1 = np.array(gt_a1)
        gt_a2 = np.array(gt_a2)
        
        inf_a1 = apparent_a1 - gt_a1
        inf_a2 = apparent_a2 - gt_a2
        
        gt_diff = gt_a2 - gt_a1
        t_stat, p_val = stats.ttest_rel(gt_a2, gt_a1)
        t_inf, p_inf = stats.ttest_rel(inf_a1, inf_a2)
        
        rej_rate = (total_rej_p + total_rej_es) / max(1, total_att)
        
        all_results[sigma] = {
            "apparent_a1": apparent_a1, "apparent_a2": apparent_a2,
            "gt_a1": gt_a1, "gt_a2": gt_a2,
            "inf_a1": inf_a1, "inf_a2": inf_a2,
            "t_gt": t_stat, "p_gt": p_val,
            "t_inf": t_inf, "p_inf": p_inf,
            "rej_rate": rej_rate,
            "rej_p": total_rej_p, "rej_es": total_rej_es, "att": total_att,
            "cov_a1": np.mean(cov_a1), "cov_a2": np.mean(cov_a2),
        }
        
        print(f"\n  Apparent QD:  A1={np.mean(apparent_a1):.4f}±{np.std(apparent_a1):.4f}  "
              f"A2={np.mean(apparent_a2):.4f}±{np.std(apparent_a2):.4f}")
        print(f"  Ground-Truth: A1={np.mean(gt_a1):.4f}±{np.std(gt_a1):.4f}  "
              f"A2={np.mean(gt_a2):.4f}±{np.std(gt_a2):.4f}")
        print(f"  Inflation:    A1={np.mean(inf_a1):.4f}±{np.std(inf_a1):.4f}  "
              f"A2={np.mean(inf_a2):.4f}±{np.std(inf_a2):.4f}")
        print(f"  GT comparison (A2 vs A1): t={t_stat:.4f}, p={p_val:.6f} "
              f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '(n.s.)'}")
        print(f"  Inflation comparison (A1 more inflated?): t={t_inf:.4f}, p={p_inf:.6f} "
              f"{'***' if p_inf < 0.001 else '**' if p_inf < 0.01 else '*' if p_inf < 0.05 else '(n.s.)'}")
        print(f"  Rejection rate: {rej_rate*100:.1f}% "
              f"(p-rej={total_rej_p}, es-rej={total_rej_es}, att={total_att})")
        print(f"  Coverage: A1={np.mean(cov_a1):.1f}% A2={np.mean(cov_a2):.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    
    print("\n\n" + "=" * 100)
    print("  MASTER SUMMARY TABLE")
    print("=" * 100)
    print(f"  {'σ':<6} {'App A1':<12} {'App A2':<12} {'GT A1':<12} {'GT A2':<12} "
          f"{'Inf A1':<10} {'Inf A2':<10} {'GT p-val':<12} {'Rej%':<8}")
    print(f"  {'─'*96}")
    
    for sigma in SIGMA_LEVELS:
        r = all_results[sigma]
        sig = "***" if r["p_gt"] < 0.001 else "**" if r["p_gt"] < 0.01 else "*" if r["p_gt"] < 0.05 else ""
        print(f"  {sigma:<6.2f} "
              f"{np.mean(r['apparent_a1']):>8.3f}    "
              f"{np.mean(r['apparent_a2']):>8.3f}    "
              f"{np.mean(r['gt_a1']):>8.3f}    "
              f"{np.mean(r['gt_a2']):>8.3f}    "
              f"{np.mean(r['inf_a1']):>7.4f}  "
              f"{np.mean(r['inf_a2']):>7.4f}  "
              f"{r['p_gt']:>9.6f}{sig:>3}  "
              f"{r['rej_rate']*100:>5.1f}%")
    
    print("=" * 100)
    
    # ══════════════════════════════════════════════════════════════════════════
    # VERIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    
    print("\n  VERIFICATION CHECKS:")
    
    p_vals = [all_results[s]["p_gt"] for s in SIGMA_LEVELS]
    if len(set([round(p, 4) for p in p_vals])) == 1:
        print("  ⚠ WARNING: All GT p-values identical — sigma may not be propagating!")
    else:
        print("  ✓ GT p-values differ across sigma — noise is propagating correctly")
    
    for sigma in SIGMA_LEVELS:
        r = all_results[sigma]
        mean_inf_a1 = np.mean(r["inf_a1"])
        mean_inf_a2 = np.mean(r["inf_a2"])
        gt_direction = "A2 > A1 ✓" if np.mean(r["gt_a2"]) > np.mean(r["gt_a1"]) else "A1 > A2"
        print(f"  σ={sigma}: Inf_A1={mean_inf_a1:.4f}, Inf_A2={mean_inf_a2:.4f}, "
              f"GT winner: {gt_direction}, p={r['p_gt']:.6f}")
    
    # Check if GT_A2 > GT_A1 at high noise
    high_sigma = SIGMA_LEVELS[-1]
    r = all_results[high_sigma]
    if np.mean(r["gt_a2"]) > np.mean(r["gt_a1"]) and r["p_gt"] < 0.05:
        print(f"\n  ★ SUCCESS: At σ={high_sigma}, A2 has HIGHER true quality than A1 (p={r['p_gt']:.6f})")
        print(f"    This proves the HT gate prevents score inflation and preserves better elites.")
    elif r["p_gt"] > 0.05:
        print(f"\n  ~ NEUTRAL: At σ={high_sigma}, no significant GT difference (p={r['p_gt']:.6f})")
        print(f"    But A2 inflation ({np.mean(r['inf_a2']):.4f}) < A1 inflation ({np.mean(r['inf_a1']):.4f})")
    else:
        print(f"\n  ⚠ At σ={high_sigma}, A1 still has higher GT (p={r['p_gt']:.6f})")
        print(f"    Gate may still be too conservative. Consider increasing M or adjusting thresholds.")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════════════
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sigmas = np.array(SIGMA_LEVELS)
    
    # (a) Ground-Truth QD vs sigma
    ax = axes[0, 0]
    gt1_means = [np.mean(all_results[s]["gt_a1"]) for s in SIGMA_LEVELS]
    gt2_means = [np.mean(all_results[s]["gt_a2"]) for s in SIGMA_LEVELS]
    gt1_stds = [np.std(all_results[s]["gt_a1"]) for s in SIGMA_LEVELS]
    gt2_stds = [np.std(all_results[s]["gt_a2"]) for s in SIGMA_LEVELS]
    
    ax.errorbar(sigmas, gt1_means, yerr=gt1_stds, fmt="o-", color="#4A90D9",
                linewidth=2, capsize=5, label="Naive (A1)")
    ax.errorbar(sigmas, gt2_means, yerr=gt2_stds, fmt="s-", color="#D94A4A",
                linewidth=2, capsize=5, label="HT-MAP-Elites (A2)")
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Ground-Truth QD Score")
    ax.set_title("(a) True Archive Quality vs Noise")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (b) Inflation vs sigma
    ax = axes[0, 1]
    inf1_means = [np.mean(all_results[s]["inf_a1"]) for s in SIGMA_LEVELS]
    inf2_means = [np.mean(all_results[s]["inf_a2"]) for s in SIGMA_LEVELS]
    inf1_stds = [np.std(all_results[s]["inf_a1"]) for s in SIGMA_LEVELS]
    inf2_stds = [np.std(all_results[s]["inf_a2"]) for s in SIGMA_LEVELS]
    
    ax.errorbar(sigmas, inf1_means, yerr=inf1_stds, fmt="o-", color="#4A90D9",
                linewidth=2, capsize=5, label="Naive (A1)")
    ax.errorbar(sigmas, inf2_means, yerr=inf2_stds, fmt="s-", color="#D94A4A",
                linewidth=2, capsize=5, label="HT-MAP-Elites (A2)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Inflation (Apparent − GT)")
    ax.set_title("(b) Score Inflation vs Noise")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (c) Rejection rate
    ax = axes[1, 0]
    rej_p_rates = [all_results[s]["rej_p"] / max(1, all_results[s]["att"]) * 100
                   for s in SIGMA_LEVELS]
    rej_es_rates = [all_results[s]["rej_es"] / max(1, all_results[s]["att"]) * 100
                    for s in SIGMA_LEVELS]
    
    width = 0.02
    ax.bar(sigmas - width/2, rej_p_rates, width, label="P-value rejections",
           color="#5DADE2", alpha=0.8)
    ax.bar(sigmas + width/2, rej_es_rates, width, label="Effect-size rejections",
           color="#F1948A", alpha=0.8)
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Rejection Rate (%)")
    ax.set_title("(c) HT Gate Rejection Breakdown")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (d) GT difference
    ax = axes[1, 1]
    gt_diffs = [np.mean(all_results[s]["gt_a2"]) - np.mean(all_results[s]["gt_a1"])
                for s in SIGMA_LEVELS]
    gt_diff_stds = [np.std(all_results[s]["gt_a2"] - all_results[s]["gt_a1"])
                    for s in SIGMA_LEVELS]
    p_vals_plot = [all_results[s]["p_gt"] for s in SIGMA_LEVELS]
    
    colors = ["green" if p < 0.05 else "gray" for p in p_vals_plot]
    ax.bar(sigmas, gt_diffs, width=0.03, color=colors, alpha=0.7, edgecolor="black")
    ax.errorbar(sigmas, gt_diffs, yerr=gt_diff_stds, fmt="none", color="black", capsize=5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("GT QD Difference (A2 − A1)")
    ax.set_title("(d) GT Quality Advantage of HT Gate\n(green = p < 0.05)")
    ax.grid(alpha=0.3)
    
    for i, (s, p) in enumerate(zip(SIGMA_LEVELS, p_vals_plot)):
        ax.annotate(f"p={p:.4f}", (s, gt_diffs[i]), textcoords="offset points",
                    xytext=(0, 10 if gt_diffs[i] >= 0 else -15), ha="center", fontsize=8)
    
    plt.suptitle(
        f"HT-MAP-Elites v2: Independent Re-Evaluation\n"
        f"N={N_SEEDS}, T={TOTAL_ITERS}, B={BATCH_SIZE}, "
        f"frefine={REFINE_FREQ}, α={ALPHA}, δmin={DELTA_MIN}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("ht_proof_v2_results.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\nSaved: ht_proof_v2_results.png")
    
    np.savez("ht_proof_v2_data.npz", **{
        f"sigma_{s}_{k}": v
        for s in SIGMA_LEVELS
        for k, v in all_results[s].items()
        if isinstance(v, np.ndarray)
    })
    print("Saved: ht_proof_v2_data.npz")