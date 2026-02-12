import numpy as np
import matplotlib.pyplot as plt
from fd_me_ht import Archive, sampler_S_theta, score_with_vlms, brightness, entropy, D, K, B, T

def run_experiment(alpha_val, delta_val):
    """Runs a single simulation with specific HT parameters."""
    archive = Archive()
    total_replacements = 0
    potential_replacements = 0
    
    # Global BD scaling (simplified for analysis)
    bd_min, bd_max = -2.0, 2.0 

    for t in range(T):
        for b in range(B):
            # 1. Generate sample [cite: 113, 236]
            x0 = np.random.randn(D)
            eps_seq = np.random.randn(K, D)
            x_K = sampler_S_theta(x0, eps_seq)

            # 2. Score and Descriptors [cite: 144, 151]
            scores_vec = score_with_vlms(x_K)
            bd1_norm = np.clip((brightness(x_K) - bd_min) / (bd_max - bd_min), 0, 1)
            bd2_norm = np.clip((entropy(x_K) - 0) / (3.0), 0, 1) # Entropy range approx 0-3
            
            # 3. HT Update [cite: 158, 193]
            # Track if cell was occupied to calculate 'Replacement Rate'
            i, j = archive._indices_from_bd((bd1_norm, bd2_norm))
            if archive.grid[i][j] is not None:
                potential_replacements += 1

            replaced, _, _, _, _ = archive.update_ht(
                x_K, (bd1_norm, bd2_norm), scores_vec, alpha=alpha_val, delta=delta_val
            )
            
            if replaced:
                total_replacements += 1

    # Calculate QD Score (Sum of fitness of all elites) [cite: 11, 156]
    _, _, means = archive.get_all_elites()
    qd_score = np.sum(means) if means is not None else 0
    replacement_rate = total_replacements / (potential_replacements + 1e-8)
    
    return archive.occupancy(), qd_score, replacement_rate

# --- Sensitivity Sweep ---
alphas = [0.01, 0.05, 0.1]
deltas = [0.147, 0.33, 0.49, 0.60] # [cite: 314, 347]
results = []

print("Starting Sensitivity Analysis...")
print(f"{'Alpha':<8} | {'Delta':<8} | {'Occupancy':<10} | {'QD Score':<10} | {'Rep. Rate'}")
print("-" * 60)

for a in alphas:
    for d in deltas:
        occ, qd, rep = run_experiment(a, d)
        results.append((a, d, occ, qd, rep))
        print(f"{a:<8} | {d:<8} | {occ:<10} | {qd:<10.2f} | {rep:.2%}")

# --- Visualization ---
# Example: Plotting Occupancy vs Delta for different Alphas
plt.figure(figsize=(10, 6))
for a in alphas:
    subset = [r for r in results if r[0] == a]
    plt.plot([r[1] for r in subset], [r[2] for r in subset], marker='o', label=f'Alpha={a}')

plt.title("Effect of HT Thresholds on Archive Occupancy")
plt.xlabel("Effect Size Threshold (Delta)")
plt.ylabel("Cells Filled")
plt.legend()
plt.grid(True)
plt.savefig("sensitivity_results.png")
plt.show()