import numpy as np
from scipy.stats import mannwhitneyu

# optional QDax hook: used later if you want to swap to a real Map-Elites repertoire
try:
    from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
except ImportError:
    compute_euclidean_centroids = None

K = 5          # number of sampler steps
B = 4          # batch size (samples per iteration)
T = 5          # number of outer iterations
D = 8          # dimension of latent vector
N_v = 10        # number of fake VLMs
M_per_eval = 5 # number of VLMs used per evaluation
N_BINS = 5     # grid size for each BD dimension

np.random.seed(0)

vlm_weights = [np.random.randn(D) for _ in range(N_v)]


def sampler_S_theta(x0, eps_seq):
    x = x0.copy()
    for k in range(len(eps_seq)):
        eps_k = eps_seq[k]
        x = 0.8 * x + 0.2 * eps_k
    return x


def vlm_score(v_index, x):
    w = vlm_weights[v_index]
    raw = float(np.dot(w, x))
    noise = float(0.1 * np.random.randn())
    return raw + noise


def score_with_vlms(x, m=M_per_eval):
    indices = np.random.choice(N_v, size=m, replace=False)
    scores = []
    for idx in indices:
        s = vlm_score(idx, x)
        scores.append(s)
    return np.array(scores, dtype=float)


def brightness(x):
    return float(np.mean(x))


def entropy(x, num_bins=8):
    hist, _ = np.histogram(x, bins=num_bins, density=True)
    p = hist[hist > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + 1e-12)))


class ArchiveCell:
    def __init__(self, x, bd_norm, scores):
        self.x = x.copy()
        self.bd_norm = np.array(bd_norm, dtype=float)
        self.scores = np.array(scores, dtype=float)  # vector of VLM scores
        self.mean = float(np.mean(self.scores))


class Archive:
    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.grid = [[None for _ in range(n_bins)] for _ in range(n_bins)]

    def _indices_from_bd(self, bd_norm):
        b1, b2 = bd_norm
        i = int(b1 * self.n_bins)
        j = int(b2 * self.n_bins)
        if i >= self.n_bins:
            i = self.n_bins - 1
        if j >= self.n_bins:
            j = self.n_bins - 1
        return i, j

    def update_ht(self, x, bd_norm, scores, alpha=0.05, delta=0.60):
        """
        Hypothesis-testing replacement rule (Mann–Whitney + CLES).
        Returns: (replaced, i, j, p_value, effect_size)
        """
        i, j = self._indices_from_bd(bd_norm)
        candidate = np.array(scores, dtype=float)
        cell = self.grid[i][j]

        # if cell empty: insert without a test
        if cell is None:
            self.grid[i][j] = ArchiveCell(x, bd_norm, candidate)
            return True, i, j, None, None

        incumbent = cell.scores
        m = candidate.size
        n = incumbent.size
        if m == 0 or n == 0:
            return False, i, j, None, None

        # we want candidate scores to be smaller (better), hence alternative="less"
        res = mannwhitneyu(candidate, incumbent, alternative="less")
        U = float(res.statistic)
        p = float(res.pvalue)

        # common language effect size: P(candidate < incumbent) = U / (m * n)
        effect_size = 1.0 - (U / (m * n))

        if p < alpha and effect_size >= delta:
            self.grid[i][j] = ArchiveCell(x, bd_norm, candidate)
            return True, i, j, p, effect_size

        return False, i, j, p, effect_size

    def occupancy(self):
        count = 0
        for row in self.grid:
            for cell in row:
                if cell is not None:
                    count += 1
        return count

    def get_all_elites(self):
        xs = []
        bds = []
        means = []
        for i in range(self.n_bins):
            for j in range(self.n_bins):
                cell = self.grid[i][j]
                if cell is not None:
                    xs.append(cell.x)
                    bds.append(cell.bd_norm)
                    means.append(cell.mean)
        if not xs:
            return None, None, None
        return np.array(xs), np.array(bds), np.array(means)


def run_fd_me_ht():
    x_hat = np.random.randn(D)

    best_y_global = np.inf
    best_x_global = None

    archive = Archive()

    batch_scalar_all = []
    batch_vector_all = []
    best_y_all = []
    global_best_all = []
    archive_occupancy_all = []
    replacements_per_iter = []
    mean_p_per_iter = []
    mean_effect_per_iter = []

    bd1_min = None
    bd1_max = None
    bd2_min = None
    bd2_max = None

    for t in range(1, T + 1):
        batch_scalar = []
        batch_vectors = []
        batch_samples = []

        reps_this_iter = 0
        p_vals = []
        effects = []

        for b in range(B):
            x0 = np.random.randn(D)
            eps_seq = np.random.randn(K, D)

            x_K = sampler_S_theta(x0, eps_seq)

            scores_vec = score_with_vlms(x_K, m=M_per_eval)
            y_scalar = float(np.mean(scores_vec))

            batch_scalar.append(y_scalar)
            batch_vectors.append(scores_vec)
            batch_samples.append(x_K)

            bd1 = brightness(x_K)
            bd2 = entropy(x_K)

            if bd1_min is None:
                bd1_min = bd1_max = bd1
                bd2_min = bd2_max = bd2
            else:
                bd1_min = min(bd1_min, bd1)
                bd1_max = max(bd1_max, bd1)
                bd2_min = min(bd2_min, bd2)
                bd2_max = max(bd2_max, bd2)

            bd1_norm = (bd1 - bd1_min) / (bd1_max - bd1_min + 1e-8)
            bd2_norm = (bd2 - bd2_min) / (bd2_max - bd2_min + 1e-8)
            bd_norm = (bd1_norm, bd2_norm)

            replaced, i, j, p, effect = archive.update_ht(
                x_K, bd_norm, scores_vec, alpha=0.05, delta=0.60
            )

            if replaced:
                reps_this_iter += 1
            if p is not None:
                p_vals.append(p)
            if effect is not None:
                effects.append(effect)

        batch_scalar = np.array(batch_scalar)
        batch_vectors = np.stack(batch_vectors, axis=0)
        batch_samples = np.array(batch_samples)

        idx_best = int(np.argmin(batch_scalar))
        y_best_batch = batch_scalar[idx_best]
        x_best_batch = batch_samples[idx_best]

        x_hat = x_best_batch.copy()

        if y_best_batch < best_y_global:
            best_y_global = y_best_batch
            best_x_global = x_best_batch.copy()

        batch_scalar_all.append(batch_scalar)
        batch_vector_all.append(batch_vectors)
        best_y_all.append(y_best_batch)
        global_best_all.append(best_y_global)
        archive_occupancy_all.append(archive.occupancy())
        replacements_per_iter.append(reps_this_iter)

        if p_vals:
            mean_p_per_iter.append(float(np.mean(p_vals)))
        else:
            mean_p_per_iter.append(np.nan)

        if effects:
            mean_effect_per_iter.append(float(np.mean(effects)))
        else:
            mean_effect_per_iter.append(np.nan)

        print(f"[FD-ME-HT] Iteration {t}/{T}")
        print("  Batch scalar scores:", batch_scalar)
        print(f"  Best in batch (scalar): {y_best_batch:.4f}")
        print(f"  Global best so far: {best_y_global:.4f}")
        print(f"  Archive occupancy: {archive.occupancy()} cells")
        print(f"  Replacements this iter: {reps_this_iter}")
        if effects:
            print(f"  Mean effect size this iter: {np.mean(effects):.3f}")
        print("-" * 40)

    batch_scalar_all = np.stack(batch_scalar_all, axis=0)
    batch_vector_all = np.stack(batch_vector_all, axis=0)
    best_y_all = np.array(best_y_all)
    global_best_all = np.array(global_best_all)
    archive_occupancy_all = np.array(archive_occupancy_all)
    replacements_per_iter = np.array(replacements_per_iter)
    mean_p_per_iter = np.array(mean_p_per_iter)
    mean_effect_per_iter = np.array(mean_effect_per_iter)

    elites_x, elites_bd, elites_mean = archive.get_all_elites()

    np.savez(
        "fd_me_ht_logs.npz",
        batch_scalar_all=batch_scalar_all,
        batch_vector_all=batch_vector_all,
        best_y_all=best_y_all,
        global_best_all=global_best_all,
        archive_occupancy_all=archive_occupancy_all,
        replacements_per_iter=replacements_per_iter,
        mean_p_per_iter=mean_p_per_iter,
        mean_effect_per_iter=mean_effect_per_iter,
        elites_x=elites_x,
        elites_bd=elites_bd,
        elites_mean=elites_mean,
    )

    print("Finished FD-ME-HT run.")
    print(f"Global best objective (scalar): {best_y_global:.4f}")
    print("Best x (first 3 dims):", best_x_global[:3])
    print("Final archive occupancy:", archive.occupancy(), "cells")


if __name__ == "__main__":
    run_fd_me_ht()
