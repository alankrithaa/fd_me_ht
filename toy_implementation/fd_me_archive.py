import numpy as np

# basic settings
K = 5          # number of sampler steps
B = 4          # batch size (samples per iteration)
T = 5          # number of outer iterations
D = 8          # dimension of latent vector
N_v = 10        # number of fake VLMs
M_per_eval = 5 # number of VLMs used per evaluation
N_BINS = 5     # MAP-Elites grid size per BD dimension

np.random.seed(0)

# fake VLM pool: each VLM has its own weight vector
vlm_weights = [np.random.randn(D) for _ in range(N_v)]


def sampler_S_theta(x0, eps_seq):
    """Toy sampler: iteratively mixes x with Gaussian noise."""
    x = x0.copy()
    for k in range(len(eps_seq)):
        eps_k = eps_seq[k]
        x = 0.8 * x + 0.2 * eps_k
    return x  # x_K


def vlm_score(v_index, x):
    """Toy VLM scoring function."""
    w = vlm_weights[v_index]
    raw = float(np.dot(w, x))
    noise = float(0.1 * np.random.randn())
    return raw + noise


def score_with_vlms(x, m=M_per_eval):
    """Evaluate x with m VLMs sampled from the pool."""
    indices = np.random.choice(N_v, size=m, replace=False)
    scores = []
    for idx in indices:
        s = vlm_score(idx, x)
        scores.append(s)
    return np.array(scores, dtype=float)


def brightness(x):
    """BD1: brightness = mean value of x."""
    return float(np.mean(x))


def entropy(x, num_bins=8):
    """BD2: entropy over histogram of x."""
    hist, _ = np.histogram(x, bins=num_bins, density=True)
    p = hist[hist > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + 1e-12)))


class ArchiveCell:
    def __init__(self, x, bd_norm, scores):
        self.x = x.copy()
        self.bd_norm = np.array(bd_norm, dtype=float)
        self.scores = np.array(scores, dtype=float)  # shape (m,)
        self.mean = float(np.mean(self.scores))


class Archive:
    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.grid = [[None for _ in range(n_bins)] for _ in range(n_bins)]

    def _indices_from_bd(self, bd_norm):
        """bd_norm in [0,1]^2 -> (i,j) grid indices."""
        b1, b2 = bd_norm
        i = int(b1 * self.n_bins)
        j = int(b2 * self.n_bins)
        if i >= self.n_bins:
            i = self.n_bins - 1
        if j >= self.n_bins:
            j = self.n_bins - 1
        return i, j

    def update_naive(self, x, bd_norm, scores):
        """
        Naive replacement:
        - if cell empty: insert
        - else: replace if new mean score is lower (minimisation)
        """
        i, j = self._indices_from_bd(bd_norm)
        new_mean = float(np.mean(scores))
        cell = self.grid[i][j]

        if cell is None or new_mean < cell.mean:
            self.grid[i][j] = ArchiveCell(x, bd_norm, scores)
            return True
        return False

    def occupancy(self):
        """Number of non-empty cells."""
        count = 0
        for row in self.grid:
            for cell in row:
                if cell is not None:
                    count += 1
        return count


def run_fd_me_archive():
    x_hat = np.random.randn(D)  # pseudo-target (not really used yet)

    best_y_global = np.inf
    best_x_global = None

    archive = Archive()

    batch_scalar_all = []
    batch_vector_all = []
    best_y_all = []
    global_best_all = []
    archive_occupancy_all = []

    # running min/max for BD normalisation
    bd1_min = None
    bd1_max = None
    bd2_min = None
    bd2_max = None

    for t in range(1, T + 1):
        batch_scalar = []
        batch_vectors = []
        batch_samples = []

        for b in range(B):
            # sample initial latent and noise sequence
            x0 = np.random.randn(D)
            eps_seq = np.random.randn(K, D)

            # generate sample
            x_K = sampler_S_theta(x0, eps_seq)

            # multi-VLM scores: vector + scalar mean
            scores_vec = score_with_vlms(x_K, m=M_per_eval)
            y_scalar = float(np.mean(scores_vec))

            batch_scalar.append(y_scalar)
            batch_vectors.append(scores_vec)
            batch_samples.append(x_K)

            # behaviour descriptors
            bd1 = brightness(x_K)
            bd2 = entropy(x_K)

            # update global BD ranges
            if bd1_min is None:
                bd1_min = bd1_max = bd1
                bd2_min = bd2_max = bd2
            else:
                bd1_min = min(bd1_min, bd1)
                bd1_max = max(bd1_max, bd1)
                bd2_min = min(bd2_min, bd2)
                bd2_max = max(bd2_max, bd2)

            # normalise BDs to [0,1]
            bd1_norm = (bd1 - bd1_min) / (bd1_max - bd1_min + 1e-8)
            bd2_norm = (bd2 - bd2_min) / (bd2_max - bd2_min + 1e-8)
            bd_norm = (bd1_norm, bd2_norm)

            # archive update (naive mean-based replacement)
            archive.update_naive(x_K, bd_norm, scores_vec)

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

        print(f"[FD-ME] Iteration {t}/{T}")
        print("  Batch scalar scores:", batch_scalar)
        print(f"  Best in batch (scalar): {y_best_batch:.4f}")
        print(f"  Global best so far: {best_y_global:.4f}")
        print(f"  Archive occupancy: {archive.occupancy()} cells")
        print("  Example score vector for best sample:", batch_vectors[idx_best])
        print("-" * 40)

    batch_scalar_all = np.stack(batch_scalar_all, axis=0)        # (T, B)
    batch_vector_all = np.stack(batch_vector_all, axis=0)        # (T, B, M_per_eval)
    best_y_all = np.array(best_y_all)
    global_best_all = np.array(global_best_all)
    archive_occupancy_all = np.array(archive_occupancy_all)

    np.savez(
        "fd_me_logs.npz",
        batch_scalar_all=batch_scalar_all,
        batch_vector_all=batch_vector_all,
        best_y_all=best_y_all,
        global_best_all=global_best_all,
        archive_occupancy_all=archive_occupancy_all,
    )

    print("Finished FD-ME archive run.")
    print(f"Global best objective (scalar): {best_y_global:.4f}")
    print("Best x (first 3 dims):", best_x_global[:3])
    print("Final archive occupancy:", archive.occupancy(), "cells")


if __name__ == "__main__":
    run_fd_me_archive()
