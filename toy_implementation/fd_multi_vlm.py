import numpy as np

K = 5          # number of sampler steps
B = 4          # batch size (samples per iteration)
T = 5          # number of outer iterations
D = 8          # dimension of latent vector
N_v = 3        # number of fake VLMs

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


def score_with_vlms(x, m=2):
    indices = np.random.choice(N_v, size=m, replace=False)
    scores = []
    for idx in indices:
        s = vlm_score(idx, x)
        scores.append(s)
    return scores


def f(x):
    scores = score_with_vlms(x, m=2)
    mean_score = float(np.mean(scores))
    return mean_score


def run_fd_multi_vlm():
    x_hat = np.random.randn(D)

    best_y_global = np.inf
    best_x_global = None

    batch_scores_all = []
    best_y_all = []
    global_best_all = []

    for t in range(1, T + 1):
        batch_scores = []
        batch_samples = []

        for b in range(B):
            x0 = np.random.randn(D)
            eps_seq = np.random.randn(K, D)

            x_K = sampler_S_theta(x0, eps_seq)
            y = f(x_K)

            batch_scores.append(y)
            batch_samples.append(x_K)

        batch_scores = np.array(batch_scores)
        batch_samples = np.array(batch_samples)

        idx_best = int(np.argmin(batch_scores))
        y_best_batch = batch_scores[idx_best]
        x_best_batch = batch_samples[idx_best]

        x_hat = x_best_batch.copy()

        if y_best_batch < best_y_global:
            best_y_global = y_best_batch
            best_x_global = x_best_batch.copy()

        batch_scores_all.append(batch_scores)
        best_y_all.append(y_best_batch)
        global_best_all.append(best_y_global)

        print(f"[multi-VLM] Iteration {t}/{T}")
        print("  Batch scores:", batch_scores)
        print(f"  Best in batch: {y_best_batch:.4f}")
        print(f"  Global best so far: {best_y_global:.4f}")
        print("  Example x_K (first 3 dims):", x_best_batch[:3])
        print("-" * 30)

    batch_scores_all = np.stack(batch_scores_all, axis=0)
    best_y_all = np.array(best_y_all)
    global_best_all = np.array(global_best_all)

    np.savez(
        "fd_logs_multi_vlm.npz",
        batch_scores_all=batch_scores_all,
        best_y_all=best_y_all,
        global_best_all=global_best_all,
    )

    print("Finished FD multi-VLM run.")
    print(f"Global best objective: {best_y_global:.4f}")
    print("Best x (first 3 dims):", best_x_global[:3])


if __name__ == "__main__":
    run_fd_multi_vlm()
