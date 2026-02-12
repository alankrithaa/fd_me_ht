import numpy as np


# Simple hyperparameters


K = 5          # number of sampler steps
B = 4          # batch size (samples per iteration)
T = 5          # number of outer iterations
D = 8          # dimension of "latent" vector (toy x)

# fix random seed for reproducibility
np.random.seed(0)



# Toy sampler S_theta

# This just nudges x toward random noise K times.
# In real FD, this would be the diffusion sampler.

def sampler_S_theta(x0, eps_seq):
    x = x0.copy()
    for k in range(K):
        eps_k = eps_seq[k]
        # very simple "update": mix current x with noise
        x = 0.8 * x + 0.2 * eps_k
    return x  # this is x_K



# Toy objective f(x)
# We pretend there is a hidden "good" target x* (pseudo-target).
# f(x) = squared distance to x_star (lower is better).

x_star = np.random.randn(D)

def f(x):
    diff = x - x_star
    return float(np.sum(diff * diff))



# Fast-Direct-like loop (Algo 2 style)


def run_fd_baseline():
    # pseudo-target x_hat (not really used yet, but we keep it)
    x_hat = np.random.randn(D)

    best_y_global = np.inf
    best_x_global = None

    for t in range(1, T + 1):
        batch_scores = []
        batch_samples = []

        # ----- generate B samples in this iteration -----
        for b in range(B):
            # sample initial latent x0 and noise sequence eps_1..eps_K
            x0 = np.random.randn(D)
            eps_seq = np.random.randn(K, D)

            # run sampler to get x_K
            x_K = sampler_S_theta(x0, eps_seq)

            # evaluate objective
            y = f(x_K)

            batch_scores.append(y)
            batch_samples.append(x_K)

        batch_scores = np.array(batch_scores)
        batch_samples = np.array(batch_samples)

        # find best in this batch
        idx_best = int(np.argmin(batch_scores))
        y_best_batch = batch_scores[idx_best]
        x_best_batch = batch_samples[idx_best]

        # update pseudo-target (very simple rule: best of batch)
        x_hat = x_best_batch.copy()

        # update global best
        if y_best_batch < best_y_global:
            best_y_global = y_best_batch
            best_x_global = x_best_batch.copy()

        # print progress to confirm things work 
        print("  Batch scores:", batch_scores)
        print(f"  Best in batch: {y_best_batch:.4f}")
        print(f"  Global best so far: {best_y_global:.4f}")
        print("  Example x_K (first 3 dims):", x_best_batch[:3])
        print("-" * 40)

    print("Finished FD baseline run.")
    print(f"Global best objective: {best_y_global:.4f}")
    print("Best x (first 3 dims):", best_x_global[:3])


if __name__ == "__main__":
    run_fd_baseline()
