import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("fd_logs.npz")

    batch_scores_all = data["batch_scores_all"]      # shape (T, B)
    best_y_all = data["best_y_all"]                  # shape (T,)
    global_best_all = data["global_best_all"]        # shape (T,)

    # Plot 1: best score per iteration
    plt.figure()
    plt.plot(best_y_all, marker="o")
    plt.title("Best Score in Each FD Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Batch Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: global best progression
    plt.figure()
    plt.plot(global_best_all, marker="o")
    plt.title("Global Best Score Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Global Best Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: batch score distributions as boxplot
    plt.figure()
    plt.boxplot(batch_scores_all)
    plt.title("Batch Score Distribution per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
