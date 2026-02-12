import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("fd_me_ht_logs.npz", allow_pickle=True)

    best_y_all = data["best_y_all"]
    global_best_all = data["global_best_all"]
    archive_occupancy_all = data["archive_occupancy_all"]
    replacements_per_iter = data["replacements_per_iter"]
    mean_effect_per_iter = data["mean_effect_per_iter"]

    plt.figure()
    plt.plot(best_y_all, marker="o", label="Best in batch")
    plt.plot(global_best_all, marker="x", label="Global best")
    plt.title("FD-ME-HT Scores per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Scalar score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(archive_occupancy_all, marker="o")
    plt.title("Archive Occupancy Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Number of filled cells")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(range(len(replacements_per_iter)), replacements_per_iter)
    plt.title("Number of Replacements per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Replacements")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(mean_effect_per_iter, marker="o")
    plt.title("Mean Effect Size (CLES) per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Effect size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
