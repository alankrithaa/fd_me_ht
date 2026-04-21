"""Compatibility wrapper for pairwise Cohen's d analysis.

The actual implementation now lives in diversity_metrics.py so the metric is
defined in one place only.
"""

from diversity_metrics import compute_pairwise_cohens_d

__all__ = ["compute_pairwise_cohens_d"]


if __name__ == "__main__":
    raise SystemExit(
        "compute_pairwise_cohens_d now lives in diversity_metrics.py. "
        "Import it from there or use mutation_fitness.py for the heatmap script."
    )
