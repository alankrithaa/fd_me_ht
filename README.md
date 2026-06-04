**Improved Quality-Diversity for Noisy Fitness Functions using Hypothesis-Testing Replacement Gates.**

This repository implements **HT-MAP-Elites** (also referred to as **DROME**), a distributionally robust Quality-Diversity (QD) framework engineered for stochastic optimization environments—such as multi-rater ensembles or Vision-Language Model (VLM) feedback pipelines. 

By replacing naive mean-score archive updates with a rigorous, statistically-grounded **Hypothesis Testing (HT) Replacement Gate**, this architecture eliminates systemic *score inflation* and *archive drift* caused by high-variance noise spikes ("winner's curse").

---

## 💡 Key Concepts & Core Architecture

Traditional MAP-Elites setups suffer from severe score inflation in stochastic regimes because lucky noise spikes permanently overwrite high-quality incumbents. HT-MAP-Elites addresses this vulnerability at the decision layer through two core components:

1. **Concurrent Leader-Follower Architecture:**
   - **Exploratory Scout Archive ($A_1$):** Prioritizes rapid behavioral discovery using a standard, high-turnover mean-based update.
   - **Robust Judge Archive ($A_2$):** Acts as the certified high-performance repository, intermittently pulling elites from $A_1$ at a designated `Refine Frequency`.
2. **Three-Condition Statistical Update Gate:**
   To successfully displace an incumbent in the Judge archive, a challenger candidate must strictly pass:
   - **Directional Improvement:** $\overline{r}_{\text{new}} < \overline{r}_{\text{old}}$ (lower score = better perceptual alignment).
   - **Statistical Significance ($p < \alpha$):** Verified via a JAX-native, exact two-tailed $Z$-test accounting for distribution variance and rater count ($M$).
   - **Practical Significance ($\hat{\delta} > \delta_{\text{min}}$):** Quantified using the Common Language Effect Size (CLES) to avoid minor background gains.

---

## 📁 Repository Structure

```text
├── qdax_imp/                  # JAX/QDAX Hardware-Accelerated Production Core
│   ├── ht_logic.py            # JAX-native, JIT-compiled exact Z-test & CLES gate
│   ├── repertoire.py          # DistributionalRepertoire extending MapElitesRepertoire
│   ├── fdme_map_elites.py     # Main DROMEMAPElites dual-archive lifecycle manager
│   ├── fdme_emitter.py        # Vectorized genetic exploration state machine
│   ├── evaluator.py           # Synthetic multi-rater VLM preference environment
│   ├── cohens_d.py            # Inter-elite diversity & effect-size validation utilities
│   ├── diversity_metrics.py   # Repertoire coverage and BD-spread analytics
│   └── check_jax.py           # Verification script for GPU/XLA environments
│
└── toy_implementation/        # NumPy & SciPy Lightweight Prototyping Playground
    ├── fd_baseline.py         # Standard Fast-Direct generative guidance loop
    ├── fd_me_archive.py       # Deterministic mean-based baseline archive
    ├── fd_me_ht.py            # Non-parametric Mann-Whitney U update loop prototype
    └── sensitivity_analysis.py# Threshold sweeping script for α and δ settings
