import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt

# Import your custom modules
from repertoire import DistributionalRepertoire, compute_cohens_d
from ht_logic import calculate_ht_replacement
from evaluator import evaluate_via_pytorch

# --- 1. SETTINGS & HYPERPARAMETERS ---
GRID_SIZE = 5      # 5x5 Grid
LATENT_DIM = 8     # Dimension of your noise/prompt vector
NUM_RATERS = 5     # Number of VLM ratings per evaluation
ITERATIONS = 20    # Number of candidates to test in this run
IMG_RES = 64       # Resolution for the 'toy' image storage

# --- 2. CORE LOGIC ---

@jax.jit
def get_grid_indices(bd_coordinates, num_bins=GRID_SIZE):
    """Maps normalized BDs [0, 1] to Grid Indices [0, 4]."""
    indices = jnp.floor(bd_coordinates * num_bins).astype(jnp.int32)
    return jnp.clip(indices, 0, num_bins - 1)

def update_step(repertoire, candidate_data):
    """Process a single candidate and update the archive."""
    gen, phen, scores, bd = candidate_data
    idx = tuple(get_grid_indices(bd))
    
    # Check if cell is empty
    is_empty = (repertoire.occupancy[idx] == 0)
    
    # Run Hypothesis Testing logic against current elite
    # calculate_ht_replacement returns (should_replace, cles)
    should_replace_ht, cles = calculate_ht_replacement(
        scores, repertoire.scores[idx], alpha=0.05, delta_min=0.6
    )
    
    # Replacement logic: Replace if empty OR if it passes the HT test
    do_replace = is_empty | should_replace_ht
    
    # Update the Repertoire (JAX PyTree update)
    new_repertoire = repertoire._replace(
        genotypes=repertoire.genotypes.at[idx].set(jnp.where(do_replace, gen, repertoire.genotypes[idx])),
        phenotypes=repertoire.phenotypes.at[idx].set(jnp.where(do_replace, phen, repertoire.phenotypes[idx])),
        scores=repertoire.scores.at[idx].set(jnp.where(do_replace, scores, repertoire.scores[idx])),
        occupancy=repertoire.occupancy.at[idx].set(1),
        # Metrics Logging for Tech Report
        attempt_counts=repertoire.attempt_counts.at[idx].add(1),
        acceptance_counts=repertoire.acceptance_counts.at[idx].add(do_replace.astype(int)),
        sum_effect_sizes=repertoire.sum_effect_sizes.at[idx].add(jnp.where(do_replace, cles, 0.0))
    )
    
    return new_repertoire

def save_elites_to_disk(repertoire, folder="results/elites"):
    """Exports all stored images to the results folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    indices = jnp.argwhere(repertoire.occupancy > 0)
    print(f"\n--- Exporting {len(indices)} Elites ---")
    
    for idx in indices:
        r, c = int(idx[0]), int(idx[1])
        img_data = repertoire.phenotypes[r, c]
        fname = f"{folder}/elite_R{r}_C{c}.png"
        plt.imsave(fname, np.array(img_data))
    
    print(f"Done. Images saved in: {folder}")

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    # Initialize the empty Distributional Repertoire
    repertoire = DistributionalRepertoire.init(GRID_SIZE, LATENT_DIM, NUM_RATERS, IMG_RES)
    
    print(f"Starting QDAX Pipeline with {ITERATIONS} iterations...")

    for t in range(ITERATIONS):
        # A. Create a dummy candidate
        # In the real experiment, this is your Diffusion Sampler output
        candidate_gen = jnp.array(np.random.randn(LATENT_DIM))
        candidate_phen = jnp.array(np.random.rand(IMG_RES, IMG_RES, 3)) 
        candidate_bd = jnp.array([np.random.rand(), np.random.rand()]) # [Brightness, Entropy]
        
        # B. Get Multi-Rater scores via the PyTorch Bridge
        # This calls the evaluate_via_pytorch function in evaluator.py
        candidate_scores = evaluate_via_pytorch(jnp.expand_dims(candidate_phen, 0))[0]
        
        # C. Run the update step
        candidate_data = (candidate_gen, candidate_phen, candidate_scores, candidate_bd)
        repertoire = update_step(repertoire, candidate_data)
        
        if (t + 1) % 5 == 0:
            print(f"  Iteration {t+1}: Current Occupancy = {int(jnp.sum(repertoire.occupancy))}")

    # D. Finalize and Save
    save_elites_to_disk(repertoire)