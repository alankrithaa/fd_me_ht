import torch
import numpy as np
import jax
import jax.numpy as jnp
from jax import pure_callback

def torch_vlm_scorer(images_numpy):
    # FIX: Ensure images are NumPy before passing to Torch
    images_np = np.asarray(images_numpy)
    
    # Convert to Torch tensor
    images_torch = torch.from_numpy(images_np).float()
    
    # Simulate VLM ratings: (Batch, M_ratings)
    # This is where your actual VLM model would run
    with torch.no_grad():
        # Generating random scores for 5 VLMs
        scores = torch.randn(images_np.shape[0], 5) 
    
    return scores.numpy().astype(np.float32)

def evaluate_via_pytorch(images):
    """
    Wraps the PyTorch logic so JAX can call it during the QD loop.
    """
    result_shape = jax.ShapeDtypeStruct(
        shape=(images.shape[0], 5), 
        dtype=jnp.float32
    )
    
    # pure_callback handles the transition from JAX to Python/Torch
    return pure_callback(torch_vlm_scorer, result_shape, images)