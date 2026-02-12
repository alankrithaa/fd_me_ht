import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

print(f"JAX Device: {jax.devices()}")

# A tiny JAX operation
x = jnp.array([1.0, 2.0, 3.0])
print(f"JAX Array: {x * 2}")

print("QDAX and JAX are ready to go!")