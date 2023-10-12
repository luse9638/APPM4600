import jax
import jax.numpy as jnp

def f(x):
    return x[0] ** 2 + x[1] ** 2

print(jax.jacfwd(f)(jnp.array([1., 1.])))