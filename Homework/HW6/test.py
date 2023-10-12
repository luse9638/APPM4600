import jax
import jax.numpy as jnp
from HW6 import nDLazyNewton, jacobian, nDBroyden, gradient

def F(x: jnp.array):
    return jnp.array(x[0] ** 2 * x[1] - 5 * x[0] + x[1])

print(gradient(F, jnp.array([1., 1.])))

