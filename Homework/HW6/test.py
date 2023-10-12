import jax
import jax.numpy as jnp
from HW6 import nDLazyNewton, jacobian, nDBroyden

def F(x: jnp.array):
    F = []
    F.append(3 * x[0] - jnp.cos(x[1] * x[2]) - (1 / 2))
    F.append(x[0] - 81 * (x[1] + 0.1) ** 2 + jnp.sin(x[2]) + 1.06)
    F.append(jnp.exp(-1 * x[0] * x[1]) + 20 * x[2] + (10 * jnp.pi - 3) / 3)
    return jnp.array(F)

evaluation = nDBroyden(jnp.array([0.1, 0.1, -0.1]), F, jnp.float32(1e-10), 10)
for item in evaluation:
    print(item) 