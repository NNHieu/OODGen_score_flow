import jax
import jax.numpy as jnp

data = jnp.arange(4) if jax.process_index() == 0 else jnp.arange(4, 8)
print(data)
f = lambda x : x  / jax.lax.psum(x, axis_name = 'i')
a = jax.pmap(f, axis_name='i')(data)
print(a)
