import jax.numpy as jnp
from jax import jit, grad
import jax
#print(f"Device f{jax.devices()[0]}")

AM = jnp.array([[0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0]
               ])

def add_e(s, t, am):
    updated = am.at[s, t].set(1)
    updated = updated.at[t, s].set(1)
    return updated

def sub_e(s, t, am):
    updated = am.at[s, t].set(0)
    updated = updated.at[t, s].set(0)
    return updated

print(f"Original Array\n{AM}")
add_e = jit(add_e)
sub_e = jit(sub_e)

for i in range(len(AM)-1):
    AM = add_e(i, i+1, AM)

print(f"Updated Array\n{AM}")
for i in range(len(AM)-1):
    AM = sub_e(i, i+1, AM)
print(f"Updated after sub\n{AM}")

