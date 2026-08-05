############################################################
################## CONTROL OF NV CENTERS ###################
############################################################
from scripts._config import *

def initial_guess(control_system, key=keys[0], T_guess=None):
    K = jax.random.split(key, 3)
    T = jax.random.uniform(K[-1])*2*jnp.pi if T_guess is None else T_guess  # time horizon
    n_pieces = n // 10
    if isinstance(control_system, Piecewise):
        n_components = jnp.size(static_p["system"]["ctrl"][0], 0)
        u = jax.random.normal(K[-2])*jnp.ones((n_pieces, n_components)) # u
        return T, (u,)
    elif isinstance(control_system, Neural):
        neurons = jnp.array([1, 6, 8, 8, 6, 2])
        u = rand_weights(K[-2], neurons)
        return T, (u,)
    elif isinstance(control_system, Mixed):
        neurons = jnp.array([1, 8, 8, 1])
        radius = jnp.ones((n_pieces, 1))
        angle = rand_weights(K[-2], neurons)
        u = (radius, angle)
        return T, (u,)
    else:
        raise ValueError("only available classes: Piecewise, Neural, Mixed")

def split_batch(results_batch, i):
    control = jax.tree.map(lambda x: x[i], results_batch[0])
    n_iter = results_batch[3][i]
    losses = results_batch[1][:n_iter + 1, i]
    metrics = results_batch[2][:n_iter + 1, i, :]
    print(f"\n ** Batch {i} ** \n n: {n_iter} \n J: {losses[-1]:.1e} \n I: {metrics[-1, 1]:.1e} \n T: {control[0]:.3f}")
    return control, losses, metrics, n_iter

##############################
######## METHOD CHOICE #######
##############################
ControlGP = Neural # Neural/Piecewise/Mixed
batch_size = 5
csys = ControlGP(static_p)
batch_solve_fn = jax.jit(jax.vmap(csys.solve_ocp, (0, None)))
dynamic_p = {"target_state": psi1, "initial_state": psi0, "beta": beta, "epsilon": eps}
init_control_batch = jax.vmap(initial_guess, in_axes=(None, 0))(csys, keys[:batch_size])

##############################
######### SOLVE ##############
##############################
##### compilation time included
out = batch_solve_fn(init_control_batch, dynamic_p)

##### split batch results to analyze each run
split = tuple(split_batch(out, i) for i in range(batch_size))
batch_metrics = jnp.stack(tuple(batch[2][-1] for batch in split))

##### retrieve most promising run
batch_index = 4
control = split[batch_index][0]
losses, metrics = split[batch_index][1:3]
csys.plot_optimization(control, dynamic_p, losses, metrics)
# plt.savefig("./sims/optimization", dpi=300, bbox_inches='tight')
csys.plot_densities(control, dynamic_p)
# plt.savefig(".sims/densities", dpi=300, bbox_inches='tight')
csys.plot_pulses(control, dynamic_p)
# plt.savefig("./sims/pulses", dpi=300, bbox_inches='tight')