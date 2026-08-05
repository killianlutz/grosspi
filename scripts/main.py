############################################################
################## CONTROL OF NV CENTERS ###################
############################################################
from scripts._config import *

##############################
######## METHOD CHOICE #######
##############################
ControlGP = Neural # Neural/Piecewise/Mixed
csys = ControlGP(static_p)
solve_fn = jax.jit(csys.solve_ocp)

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

##############################
##### REACHABLE TARGET #######
##############################
toy_csys = Mixed(static_p)
dynamic_p = {"target_state": None, "initial_state": psi0, "beta": beta, "epsilon": None}
toy_control = initial_guess(toy_csys, keys[-1], T_guess=jnp.pi)
toy_target = toy_csys.final_state(toy_control, dynamic_p)
psi1 = toy_target/l2_norm(toy_target)

psi0 = jnp.exp(-5*jnp.abs(xs - jnp.pi)**2) + 0*1j
# psi1 = jnp.exp(1j*xs)
psi0, psi1 = jax.tree.map(lambda x: x/l2_norm(x), (psi0, psi1))
##############################
######### SOLVE ##############
##############################
dynamic_p = {"target_state": psi1, "initial_state": psi0, "beta": beta, "epsilon": eps}
init_control = initial_guess(csys, keys[0], T_guess=2.0)
control, losses, metrics, n_iter = solve_fn(init_control, dynamic_p)
losses, metrics = losses[:n_iter + 1], metrics[:n_iter + 1] # not supported inside solve_fn (jit-compilation + n_iter dynamic)
print(f"\n *** Optimization *** \n Iter: {n_iter} \n Loss: {losses[-1]:.1e} \n Infidelity: {metrics[-1, 1]:.1e} \n Gate time: {control[0]:.3f}")

##############################
######### POST PROCESS #######
##############################
csys.plot_optimization(control, dynamic_p, losses, metrics, figsize=(10, 6))
csys.plot_densities(control, dynamic_p, figsize=(10, 6))
csys.plot_pulses(control, dynamic_p, figsize=(10, 6))
csys.plot_pulses(init_control, dynamic_p, figsize=(10, 6))

##############################
#### VALIDATE AND SAVE #######
##############################
pulses = csys.pulses(control, dynamic_p)
orbit = csys.trajectory(control, dynamic_p)
final_state = csys.final_state(control, dynamic_p)

loss_after_optimizer = csys.loss(control, dynamic_p)
loss_check = csys.validate(control, dynamic_p)
print(f"\n *** Loss *** \n Direct: {loss_after_optimizer:.1e} \n Validation: {loss_check:.1e}")

if isinstance(csys, Piecewise):
    filename = "./sims/piecewise.npz"
elif isinstance(csys, Neural):
    filename = "./sims/neural.npz"
else:
    filename = "./sims/else.npz"
csys.save_to_npz(filename, control, dynamic_p)
# data = jnp.load(filename, allow_pickle=True)