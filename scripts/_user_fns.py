from src._classes import ControlSystem
from src._tools import *
from src._networks import network
import matplotlib.pyplot as plt

###############################
##### AUXILIARY FUNCTIONS #####
###############################
def runge_kutta(velocity, _, x, t, p):
    h = p[-1]["integrator"]["h"]
    k1 = velocity(x           , t + 0.0*h, p)
    k2 = velocity(x + 0.5*h*k1, t + 0.5*h, p)
    k3 = velocity(x + 0.5*h*k2, t + 0.5*h, p)
    k4 = velocity(x +     h*k3, t + 1.0*h, p)
    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6

def strang_split(_, flows, x, t, p):
    h = p[-1]["integrator"]["h"]
    flow_a, flow_b = flows # A = diffusion, B = potential
    y = flow_b(x, t, t + 0.5*h, p)
    z = flow_a(y, t, t + h, p)
    w = flow_b(z, t + 0.5*h, t + h, p)
    return w

def loss_fn(psi_final, p):
    control, dynamic_p, _ = p
    T = control[0]
    eps = dynamic_p["epsilon"]
    psi1 = dynamic_p["target_state"]
    return eps*T/(1 + T) + (1 - eps)*infidelity(psi_final, psi1)

###############################
########## PWC ################
###############################
class GrossPitaevskii(ControlSystem):
    def __init__(self, static_parameters):
        super().__init__(static_parameters)

    def eval_metrics(self, control, dynamic_p):
        psi_final = self.final_state(control, dynamic_p)
        time_horizon = control[0]
        dist_to_target = infidelity(psi_final, dynamic_p["target_state"])
        energy = gp_energy(psi_final, dynamic_p["beta"])
        return jnp.array([time_horizon, dist_to_target, energy])

    def flow_a(self, psi, t0, t1, p):
        # DIFFUSION STEP
        control, _, static_p = p
        T = control[0]
        dt = (t1 - t0)*T
        dx = static_p["space"]["hx"]
        omega = 2 * jnp.pi * fftfreq(jnp.size(psi), dx)
        return ifft(jnp.exp(-1j * dt * omega ** 2) * fft(psi, norm="ortho"), norm="ortho")

    def flow_b(self, psi, t0, t1, p):
        # POTENTIAL STEP
        control, dynamic_p, static_p = p
        T = control[0]
        dt = (t1 - t0)*T
        beta = dynamic_p["beta"]
        C, S = static_p["system"]["ctrl"][0]  # cos and sine evaluated on mesh points

        # compute potential V(t^n, x_j) (---> zeroth-order hold)
        pulses = self.params_to_pulses(t0, control, dynamic_p, psi)[0]
        controllable = -0.5 * (pulses[0] * C + pulses[1] * S)
        drift = beta * jnp.abs(psi) ** 2
        potential = controllable + drift

        return jnp.exp(-1j * dt * potential)*psi

    def vector_field(self, psi, t, p):
        # p = (control, dynamic_p, static_p)
        control, dynamic_p, static_p = p
        beta = dynamic_p["beta"]
        C, S = static_p["system"]["ctrl"][0] # cos and sine evaluated on mesh points

        time_horizon = control[0]
        pulses = self.params_to_pulses(t, control, dynamic_p, psi)[0]
        controllable = -0.5*( pulses[0]*C + pulses[1]*S )
        drift = beta * jnp.abs(psi)**2

        potential = controllable + drift
        return -1j * time_horizon * (-mvp_laplacian(psi) + potential * psi)

    def plot_optimization(self, control, dynamic_p, losses, metrics, **kwargs):
        fig, axes = plt.subplots(1, 1, **kwargs)

        axes.semilogy(losses, '-x', color="k", label=r"$J$")
        axes.semilogy(metrics[:, 0], linewidth=2, label=r"$T$")
        axes.semilogy(metrics[:, 1], linewidth=2, label=r"$IF$")
        axes.semilogy(metrics[:, 2], linewidth=2, label=r"$E$")
        axes.set_xlabel("Iteration")
        axes.grid(True)
        axes.legend(loc='upper left')

        plt.tight_layout()
        fig.show()


class Piecewise(GrossPitaevskii):
    def __init__(self, static_parameters):
        super().__init__(static_parameters)

    def projector(self):
        return (
        lambda T, dT, lr: jnp.maximum(T + lr*dT, 1e-3),
        lambda t, dt, lr: tuple(ti + lr*dti for (ti, dti) in zip(t, dt))
    )

    def params_to_pulses(self, t, control, dynamic_p, psi):
        Ms = self.static_p["constraints"]["max_amplitude"]
        weights = control[-1]

        def pulses_fn(M, weight):
            n_pieces = jnp.size(weight, 0)
            return M*proj_ball(piecewise_cst_interp(t, weight, n_pieces))

        return jax.tree.map(pulses_fn, Ms, weights)


    def save_to_npz(self, filename, control, dynamic_p):
        initial_state = dynamic_p["initial_state"]
        target_state = dynamic_p["target_state"]
        pulses = self.pulses(control, dynamic_p)
        final_state = self.final_state(control, dynamic_p)
        loss_after_optimizer = self.loss(control, dynamic_p)
        time_points = self.static_p["integrator"]["ts"]
        space_points = self.static_p["space"]["xs"]
        time_horizon = control[0]
        pulse_parameters = control[-1][0] # array

        jnp.savez(filename,
                  psi0=initial_state,
                  psi1=target_state,
                  psi_final=final_state,
                  T=time_horizon,
                  pulses=pulses[0],
                  pulse_p=pulse_parameters,
                  ts=time_points,
                  xs=space_points,
                  loss=loss_after_optimizer
                  )

###############################
########## Neural #############
###############################
class Neural(GrossPitaevskii):
    def __init__(self, static_parameters):
        super().__init__(static_parameters)

    def projector(self):
        return (
            lambda T, dT, lr: jnp.maximum(T + lr*dT, 1e-3),
            lambda t, dt, lr: tuple(jax.tree.map(lambda x, dx: x + lr * dx, ti, dti) for (ti, dti) in zip(t, dt))
        )

    def params_to_pulses(self, t, control, dynamic_p, psi):
        Ms = self.static_p["constraints"]["max_amplitude"]
        weights = control[-1]

        def pulses_fn(M, weight):
            r, phi = network(t, weight) # 0 <= r, phi <= 1
            return M*r*jnp.array([jnp.cos(2*jnp.pi*phi), jnp.sin(2*jnp.pi*phi)]) # polar

        return jax.tree.map(pulses_fn, Ms, weights)

    def save_to_npz(self, filename, control, dynamic_p):
        initial_state = dynamic_p["initial_state"]
        target_state = dynamic_p["target_state"]
        pulses = self.pulses(control, dynamic_p)
        final_state = self.final_state(control, dynamic_p)
        loss_after_optimizer = self.loss(control, dynamic_p)
        time_points = self.static_p["integrator"]["ts"]
        space_points = self.static_p["space"]["xs"]
        time_horizon = control[0]
        neural_parameters = control[-1][0] # dict

        jnp.savez(filename,
                  psi0=initial_state,
                  psi1=target_state,
                  psi_final=final_state,
                  T=time_horizon,
                  pulses=pulses[0],
                  neural_p=neural_parameters,
                  ts=time_points,
                  xs=space_points,
                  loss=loss_after_optimizer
                  )

###############################
########## Mixed #############
###############################
class Mixed(GrossPitaevskii):
    def __init__(self, static_parameters):
        super().__init__(static_parameters)

    def projector(self):
        fns = (lambda z: proj_interval(z, 0.0, 1.0), lambda z: z)
        return (
            lambda T, dT, lr: jnp.maximum(T + lr*dT, 1e-3),
            lambda t, dt, lr: tuple(jax.tree.map(lambda x, dx: fi(x + lr * dx), ti, dti) for (ti, dti, fi) in zip(t, dt, fns))
        )

    def params_to_pulses(self, t, control, dynamic_p, psi):
        Ms = self.static_p["constraints"]["max_amplitude"]
        weights = control[-1]

        def pulses_fn(M, weight):
            # discontinuous radius and smooth angle
            radius_weight, angle_weight = weight
            n_pieces = jnp.size(radius_weight, 0)
            r = proj_interval(piecewise_cst_interp(t, radius_weight, n_pieces), 0.0, 1.0)
            phi = network(t, angle_weight) # 0 <= phi <= 1
            return M*r*jnp.array([jnp.cos(2*jnp.pi*phi), jnp.sin(2*jnp.pi*phi)]) # polar

        return jax.tree.map(pulses_fn, Ms, weights)

    def save_to_npz(self, filename, control, dynamic_p):
        initial_state = dynamic_p["initial_state"]
        target_state = dynamic_p["target_state"]
        pulses = self.pulses(control, dynamic_p)
        final_state = self.final_state(control, dynamic_p)
        loss_after_optimizer = self.loss(control, dynamic_p)
        time_points = self.static_p["integrator"]["ts"]
        space_points = self.static_p["space"]["xs"]
        time_horizon = control[0]
        radius = control[-1][0][0] # array
        phase = control[-1][0][1] # dict

        jnp.savez(filename,
                  psi0=initial_state,
                  psi1=target_state,
                  psi_final=final_state,
                  T=time_horizon,
                  pulses=pulses[0],
                  radius=radius,
                  phase=phase,
                  ts=time_points,
                  xs=space_points,
                  loss=loss_after_optimizer
                  )
