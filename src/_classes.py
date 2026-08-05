import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
from src._tools import *
from diffrax import diffeqsolve, ODETerm, ImplicitEuler, PIDController
import lineax as lx


@jax.tree_util.register_pytree_node_class
class ControlSystem:
    def __init__(self, static_parameters):
        self.static_p = static_parameters

    def final_state(self, control, dynamic_p):
        return self.trajectory(control, dynamic_p)[-1, ...]

    def ode_step(self, control, dynamic_p, x, t):
        scheme = self.static_p["integrator"]["scheme"]
        flows = (self.flow_a, self.flow_b)
        return scheme(self.vector_field, flows, x, t, (control, dynamic_p, self.static_p))

    def trajectory(self, control, dynamic_p):
        ts = self.static_p["integrator"]["ts"]
        x0 = dynamic_p["initial_state"]

        # close over static parameters with e.g. Python functions
        def one_step(carry, _):
            x, i = carry
            y = self.ode_step(control, dynamic_p, x, ts[i])
            return (y, i + 1), y

        carry = (x0, 0) # only dynamic variables (jax traceable) inside carry
        xs = jax.lax.scan(one_step, carry, None, length=len(ts) - 1)[1]

        return jnp.concatenate((x0[None, ...], xs), axis=0) # view of x0

    def loss(self, control, dynamic_p):
        loss_fn = self.static_p["loss_fn"]
        psi_final = self.final_state(control, dynamic_p)
        return loss_fn(psi_final, (control, dynamic_p, self.static_p))

    def natural_gradient(self, control, dynamic_p):
        lstq_p = self.static_p["optimizer"]["least_squares"]
        flat_control, unravel = ravel_pytree(control)

        def _model(u_flat, p):
            control = unravel(u_flat)
            psi_real = as_coefficients(self.final_state(control, p), self.static_p["mat_basis"])
            return psi_real

        # augment system e(T, u) = psi(T; u) with the control to deal with T penalty
        # ---> E(T, u) := (T, u, e(T, u)) for natural gradient purpose
        psi_real = _model(flat_control, dynamic_p)
        _, unravel_augmented = ravel_pytree((flat_control, psi_real))

        def model(u_flat, p):
            psi_real = _model(u_flat, p)
            u_psi = ravel_pytree((u_flat, psi_real))[0]
            return u_psi, u_psi

        def cost(u_psi, p):
            loss_fn = self.static_p["loss_fn"]
            u_flat, psi_real = unravel_augmented(u_psi)
            return loss_fn(
                as_vector(psi_real, self.static_p["mat_basis"]),
                (unravel(u_flat), p, self.static_p)
            )

        eps = lstq_p["regularization"]
        if lstq_p["is_iterative"]:
            partial_fn = lambda x: model(x, dynamic_p)[0]
            u_psi, f_jvp = jax.linearize(partial_fn, flat_control)
            _, f_vjp = jax.vjp(partial_fn, flat_control)
            current_loss, grad_cost = jax.value_and_grad(cost)(u_psi, dynamic_p)
            def gram_mvp(x):
                # Tikhonov least-squares operator (normal equations)
                # Gram matrix calculated by combined vjp - jvp.
                return f_vjp(f_jvp(x))[0] + eps*x

            A = lx.FunctionLinearOperator(gram_mvp, flat_control, tags=lstq_p["tags"])
            b = f_vjp(-grad_cost)[0]
            solver = lstq_p["iterative_solver"]
        else:
            linearized_model, u_psi = jax.jacobian(model, has_aux=True)(flat_control, dynamic_p)
            current_loss, grad_cost = jax.value_and_grad(cost)(u_psi, dynamic_p)

            # A = lx.MatrixLinearOperator(
            #     linearized_model.T @ linearized_model + eps*jnp.eye(jnp.size(flat_control)),
            #     tags=lstq_p["tags"]
            # )
            # b = -(linearized_model.T @ grad_cost)
            A = lx.MatrixLinearOperator(linearized_model)
            b = -grad_cost
            solver = lstq_p["direct_solver"]

        flat_step = lx.linear_solve(A, b, solver).value

        normalize_gradient = self.static_p["optimizer"]["normalize_gradient"]
        _flat_step = jax.lax.cond(
            normalize_gradient,
            normalize_if_not_zero,
            lambda y: y,
            flat_step
        )

        return unravel(_flat_step), current_loss

    def apply_update(self, control, direction, learning_rate):
        return tuple(
            proj(p, d, learning_rate) for (p, d, proj) in zip(control, direction, self.projector())
        )

    def line_search(self, control, dynamic_p, direction, reference_loss):
        p = (control, dynamic_p, direction)
        search_parameters = self.static_p["optimizer"]["line_search"]
        search_fn = search_parameters["search_fn"]

        def loss_along_line(e, p):
            lr = jnp.pow(10.0, e)
            control, dynamic_p, direction = p
            new_control = self.apply_update(control, direction, lr)
            return self.loss(new_control, dynamic_p)

        e, next_loss = search_fn(loss_along_line, p, (search_parameters, reference_loss))
        return jnp.pow(10.0, e), next_loss

    def optimizer_step(self, control, dynamic_p):
        direction, current_loss = self.natural_gradient(control, dynamic_p)
        learning_rate, next_loss = self.line_search(control, dynamic_p, direction, current_loss)
        control = self.apply_update(control, direction, learning_rate)

        return control, next_loss

    def solve_ocp(self, init_control, dynamic_p):
        optimizer_p = self.static_p["optimizer"]
        abstol_loss = optimizer_p["abstol_loss"]
        reltol_dist = optimizer_p["reltol_dist"]
        n_max = optimizer_p["n_max"]
        init_metrics = self.eval_metrics(init_control, dynamic_p)
        n_metrics = jnp.size(init_metrics)

        # initial guess
        init_loss = self.loss(init_control, dynamic_p)
        losses = jnp.zeros(n_max).at[0].set(init_loss)
        metrics = jnp.zeros((n_max, n_metrics)).at[0].set(init_metrics)

        # two steps recurrence
        control, current_loss = self.optimizer_step(init_control, dynamic_p)
        losses = losses.at[1].set(current_loss)
        metrics = metrics.at[1].set(self.eval_metrics(control, dynamic_p))
        init = (control, init_control, losses, metrics, 1, dynamic_p)

        def keep_going_criteria(val):
            current_u, old_u, losses, _, i, _ = val
            current_loss = losses[i]
            x, _ = ravel_pytree(current_u)
            y, _ = ravel_pytree(old_u)
            # relative distance between consecutive iterates
            # iteration counter
            # abs tolerance on objective value
            return jnp.logical_and(
                i < n_max - 1,
                jnp.logical_and(
                    jnp.linalg.norm(x - y) > reltol_dist*jnp.linalg.norm(y),
                    current_loss > abstol_loss
                )
            )

        def one_optimizer_step(val):
            current_u, _, losses, metrics, i, dyn_p = val
            new_u, new_loss = self.optimizer_step(current_u, dyn_p)
            return (
                new_u,
                current_u,
                losses.at[i + 1].set(new_loss),
                metrics.at[i + 1].set(self.eval_metrics(new_u, dynamic_p)),
                i + 1,
                dyn_p
            )

        val = jax.lax.while_loop(keep_going_criteria, one_optimizer_step, init)
        optimized_control = val[0]
        losses, metrics, n_iter = val[-4:-1]
        return optimized_control, losses, metrics, n_iter

    def solve_ocp_batch(self, init_control_batch, dynamic_p):
        optimizer_p = self.static_p["optimizer"]
        abstol_loss = optimizer_p["abstol_loss"]
        reltol_dist = optimizer_p["reltol_dist"]

        # batch over initial controls
        loss_batch = jax.vmap(self.loss, in_axes=(0, None))
        metrics_batch = jax.vmap(self.eval_metrics, in_axes=(0, None))
        optimizer_step_batch = jax.vmap(self.optimizer_step, in_axes=(0, None))

        # initial guess
        init_loss = loss_batch(init_control_batch, dynamic_p)
        init_metrics = metrics_batch(init_control_batch, dynamic_p)
        n_max = optimizer_p["n_max"]
        n_metrics = jnp.size(init_metrics, 1)
        n_batch = jnp.size(jax.tree.leaves(init_control_batch)[0], 0)

        losses = jnp.zeros((n_max, n_batch)).at[0].set(init_loss)
        metrics = jnp.zeros((n_max, n_batch, n_metrics)).at[0].set(init_metrics)

        # two steps recurrence
        control, current_loss = optimizer_step_batch(init_control_batch, dynamic_p)
        losses = losses.at[1].set(current_loss)
        metrics = metrics.at[1].set(metrics_batch(init_control_batch, dynamic_p))
        n_iter = jnp.ones((n_batch,), dtype=jnp.int32) # per trajectory
        active = jnp.ones((n_batch,), dtype=bool) # mask indicating convergence
        init = (control, init_control_batch, active, losses, metrics, 1, n_iter, dynamic_p)

        def pytree_batch_norm(tree):
            # returns shape = batch dimension
            sq_tree = jax.tree.map(lambda x: jnp.sum(x * x, axis=tuple(range(1, x.ndim))), tree)
            sq = jax.tree_util.tree_reduce(lambda a, b: a + b, sq_tree)
            return jnp.sqrt(sq)

        def tree_where(mask, new_tree, old_tree):
            # broadcast shape of mask to match the shape of any leaf (broadcasting aligns dimensions from the right)
            # (batch dimension is the leading one for any leaf)
            return jax.tree.map(
                lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (n.ndim - 1)),
                    n,
                    o,
                ),
                new_tree,
                old_tree,
            )

        def keep_going_criteria(val):
            current_u, old_u, active, losses, _, i, _, _ = val
            current_loss = losses[i]
            diff = jax.tree.map(lambda a, b: a - b, current_u, old_u)
            dist = pytree_batch_norm(diff)
            ref = jnp.maximum(pytree_batch_norm(old_u), 1e-12)
            still_active = jnp.logical_and(dist > reltol_dist * ref, current_loss > abstol_loss)
            return jnp.logical_and(i < n_max - 1, jnp.any(jnp.logical_and(active, still_active)))

        def one_optimizer_step(val):
            current_u, old_u, active, losses, metrics, i, n_iter, dyn_p = val
            # candidate update for all batch elements
            new_u_all, new_loss_all = optimizer_step_batch(current_u, dyn_p)
            # convergence test based on current iterate
            diff = jax.tree.map(lambda a, b: a - b, current_u, old_u)
            dist = pytree_batch_norm(diff)
            ref = jnp.maximum(pytree_batch_norm(old_u), 1e-12)
            still_active = jnp.logical_and(dist > reltol_dist * ref, losses[i] > abstol_loss)
            new_active = jnp.logical_and(active, still_active)

            # update only active trajectories
            new_u = tree_where(new_active, new_u_all, current_u)
            # keep previous loss for converged trajectories
            new_loss = jnp.where(new_active, new_loss_all, losses[i])
            # keep previous metrics for converged trajectories
            # one boolean per batch element, applied to all metrics of that element
            new_metrics_all = metrics_batch(new_u, dyn_p)
            new_metrics = jnp.where(new_active[:, None], new_metrics_all, metrics[i])
            # add one iter to trajectories that were updated
            n_iter = n_iter + new_active.astype(jnp.int32)

            return (
                new_u,
                current_u,
                new_active,
                losses.at[i + 1].set(new_loss),
                metrics.at[i + 1].set(new_metrics),
                i + 1,
                n_iter,
                dyn_p
            )

        val = jax.lax.while_loop(keep_going_criteria, one_optimizer_step, init)
        optimized_control = val[0]
        losses = val[3]
        metrics = val[4]
        n_iter = val[6]

        # mark the history before convergence, put NaN's otherwise
        valid = jnp.arange(n_max)[:, None] <= n_iter[None, :]
        losses = jnp.where(valid, losses, jnp.nan)
        metrics = jnp.where(valid[:, :, None], metrics, jnp.nan)
        return optimized_control, losses, metrics, n_iter

    def validate(self, control, dynamic_p, dt0=1e-4, solver=ImplicitEuler(), stepsize_controller=PIDController(atol=1e-5, rtol=1e-4), **kwargs):
        psi0 = dynamic_p["initial_state"]
        args = (control, dynamic_p, self.static_p)

        psi_final = diffeqsolve(
            ODETerm(lambda t, U, args: self.vector_field(U, t, args)),
            t0=0.0, t1=1.0, dt0=dt0,
            y0=psi0,
            args=args,
            solver=solver,
            stepsize_controller=stepsize_controller,
            **kwargs
        ).ys[0, ...]

        loss_fn = self.static_p["loss_fn"]
        psi1 = dynamic_p["target_state"]
        return loss_fn(psi_final, args)

    def save_to_npz(self, filename, control, dynamic_p):
        initial_state = dynamic_p["initial_state"]
        target_state = dynamic_p["target_state"]
        gate_time = control[0]
        pulses = self.pulses(control, dynamic_p)
        final_state = self.final_state(control, dynamic_p)
        loss_after_optimizer = self.loss(control, dynamic_p)
        time_points = self.static_p["integrator"]["ts"]
        space_points = self.static_p["space"]["xs"]

        jnp.savez(filename, initial_state, target_state, gate_time, *pulses, time_points, space_points, final_state, loss_after_optimizer)

    def pulse_fns(self, control, dynamic_p):
        control_pulses = self.pulses(control, dynamic_p)
        def zoh(weights):
            n_pieces = jnp.size(weights, 0)
            return lambda t: piecewise_cst_interp(t, weights, n_pieces)

        return jax.tree.map(zoh, control_pulses)

    def plot_pulses(self, control, dynamic_p, **kwargs):
        Ms = self.static_p["constraints"]["max_amplitude"]
        ts = self.static_p["integrator"]["ts"]
        pulses = self.pulses(control, dynamic_p) # parameters -> pulses
        fig, axes = plt.subplots(1, len(pulses), **kwargs)
        if len(pulses) == 1:
            axes = (axes, )

        # PULSES
        for (i, (pulse, M)) in enumerate(zip(pulses, Ms)):
            energy = jnp.linalg.norm(pulse, axis=1)
            ax = axes[i]
            ax.plot(ts, pulse, linewidth=3)
            ax.plot(ts, energy, linestyle='--', color='purple', label="$norm$", linewidth=3)
            ax.plot(ts, M * jnp.ones_like(ts), color='k', linewidth=1.5)
            ax.plot(ts, -M * jnp.ones_like(ts), color='k', linewidth=1.5)
            ax.plot(ts, 0 * ts, color='k', linewidth=0.75)
            ax.set_xlabel(r"Time $t/T$")
            ax.set_title(f"Time horizon: $T =${control[0]:.3f}")
            ax.grid(True)
            ax.legend(loc='upper right')

        plt.tight_layout()
        fig.show()


    def plot_optimization(self, control, dynamic_p, losses, metrics, **kwargs):
        fig, axes = plt.subplots(1, 1, **kwargs)

        # LOSS
        axes.semilogy(losses, '-x', color="k", label=r"$J$")
        axes.semilogy(metrics, linewidth=2)
        axes.set_title("Optimization results")
        axes.set_xlabel("Iteration")
        axes.grid(True)
        axes.legend(loc='upper left')

        plt.tight_layout()
        fig.show()

    def plot_densities(self, control, dynamic_p, **kwargs):
        hx = self.static_p["space"]["hx"]
        psi0 = jnp.abs(dynamic_p["initial_state"])
        psi1 = jnp.abs(dynamic_p["target_state"])
        psi_final = jnp.abs(self.final_state(control, dynamic_p))

        fig, axes = plt.subplots(1, 1, **kwargs)
        plt.plot(psi0, color='r', linestyle='-', label=r"$\psi_0$")
        plt.plot(psi1, color='b', linestyle='--', label=r"$\psi_1$")
        plt.plot(psi_final, color='k', linestyle='-', label=r"$\psi(T)$")
        axes.set_title("densities")
        axes.set_xlabel("x")
        axes.set_ylabel(r"$|\psi|^2$")
        axes.grid(True)
        axes.legend(loc='upper left')

        plt.tight_layout()
        fig.show()

    def flow_a(self, psi, t0, t1, p):
        pass

    def flow_b(self, psi, t0, t1, p):
        pass

    def vector_field(self, psi, t, p):
        pass

    def projector(self):
        pass

    def params_to_pulses(self, t, control, dynamic_p, state):
        pass

    def eval_metrics(self, control, dynamic_p):
        pass

    def pulses(self, control, dynamic_p):
        ts = self.static_p["integrator"]["ts"]
        psi_orbit = self.trajectory(control, dynamic_p)
        def pulse(i):
            return jax.vmap(
                lambda t, psi: self.params_to_pulses(t, control, dynamic_p, psi)[i],
                in_axes=(0, 0)
            )(ts, psi_orbit)
        n_independent_controls = len(control[-1])
        indices = tuple(jnp.arange(n_independent_controls, dtype=jnp.int16))
        return jax.tree.map(pulse, indices)

    def tree_flatten(self):
        return (), self.static_p

    # reconstruct from children + aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, _):
        return cls(aux_data)
