############################################################
################## CONTROL OF NV CENTERS ###################
############################################################
from scripts._config import *

def free_dynamics(T, u, beta, snapshots=5, title="", filename="0"):
    csys = Piecewise(static_p)
    control = (T, (u,))

    dynamic_p = {"target_state": None, "initial_state": psi0, "beta": beta, "epsilon": None}
    orbit = jnp.abs(csys.trajectory(control, dynamic_p))
    keep = jnp.arange(0, n, n // snapshots)
    for (i, density) in enumerate(orbit[keep]):
        t = ts[keep[i]]
        plt.plot(xs, density, label=f"$t = ${t:.2f}", linewidth=2, alpha=0.25+0.75*t.item())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\psi|^2$")
    plt.title("Generator: "+title)
    plt.legend()
    plt.show()
    plt.savefig("./sims/"+filename, dpi=300, bbox_inches='tight')

T = 1.0
snapshots = 5
u12_zero = jnp.stack((jnp.zeros(1), jnp.zeros(1))).T
u1_nz = jnp.stack((jnp.ones(1), jnp.zeros(1))).T
u2_nz = jnp.stack((jnp.zeros(1), jnp.ones(1))).T
beta_zero = 0.0
beta = 0.5
##############################
###### EXPLORE DYNAMICS ######
##############################
# initial condition unit norm
psi0 = jnp.exp(-0.5*jnp.abs(xs - jnp.pi)**2/0.4**2) + 0*1j
psi0 /= l2_norm(psi0)

# laplacian
free_dynamics(T, u12_zero, beta_zero, snapshots, title=r"$i\partial^2_x$", filename="0")
plt.close()

# laplacian + first control component = 1
free_dynamics(T, u1_nz, beta_zero, snapshots, title=r"$i\partial^2_x + i\frac{s}{2}\cos(x)$", filename="1")
plt.close()

# laplacian + second control component = 1
free_dynamics(T, u2_nz, beta_zero, snapshots, title=r"$i\partial^2_x + i\frac{s}{2}\sin(x)$", filename="2")
plt.close()

# laplacian + non-linearity
free_dynamics(T, u12_zero, beta, snapshots, title=r"$i\partial^2_x -i\beta|\psi|^2$", filename="3")
plt.close()

# laplacian + non-linearity + first control component = 1
free_dynamics(T, u1_nz, beta, snapshots, title=r"$i\partial^2_x + i\frac{s}{2}\cos(x) -i\beta |\psi|^2$", filename="4")
plt.close()

# laplacian + non-linearity + second control component = 1
free_dynamics(T, u2_nz, beta, snapshots, r"$i\partial^2_x + i\frac{s}{2}\sin(x) -i\beta |\psi|^2$", filename="5")
plt.close()