import lineax as lx
from src._line_search import golden_section
from src._networks import *
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 100)

from scripts._user_fns import *

##############################
##### NUMERICAL SCHEME #######
##############################
n = 2**8 # time
nx = 2**8 # space
ts = jnp.linspace(0.0, 1.0, n)
xs = jnp.linspace(0.0, 2*jnp.pi, nx + 1)[:-1] # periodicity + DFT
h = ts[1] - ts[0]
hx = xs[1] - xs[0]

##############################
##### CONTROL SYSTEM #########
##############################
real_basis = basis(nx)
mu = jnp.stack((jnp.cos(xs), jnp.sin(xs)))
s = 5e0 # maximal control amplitude

###### DYNAMIC ARGUMENTS
eps = 1e-1
beta = 0.5
psi0 = jnp.exp(-5*jnp.abs(xs - jnp.pi)**2) + 0*1j # jnp.cos(xs) + 0*1j
psi1 = jnp.cos(2*xs) + 0*1j # same 2-norm as psi0
psi0, psi1 = jax.tree.map(lambda x: x/l2_norm(x), (psi0, psi1))
# dynamic_p = {"target_state": psi1, "initial_state": psi0, "beta": 0.0, "epsilon": eps}

###### STATIC ARGUMENTS
static_p = {
    "loss_fn": loss_fn,
    "mat_basis": real_basis,
    "su_basis": real_basis,
    "constraints": {
        "max_amplitude": (s, )
    },
    "system": {
        "ctrl": (mu, )
    },
    "integrator": {
        "h": h,
        "ts": ts,
        "scheme": strang_split
    },
    "space": {
            "hx": hx,
            "xs": xs
    },
    "optimizer": {
        "normalize_gradient": True,
        "n_max": 100,
        "abstol_loss": 1e-4,
        "reltol_dist": 1e-2,
        "line_search": {
            "search_fn": golden_section, # signature (f, dynamic, static) -> step, val
            "log_interval": (-5.0, 0.0),
            "abstol": 1e-2,
            "n_max": 200
        },
        "least_squares": {
            "regularization": 1e-3,
            "is_iterative": False,
            "tags": (lx.positive_semidefinite_tag,),
            "iterative_solver": lx.CG(atol=1e-8, rtol=1e-3, max_steps=500),
            "direct_solver":  lx.AutoLinearSolver(well_posed=False)
        }
    }
}