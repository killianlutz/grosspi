import jax.numpy as jnp
import jax
from jax.numpy.fft import fft, ifft, fftfreq

def basis(n):
    B = []
    for i in range(n):
        b = jnp.zeros(n, dtype=jnp.complex64)
        b = b.at[i].set(1.0)
        B.append(b)
        B.append(1j*b)

    return jnp.array(B)

def vector_to_coeff(x, basis_vector):
    return jnp.real(jnp.dot(x, jnp.conj(basis_vector)))

def as_coefficients(x, basis):
    return jax.vmap(vector_to_coeff, (None, 0))(x, basis)

def as_vector(v, basis):
    z = jax.vmap(lambda x, y: x*y, 0, 0)(v, basis)
    return jnp.sum(z, axis=0)

def l2_dot(a, b):
    hx = 2 * jnp.pi / jnp.size(a)
    return hx*jnp.dot(a, jnp.conj(b))

def l2_norm(a):
    return jnp.sqrt(jnp.real(l2_dot(a, a)))

def infidelity(a, b):
    scale = l2_norm(a)*l2_norm(b)
    return jnp.abs(1 - jnp.abs(l2_dot(a, b)/scale)**2)

def mvp_laplacian(a):
    hx = 2*jnp.pi/jnp.size(a)
    omega = 2*jnp.pi*fftfreq(jnp.size(a), hx)
    return ifft(-omega**2 * fft(a, norm="ortho"), norm="ortho")

def fft_diff(a):
    hx = 2 * jnp.pi / jnp.size(a)
    omega = 2 * jnp.pi * fftfreq(jnp.size(a), hx)
    return ifft(1j * omega * fft(a, norm="ortho"), norm="ortho")

def gp_energy(a, beta):
    hx = 2 * jnp.pi / jnp.size(a)
    grad_energy = l2_norm(fft_diff(a))**2
    l4_energy = hx*jnp.sum(jnp.abs(a)**4)
    return 0.5*grad_energy + 0.25*beta*l4_energy

def piecewise_cst_interp(t, weights, n_pieces):
    i = jnp.floor(t * n_pieces).astype(jnp.int16)
    return weights[i]

def normalize_if_not_zero(x, atol=1e-7):
    l = jnp.linalg.norm(x)
    return jax.lax.cond(l > atol, lambda y: y / l, lambda y: y, x)

def proj_ball(x, r=1.0):
    l = jnp.linalg.norm(x)
    return jax.lax.cond(l > r, lambda y: r*y/l, lambda y: y, x)

def proj_interval(x, a=0.0, b=1.0):
    return jnp.maximum(a, jnp.minimum(b, x))