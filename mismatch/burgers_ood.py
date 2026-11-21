"""Out-of-dictionary missing terms for System 3 (parametric Burgers), for addendum v1.8.

The frozen dictionary (`mismatch.burgers.TERMS`) holds local functions of u and x only. The two
terms here contain spatial derivatives, so their responses are not in its span:

  P3  -u_x   transport by a uniform background drift (missing advection)
  P4  u_xx   extra diffusion (the simulator's viscosity is too low)

Derivatives are spectral on the periodic grid. The solver, grid and outputs are those of
`mismatch.burgers`, which is not modified.
"""

import numpy as np

from mismatch import burgers as B


def _dx(u, order):
    n = u.shape[-1]
    k = np.fft.rfftfreq(n, d=1.0 / n)
    return np.fft.irfft((1j * k) ** order * np.fft.rfft(u, axis=-1), n=n, axis=-1)


MISSING_OOD = {"P3": lambda u, x: -_dx(u, 1),
               "P4": lambda u, x: _dx(u, 2)}


def simulate_batch(thetas, system, eps=0.0, chunk=256):
    """Flattened outputs (n, N_OUT) of the nominal (eps = 0) or true system with an OOD term."""
    term = None if eps == 0 else (lambda u, x, f=MISSING_OOD[system]: eps * f(u, x))
    th = np.atleast_2d(thetas)
    parts = [B.solve(th[i:i + chunk], term) for i in range(0, th.shape[0], chunk)]
    return np.concatenate(parts).reshape(th.shape[0], B.N_OUT)


def response(thetas, system, scale=1e-3):
    """Linear response (n, N_OUT) of the nominal output to the OOD term, per unit coefficient."""
    base = B.simulate_batch(thetas)
    return (simulate_batch(thetas, system, scale) - base) / scale


def out_of_span(resp, dict_resp):
    """Per point, the fraction of the response norm not explained by the dictionary's span
    (least squares on that point's dictionary responses). resp (n, T), dict_resp (m, n, T)."""
    out = np.empty(resp.shape[0])
    for i in range(resp.shape[0]):
        Mi = dict_resp[:, i, :].T                                  # (T, m)
        coef, *_ = np.linalg.lstsq(Mi, resp[i], rcond=None)
        out[i] = np.linalg.norm(resp[i] - Mi @ coef) / np.linalg.norm(resp[i])
    return out
