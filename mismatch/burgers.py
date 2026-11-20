"""System 3: 1D viscous Burgers with a missing term (PDE track; design to be frozen in addendum
v1.7 before any registered run).

Nominal:  u_t + u u_x = nu u_xx, periodic on [0, 2 pi).
True:     the same plus a missing term eps * h, with
  P1  h = -u|u|        state-dependent nonlinear damping
  P2  h = sin(3x)      fixed forcing, independent of the state
theta = (nu, A1, A2, phase), u0 = A1 sin(x) + A2 sin(2x + phase).

Solver: pseudo-spectral on N_X points, 2/3 dealiasing, integrating-factor RK4 (diffusion exact),
batched over theta. Output: u on N_OX equispaced points at N_OT times in [0, T_END], flattened
time-major to a vector of length N_OUT, so the surrogate predicts a space-time field and the PDE
residual of its prediction can be computed (signal 3, as for the pendulum).

The surrogate (`mismatch.surrogate.Ensemble`) scales inputs with the pendulum box; `to_unit_box`
maps this system's box onto it affinely, so the network sees inputs in [-1, 1] exactly as there.
"""

import numpy as np

from mismatch import config

N_X = 256
N_OX, N_OT, T_END = 32, 21, 2.0
N_OUT = N_OX * N_OT
DT = 0.002
BOX_LOW = np.array([0.10, 0.5, 0.0, 0.0])
BOX_HIGH = np.array([0.30, 1.5, 1.0, np.pi])
NOISE = 0.01

X_OUT = np.linspace(0.0, 2 * np.pi, N_OX, endpoint=False)
T_OUT = np.linspace(0.0, T_END, N_OT)

MISSING = {"P1": lambda u, x: -u * np.abs(u),
           "P2": lambda u, x: np.sin(3 * x) + 0 * u}


def grid(n_x=N_X):
    x = np.linspace(0.0, 2 * np.pi, n_x, endpoint=False)
    k = np.fft.rfftfreq(n_x, d=1.0 / n_x)
    keep = (k <= n_x // 3).astype(float)                    # 2/3 rule
    return x, k, keep


def initial(thetas, x):
    th = np.atleast_2d(thetas)
    return th[:, 1:2] * np.sin(x) + th[:, 2:3] * np.sin(2 * x + th[:, 3:4])


def integrate(u0, nu, term=None, n_x=N_X, dt=DT, n_ox=N_OX):
    """Batched integration from initial profiles u0 (n, n_x) with viscosities nu (n,).
    term(u, x) -> array like u, added to u_t (None = nominal). Returns (n, N_OT, n_ox)."""
    x, k, keep = grid(n_x)
    nu = np.asarray(nu, float).reshape(-1, 1)
    E = np.exp(-nu * k ** 2 * dt / 2)
    E2 = E * E

    def nonlin(v):
        u = np.fft.irfft(v, n=n_x)
        out = -0.5j * k * np.fft.rfft(u * u)
        if term is not None:
            out = out + np.fft.rfft(term(u, x))
        return out * keep

    v = np.fft.rfft(np.atleast_2d(u0))
    steps_per_out = int(round((T_END / (N_OT - 1)) / dt))
    stride = n_x // n_ox
    out = np.empty((v.shape[0], N_OT, n_ox))
    out[:, 0] = np.fft.irfft(v, n=n_x)[:, ::stride]
    for j in range(1, N_OT):
        for _ in range(steps_per_out):
            a = dt * nonlin(v)
            b = dt * nonlin(E * (v + a / 2))
            c = dt * nonlin(E * v + b / 2)
            d = dt * nonlin(E2 * v + E * c)
            v = E2 * v + (E2 * a + 2 * E * (b + c) + d) / 6
        out[:, j] = np.fft.irfft(v, n=n_x)[:, ::stride]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Burgers solve produced non-finite values")
    return out


def solve(thetas, term=None, n_x=N_X, dt=DT):
    """Batched solve of the parametric family. Returns (n, N_OT, N_OX)."""
    th = np.atleast_2d(np.asarray(thetas, float))
    x, _, _ = grid(n_x)
    return integrate(initial(th, x), th[:, 0], term, n_x, dt)


def simulate_batch(thetas, system="P1", eps=0.0, chunk=256):
    """Flattened outputs (n, N_OUT) of the nominal (eps = 0) or true system."""
    term = None if eps == 0 else (lambda u, x, f=MISSING[system]: eps * f(u, x))
    th = np.atleast_2d(thetas)
    parts = [solve(th[i:i + chunk], term) for i in range(0, th.shape[0], chunk)]
    return np.concatenate(parts).reshape(th.shape[0], N_OUT)


def to_unit_box(thetas):
    """Affine map of this box onto the pendulum box used by surrogate.scale_inputs."""
    u = (np.asarray(thetas, float) - BOX_LOW) / (BOX_HIGH - BOX_LOW)
    return config.BOX_LOW + u * (config.BOX_HIGH - config.BOX_LOW)


def from_unit(u):
    return BOX_LOW + np.asarray(u) * (BOX_HIGH - BOX_LOW)


def residual_field(u, nu):
    """RMS over x of u_t + u u_x - nu u_xx for fields u (n, n_t, n_x) on the output times T_OUT
    (spectral x-derivatives, central differences in t); shape (n, n_t - 2)."""
    n_x = u.shape[2]
    k = np.fft.rfftfreq(n_x, d=1.0 / n_x)
    uh = np.fft.rfft(u, axis=2)
    ux = np.fft.irfft(1j * k * uh, n=n_x, axis=2)
    uxx = np.fft.irfft(-(k ** 2) * uh, n=n_x, axis=2)
    ut = (u[:, 2:] - u[:, :-2]) / (2 * (T_OUT[1] - T_OUT[0]))
    r = ut + (u * ux - np.asarray(nu, float).reshape(-1, 1, 1) * uxx)[:, 1:-1]
    return np.sqrt(np.mean(r ** 2, axis=2))


def residual_profile(pred, thetas):
    """Residual of a predicted System 3 field (flattened, N_OUT values) per time point."""
    th = np.atleast_2d(thetas)
    return residual_field(np.asarray(pred).reshape(-1, N_OT, N_OX), th[:, 0])


def residual_score(pred, thetas):
    """Signal 3: time-averaged RMS PDE residual of the prediction."""
    return residual_profile(pred, thetas).mean(axis=1)


# Dictionary of hypothesised missing terms h(u, x) for the alignment range (v1.6 section 5
# carried to System 3; final list to be frozen in v1.7). True families: P1 {u|u|, u^3},
# P2 {sin(3x), cos(3x)}.
TERMS = {
    "u": lambda u, x: u,
    "u|u|": lambda u, x: u * np.abs(u),
    "u^3": lambda u, x: u ** 3,
    "u^2": lambda u, x: u ** 2,
    "sin(x)": lambda u, x: np.sin(x) + 0 * u,
    "sin(2x)": lambda u, x: np.sin(2 * x) + 0 * u,
    "sin(3x)": lambda u, x: np.sin(3 * x) + 0 * u,
    "cos(3x)": lambda u, x: np.cos(3 * x) + 0 * u,
    "u sin(x)": lambda u, x: u * np.sin(x),
}
TRUE_FAMILY = {"P1": ("u|u|", "u^3"), "P2": ("sin(3x)", "cos(3x)")}


def term_responses(thetas, terms=None, scale=1e-3):
    """Linear response of the nominal field to each term (finite perturbation), as in
    mismatch.alignment.term_responses. Returns (names, responses (m, n, N_OUT), base)."""
    terms = terms or TERMS
    base = simulate_batch(thetas)
    out = []
    for fn in terms.values():
        th = np.atleast_2d(thetas)
        moved = np.concatenate([solve(th[i:i + 256], lambda u, x, f=fn: scale * f(u, x))
                                for i in range(0, th.shape[0], 256)]).reshape(th.shape[0], N_OUT)
        out.append((moved - base) / scale)
    return list(terms), np.stack(out), base
