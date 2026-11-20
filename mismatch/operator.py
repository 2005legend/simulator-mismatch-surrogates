"""System 4: a neural operator on Burgers (function to function). Gate G3 decides whether it
enters the paper (at most 5 minutes per member fit on CPU and at most 5% median relative error).

Input: an initial profile u0(x), a random Fourier series with K_MODES modes and decaying
coefficients, scaled to max |u0| = AMP. Output: u(x, t) at the N_OT - 1 times after 0 on N_OP
points (same solver as System 3, fixed NU). The missing terms are those of System 3 (P1, P2).

Model: a 1D Fourier neural operator (lift, FNO_LAYERS spectral-conv layers with FNO_MODES modes,
projection to the time channels). An ensemble of M members differs by initialisation and batch
order, as the MLP ensembles do. Standardisation: one global mean and scale for inputs and outputs.
"""

import time

import numpy as np
import torch
from torch import nn

from mismatch import burgers as B

N_OP = 64
K_MODES = 6
AMP = 1.5
NU = 0.15
FNO_WIDTH, FNO_MODES, FNO_LAYERS = 32, 16, 4


def sample_coefs(n, seed):
    """Fourier coefficients (n, 2 K_MODES): a_k, b_k ~ N(0, k^-2)."""
    rng = np.random.default_rng(seed)
    k = np.arange(1, K_MODES + 1)
    return rng.normal(size=(n, 2 * K_MODES)) / np.concatenate([k, k])


def profiles(coefs, x):
    c = np.atleast_2d(coefs)
    k = np.arange(1, K_MODES + 1)
    u = c[:, :K_MODES] @ np.cos(np.outer(k, x)) + c[:, K_MODES:] @ np.sin(np.outer(k, x))
    return AMP * u / np.abs(u).max(axis=1, keepdims=True)


def simulate(coefs, system="P1", eps=0.0):
    """(n, N_OT - 1, N_OP) fields after t = 0, and the inputs u0 on the N_OP grid."""
    x_fine, _, _ = B.grid()
    term = None if eps == 0 else (lambda u, x, f=B.MISSING[system]: eps * f(u, x))
    c = np.atleast_2d(coefs)
    out = np.concatenate([B.integrate(profiles(c[i:i + 256], x_fine), np.full(len(c[i:i + 256]), NU),
                                      term, n_ox=N_OP) for i in range(0, len(c), 256)])
    return out[:, 1:], out[:, 0]


class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        self.w = nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, x):                                  # x: (b, c, n)
        xf = torch.fft.rfft(x)
        out = torch.zeros(x.shape[0], x.shape[1], xf.shape[-1], dtype=torch.cfloat)
        out[:, :, :self.modes] = torch.einsum("bin,ion->bon", xf[:, :, :self.modes], self.w)
        return torch.fft.irfft(out, n=x.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, n_out, width=FNO_WIDTH, modes=FNO_MODES, layers=FNO_LAYERS):
        super().__init__()
        self.lift = nn.Conv1d(2, width, 1)
        self.spec = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv1d(width, 64, 1), nn.GELU(), nn.Conv1d(64, n_out, 1))

    def forward(self, u0):                                 # u0: (b, n)
        n = u0.shape[-1]
        grid = torch.linspace(0, 1, n + 1)[:-1].expand(u0.shape[0], n)
        h = self.lift(torch.stack([u0, grid], dim=1))
        for i, (s, p) in enumerate(zip(self.spec, self.pw)):
            h = s(h) + p(h)
            if i < len(self.spec) - 1:
                h = torch.nn.functional.gelu(h)
        return self.proj(h)                                # (b, n_out, n)


class OperatorEnsemble:
    """M independently initialised FNOs; predict returns mean and std over members."""

    def __init__(self, n_members=5, base_seed=0, epochs=400, lr=1e-3, batch=32, patience=40,
                 val_frac=0.1):
        self.M, self.base_seed = n_members, base_seed
        self.epochs, self.lr, self.batch, self.patience, self.val_frac = epochs, lr, batch, patience, val_frac
        self.members, self.fit_seconds = [], []

    def fit(self, u0, y):
        u0, y = np.asarray(u0, np.float32), np.asarray(y, np.float32)
        self.mu_x, self.sd_x = float(u0.mean()), float(u0.std())
        self.mu_y, self.sd_y = float(y.mean()), float(y.std())
        X = torch.from_numpy((u0 - self.mu_x) / self.sd_x)
        Y = torch.from_numpy((y - self.mu_y) / self.sd_y)
        for m in range(self.M):
            t0 = time.perf_counter()
            g = torch.Generator().manual_seed(self.base_seed * 100 + m)
            torch.manual_seed(self.base_seed * 100 + m)
            perm = torch.randperm(len(X), generator=g)
            n_val = max(1, int(self.val_frac * len(X)))
            vi, ti = perm[:n_val], perm[n_val:]
            net = FNO1d(Y.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            best, best_state, wait = np.inf, None, 0
            for _ in range(self.epochs):
                net.train()
                order = ti[torch.randperm(len(ti), generator=g)]
                for i in range(0, len(order), self.batch):
                    b = order[i:i + self.batch]
                    opt.zero_grad()
                    loss = torch.mean((net(X[b]) - Y[b]) ** 2)
                    loss.backward()
                    opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    v = float(torch.mean((net(X[vi]) - Y[vi]) ** 2))
                if v < best:
                    best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            net.load_state_dict(best_state)
            self.members.append(net)
            self.fit_seconds.append(time.perf_counter() - t0)
        return self

    def predict(self, u0):
        X = torch.from_numpy((np.asarray(u0, np.float32) - self.mu_x) / self.sd_x)
        with torch.no_grad():
            P = np.stack([m(X).numpy() for m in self.members]) * self.sd_y + self.mu_y
        return P.mean(axis=0), P.std(axis=0)
