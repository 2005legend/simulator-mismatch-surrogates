"""MLP deep-ensemble surrogate: theta -> trajectory (prereg-v1 section 4).

Inputs are scaled to [-1, 1] with the parameter box; outputs are standardised per time
point using the training set. Members differ only by their seed (random init and batch
order). The same class is used for the nominal surrogate and the discrepancy correction.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mismatch import config


@dataclass
class FitConfig:
    hidden: int = 80          # 3 hidden layers of 80 -> about 30k parameters for 201 outputs
    n_hidden_layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 1500
    patience: int = 150       # early stopping on a held-out split
    val_frac: float = 0.1
    min_val: int = 4          # never hold out fewer than this many points
    dropout: float = 0.0      # > 0 only for the MC-dropout surrogate (prereg-v1 section 6)


def _mlp(n_in, n_out, cfg):
    layers, width = [], n_in
    for _ in range(cfg.n_hidden_layers):
        layers += [nn.Linear(width, cfg.hidden), nn.SiLU()]
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        width = cfg.hidden
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


def scale_inputs(thetas, low=config.BOX_LOW, high=config.BOX_HIGH):
    return 2.0 * (np.asarray(thetas) - low) / (high - low) - 1.0


class Member:
    """One network with its own output standardisation."""

    def __init__(self, n_out, cfg, seed):
        self.cfg, self.seed = cfg, seed
        torch.manual_seed(seed)
        self.net = _mlp(len(config.BOX_LOW), n_out, cfg)
        self.y_mean = self.y_std = None
        self.fit_seconds = None
        self.epochs_run = None
        self.best_val = None            # lowest validation loss reached

    def fit(self, thetas, y):
        cfg, rng = self.cfg, np.random.default_rng(self.seed)
        x = scale_inputs(thetas).astype(np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(x)
        n_val = max(cfg.min_val, int(round(cfg.val_frac * n))) if n >= 2 * cfg.min_val else 0
        perm = rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        self.y_mean = y[tr_idx].mean(axis=0)
        self.y_std = y[tr_idx].std(axis=0) + 1e-6
        yz = (y - self.y_mean) / self.y_std
        xt, yt = torch.from_numpy(x), torch.from_numpy(yz)

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()
        best, best_state, wait = np.inf, None, 0
        start = time.perf_counter()
        for epoch in range(cfg.max_epochs):
            self.net.train()
            order = rng.permutation(tr_idx)
            for i in range(0, len(order), cfg.batch_size):
                b = torch.from_numpy(order[i:i + cfg.batch_size])
                opt.zero_grad()
                loss_fn(self.net(xt[b]), yt[b]).backward()
                opt.step()
            if n_val:
                self.net.eval()
                with torch.no_grad():
                    v = loss_fn(self.net(xt[val_idx]), yt[val_idx]).item()
                if v < best - 1e-7:
                    best, wait = v, 0
                    best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= cfg.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fit_seconds = time.perf_counter() - start
        self.epochs_run = epoch + 1
        self.best_val = float(best)
        return self

    def predict_mc(self, thetas, n_passes=50, seed=0):
        """Stochastic forward passes with dropout active; returns (n_passes, n, T)."""
        torch.manual_seed(seed)
        x = torch.from_numpy(scale_inputs(thetas).astype(np.float32))
        self.net.train()
        with torch.no_grad():
            out = np.stack([self.net(x).numpy() for _ in range(n_passes)])
        self.net.eval()
        return out * self.y_std + self.y_mean

    def predict(self, thetas):
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(scale_inputs(thetas).astype(np.float32))).numpy()
        return out * self.y_std + self.y_mean


class Ensemble:
    """Deep ensemble of `n_members` Members; seeds are base_seed * 100 + member index."""

    def __init__(self, n_out=config.N_T, n_members=5, cfg=None, base_seed=0, batched=True):
        self.cfg = cfg or FitConfig()
        self.batched = batched
        self.members = [Member(n_out, self.cfg, base_seed * 100 + m) for m in range(n_members)]

    def fit(self, thetas, y):
        if self.batched and len(self.members) > 1 and self.cfg.dropout == 0:
            return self._fit_batched(thetas, y)
        for m in self.members:
            m.fit(thetas, y)
        return self

    def _fit_batched(self, thetas, y):
        """Train all members at once as one stacked model.

        Each member keeps its own initialisation, validation split, output standardisation,
        batch order and early stopping, and the loss is the sum of the members' mean squared
        errors, so with Adam (an element-wise optimiser) every member follows the same
        updates as in independent training, up to floating-point summation order. About 4
        times faster for 5 members on one CPU thread (7.1 s against 1.6 s, 100 epochs).
        Member.fit_seconds is the shared wall time of the whole ensemble fit.
        """
        cfg, members = self.cfg, self.members
        n_members = len(members)
        x = torch.from_numpy(scale_inputs(thetas).astype(np.float32))
        y = np.asarray(y, dtype=np.float32)
        n = len(x)
        n_val = max(cfg.min_val, int(round(cfg.val_frac * n))) if n >= 2 * cfg.min_val else 0
        rngs = [np.random.default_rng(mem.seed) for mem in members]
        tr, va, yz = [], [], []
        for mem, rng in zip(members, rngs):
            perm = rng.permutation(n)
            va_idx, tr_idx = perm[:n_val], perm[n_val:]
            mem.y_mean = y[tr_idx].mean(axis=0)
            mem.y_std = y[tr_idx].std(axis=0) + 1e-6
            yz.append((y - mem.y_mean) / mem.y_std)
            tr.append(tr_idx)
            va.append(va_idx)
        yz_all = torch.from_numpy(np.stack(yz))                        # (M, n, T)
        layers = [[l for l in mem.net if isinstance(l, nn.Linear)] for mem in members]
        n_layers = len(layers[0])
        weights = [nn.Parameter(torch.stack([layers[m][k].weight.detach().T
                                             for m in range(n_members)])) for k in range(n_layers)]
        biases = [nn.Parameter(torch.stack([layers[m][k].bias.detach()[None, :]
                                            for m in range(n_members)])) for k in range(n_layers)]

        def forward(xin):                                              # xin: (M, b, n_in)
            h = xin
            for k in range(n_layers):
                h = torch.baddbmm(biases[k], h, weights[k])
                if k < n_layers - 1:
                    h = torch.nn.functional.silu(h)
            return h

        opt = torch.optim.Adam(weights + biases, lr=cfg.lr, weight_decay=cfg.weight_decay)
        rows = torch.arange(n_members)[:, None]
        va_t = torch.from_numpy(np.stack(va)) if n_val else None
        best = np.full(n_members, np.inf)
        wait = np.zeros(n_members, dtype=int)
        active = np.ones(n_members, dtype=bool)
        epochs_run = np.full(n_members, cfg.max_epochs)
        best_state = [None] * n_members
        n_tr = len(tr[0])
        start = time.perf_counter()
        for epoch in range(cfg.max_epochs):
            orders = np.stack([rng.permutation(t) for rng, t in zip(rngs, tr)])
            for i in range(0, n_tr, cfg.batch_size):
                bidx = torch.from_numpy(orders[:, i:i + cfg.batch_size])
                opt.zero_grad()
                loss = ((forward(x[bidx]) - yz_all[rows, bidx]) ** 2).mean(dim=(1, 2)).sum()
                loss.backward()
                opt.step()
            if n_val:
                with torch.no_grad():
                    v = ((forward(x[va_t]) - yz_all[rows, va_t]) ** 2).mean(dim=(1, 2)).numpy()
                for m in range(n_members):
                    if not active[m]:
                        continue
                    if v[m] < best[m] - 1e-7:
                        best[m], wait[m] = v[m], 0
                        best_state[m] = ([w.detach()[m].clone() for w in weights],
                                         [b.detach()[m].clone() for b in biases])
                    else:
                        wait[m] += 1
                        if wait[m] >= cfg.patience:
                            active[m], epochs_run[m] = False, epoch + 1
                if not active.any():
                    break
        total = time.perf_counter() - start
        for m, mem in enumerate(members):
            if best_state[m] is not None:
                ws, bs = best_state[m]
            else:
                ws, bs = [w.detach()[m] for w in weights], [b.detach()[m] for b in biases]
            for lin, w, b in zip(layers[m], ws, bs):
                lin.weight.data.copy_(w.T)
                lin.bias.data.copy_(b[0])
            mem.fit_seconds, mem.epochs_run = total, int(epochs_run[m])
            mem.best_val = float(best[m])
        return self

    def member_predictions(self, thetas):
        return np.stack([m.predict(thetas) for m in self.members])   # (M, n, T)

    def predict(self, thetas):
        """Returns (mean, sigma), each (n, T); sigma is the across-member std."""
        p = self.member_predictions(thetas)
        return p.mean(axis=0), p.std(axis=0)

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.members[0].net.parameters())

    @property
    def fit_seconds(self):
        return [m.fit_seconds for m in self.members]
