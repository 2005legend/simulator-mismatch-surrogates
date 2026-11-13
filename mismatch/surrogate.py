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


def _mlp(n_in, n_out, cfg):
    layers, width = [], n_in
    for _ in range(cfg.n_hidden_layers):
        layers += [nn.Linear(width, cfg.hidden), nn.SiLU()]
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
        return self

    def predict(self, thetas):
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(scale_inputs(thetas).astype(np.float32))).numpy()
        return out * self.y_std + self.y_mean


class Ensemble:
    """Deep ensemble of `n_members` Members; seeds are base_seed * 100 + member index."""

    def __init__(self, n_out=config.N_T, n_members=5, cfg=None, base_seed=0):
        self.cfg = cfg or FitConfig()
        self.members = [Member(n_out, self.cfg, base_seed * 100 + m) for m in range(n_members)]

    def fit(self, thetas, y):
        for m in self.members:
            m.fit(thetas, y)
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
