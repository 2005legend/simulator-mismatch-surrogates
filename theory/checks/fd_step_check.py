"""Proof to-do 1: finite-difference step check (1e-3 vs 1e-4), dev seed 999, 200 test points.
Compares response norms and their rankings for the pendulum dictionary, the Burgers dictionary
and the OOD terms P3, P4. Run from the repo root: python <this file>."""
import json, sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy.stats import spearmanr
from mismatch import alignment as AL, burgers as B, burgers_ood as BO, config
from mismatch.simulators import sobol_thetas, stream_seed

N = 200
out = {}

def compare(name, r3, r4, base):
    n3 = np.linalg.norm(r3, axis=-1) / np.linalg.norm(base, axis=-1)
    n4 = np.linalg.norm(r4, axis=-1) / np.linalg.norm(base, axis=-1)
    rel = np.linalg.norm(r3 - r4, axis=-1) / np.linalg.norm(r4, axis=-1)
    rho = spearmanr(n3, n4)[0]
    out[name] = {"median_rel_diff": float(np.median(rel)), "max_rel_diff": float(rel.max()), "spearman_norms": float(rho)}
    print(name, out[name], flush=True)

th_p = sobol_thetas(N, seed=stream_seed(999, "test"))
names, r3, base = AL.term_responses(th_p, scale=1e-3)
_, r4, _ = AL.term_responses(th_p, scale=1e-4)
for j, nm in enumerate(names):
    compare(f"pendulum:{nm}", r3[j], r4[j], base)

u = (sobol_thetas(N, seed=stream_seed(999, "test")) - config.BOX_LOW) / (config.BOX_HIGH - config.BOX_LOW)
th_b = B.from_unit(u)
names, r3, base = B.term_responses(th_b, scale=1e-3)
_, r4, _ = B.term_responses(th_b, scale=1e-4)
for j, nm in enumerate(names):
    compare(f"burgers:{nm}", r3[j], r4[j], base)
for t in ("P3", "P4"):
    compare(f"burgers_ood:{t}", BO.response(th_b, t, 1e-3), BO.response(th_b, t, 1e-4), base)

json.dump(out, open(os.path.join("runs", "fd_step_check.json"), "w"), indent=1)
