"""Constants fixed by the pre-registration (tags prereg-v1 to prereg-v1.3).

Change a value here only through a dated entry in section 11 of preregistration.md.
"""

import numpy as np

# Time grid (prereg-v1 section 3)
T_END = 10.0
N_T = 201
T_GRID = np.linspace(0.0, T_END, N_T)

# Physics
G = 9.81

# Solver (prereg-v1 section 3)
RTOL = 1e-8
ATOL = 1e-10

# Parameter box theta = (c, L, phi0, omega0); proposed defaults, the pilot may shrink it
PARAM_NAMES = ("c", "L", "phi0", "omega0")
BOX_LOW = np.array([0.05, 0.5, 0.1, -1.0])
BOX_HIGH = np.array([0.5, 2.0, 2.5, 1.0])

# S2 smoothing width for tanh(phi'/delta), rad/s (pilot may change it by its rule)
DELTA_S2 = 0.05

# Mismatch levels, as multiples of eps_ref (eps_ref set per system in the pilot)
EPS_MULTIPLES = (0.0, 0.25, 0.5, 1.0, 2.0)

# Observation noise on true-system data, as a fraction of the per-trajectory std of phi
NOISE_MAIN = 0.01
NOISE_SECONDARY = 0.05

# Horizon windows for G1 (prereg-v1.2 section 1), seconds
HORIZONS = (2.0, 4.0, 6.0, 8.0, 10.0)

SYSTEMS = ("S1", "S2")
