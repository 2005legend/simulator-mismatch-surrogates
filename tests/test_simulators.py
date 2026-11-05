import numpy as np
import pytest

from mismatch import config
from mismatch.simulators import add_noise, simulate, simulate_batch, sobol_thetas

THETA = np.array([0.2, 1.0, 1.0, 0.5])


@pytest.mark.parametrize("system", config.SYSTEMS)
def test_eps_zero_equals_nominal(system):
    nominal = simulate(THETA, "S1", eps=0.0)
    np.testing.assert_array_equal(simulate(THETA, system, eps=0.0), nominal)


def test_shape_and_initial_condition():
    phi = simulate(THETA)
    assert phi.shape == (config.N_T,)
    assert phi[0] == pytest.approx(THETA[2])


def test_small_angle_matches_linear_solution():
    # phi'' + c phi' + (g/L) phi = 0 for small phi, underdamped closed form
    c, L, phi0 = 0.1, 1.0, 1e-3
    phi = simulate([c, L, phi0, 0.0])
    w0 = np.sqrt(config.G / L)
    wd = np.sqrt(w0**2 - (c / 2) ** 2)
    t = config.T_GRID
    exact = phi0 * np.exp(-c * t / 2) * (np.cos(wd * t) + (c / (2 * wd)) * np.sin(wd * t))
    np.testing.assert_allclose(phi, exact, atol=1e-8)


def _energy(phi, omega, L):
    return 0.5 * omega**2 + (config.G / L) * (1 - np.cos(phi))


@pytest.mark.parametrize("system", config.SYSTEMS)
def test_missing_term_dissipates_extra_energy(system):
    phi_n, om_n = simulate(THETA, system, eps=0.0, return_omega=True)
    phi_t, om_t = simulate(THETA, system, eps=0.1, return_omega=True)
    L = THETA[1]
    assert _energy(phi_t[-1], om_t[-1], L) < _energy(phi_n[-1], om_n[-1], L)


@pytest.mark.parametrize("system", config.SYSTEMS)
def test_discrepancy_grows_with_eps(system):
    nominal = simulate(THETA, system, eps=0.0)
    gaps = [np.linalg.norm(simulate(THETA, system, eps=e) - nominal) for e in (0.05, 0.1, 0.2)]
    assert gaps[0] < gaps[1] < gaps[2]


def test_batch_matches_single_and_parallel():
    thetas = sobol_thetas(8, seed=0)
    serial = simulate_batch(thetas, "S2", eps=0.1)
    parallel = simulate_batch(thetas, "S2", eps=0.1, n_jobs=2)
    np.testing.assert_array_equal(serial, parallel)
    np.testing.assert_array_equal(serial[3], simulate(thetas[3], "S2", eps=0.1))


def test_sobol_in_box_and_continuation():
    first = sobol_thetas(16, seed=3)
    assert np.all(first >= config.BOX_LOW) and np.all(first <= config.BOX_HIGH)
    both = sobol_thetas(32, seed=3)
    np.testing.assert_allclose(both[:16], first)
    np.testing.assert_allclose(sobol_thetas(16, seed=3, skip=16), both[16:])


def test_noise_level():
    rng = np.random.default_rng(0)
    clean = simulate_batch(sobol_thetas(4, seed=1))
    noisy = add_noise(clean, 0.01, rng)
    ratio = (noisy - clean).std(axis=1) / clean.std(axis=1)
    np.testing.assert_allclose(ratio, 0.01, rtol=0.15)


def test_unknown_system_raises():
    with pytest.raises(ValueError):
        simulate(THETA, "S3", eps=0.1)
