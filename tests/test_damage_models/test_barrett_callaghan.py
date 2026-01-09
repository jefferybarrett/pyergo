# tests/test_damage_models/test_barrett_callaghan.py

import numpy as np
import pytest

from scipy.special import exp1
from scipy.optimize import newton

from pyergo.damage_models.barrett_callaghan import BarrettCallaghan
from pyergo.utils.units import Units


# -------------------------------------------------
# Global test hygiene: Units is mutable global state
# -------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_units():
    Units.reset_SI()
    yield
    Units.reset_SI()


# -------------------------------------------------
# Reference (legacy) implementation
# -------------------------------------------------


class OldBarrettCallaghan:
    def __init__(self, A=3.90362810e-07, B=1.42637606e-03):
        self.A, self.B = A, B

    def rate_function(self, state, force):
        D = state
        if D <= 1.0:
            return self.A * (1 - D) * np.exp(self.B * force / (1 - D))
        return 0.0

    def cycles_to_failure(self, applied_force):
        return (1.0 / self.A) * exp1(self.B * applied_force)

    def simulate(self, force_timeseries, initstate=0.0, dt=1.0):
        D = [initstate]
        for force in force_timeseries:
            d = D[-1]
            if d >= 1.0:
                D.append(d)
                continue

            k1 = self.rate_function(d, force)
            k2 = self.rate_function(d + 0.5 * dt * k1, force)
            k3 = self.rate_function(d + 0.5 * dt * k2, force)
            k4 = self.rate_function(d + dt * k3, force)

            d_next = d + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            D.append(min(d_next, 1.0))
        return np.array(D[1:], dtype=float)


# -------------------------------------------------
# Analytic helpers
# -------------------------------------------------


def _analytic_cycles_to_failure(A: float, B: float, F: float, Q0: float = 1.0) -> float:
    """
    Analytic result from the manuscript:
        N*(Q0) = (1/A) * E1(BF/Q0)
    """
    return (1.0 / A) * exp1((B * F) / Q0)


def cycles_to_failure_from_simulation(
    model,
    force,
    *,
    Q0: float = 1.0,
    dt: float,
    n_steps: int,
    method: str = "rk4",
) -> float:
    """
    Simulate constant force and return the first cycle where D reaches 1.0.
    Uses clipping semantics (exactly as in the paper).
    """
    init_damage = 1.0 - float(Q0)
    forces = np.full(int(n_steps), float(force), dtype=float)

    D = model.simulate(forces, initstate=init_damage, dt=dt, method=method)

    hit = np.where(D >= 1.0)[0]
    if hit.size == 0:
        return np.inf
    return (int(hit[0]) + 1) * dt


# -------------------------------------------------
# Tests
# -------------------------------------------------


def test_new_and_old_barrett_callaghan_simulations_are_close():
    assert (1.0 * Units.kN) == pytest.approx(1000.0)

    A = 3.90362810e-07
    B = 1.42637606e-03
    F = 3.0 * Units.kN

    dt = 1.0
    n_steps = 4000
    forces = np.full(n_steps, float(F), dtype=float)

    old = OldBarrettCallaghan(A=A, B=B)
    new = BarrettCallaghan(A=A, B=B)

    D_old = old.simulate(forces, initstate=0.0, dt=dt)
    D_new = new.simulate(forces, initstate=0.0, dt=dt, method="rk4")

    diffs = np.abs(D_old - D_new)
    assert np.max(diffs) < 5e-3
    assert np.mean(diffs) < 1e-3

    # Failure index comparison (robust)
    hit_old = np.where(D_old >= 1.0)[0]
    hit_new = np.where(D_new >= 1.0)[0]

    if hit_old.size and hit_new.size:
        rel_diff = abs(hit_old[0] - hit_new[0]) / hit_old[0]
        assert rel_diff < 0.05


def test_cycles_to_failure_matches_analytic_formula_Q0_1():
    A = 3.90362810e-07
    B = 1.42637606e-03
    m = BarrettCallaghan(A=A, B=B)

    F = 3.0 * Units.kN
    numeric = m.cycles_to_failure(F)
    analytic = _analytic_cycles_to_failure(A, B, F, Q0=1.0)

    assert numeric == pytest.approx(analytic, rel=1e-12)


def test_simulated_failure_time_matches_analytic_constant_load_Q0_1():
    A = 3.90362810e-07
    B = 1.42637606e-03
    m = BarrettCallaghan(A=A, B=B)

    F = 3.0 * Units.kN
    N_star = _analytic_cycles_to_failure(A, B, F, Q0=1.0)

    dt = 1.0
    n_steps = int(np.ceil(5 * N_star / dt)) + 1000

    N_hat = cycles_to_failure_from_simulation(m, F, Q0=1.0, dt=dt, n_steps=n_steps)

    assert np.isfinite(N_hat)
    assert N_hat == pytest.approx(N_star, rel=0.03, abs=2 * dt)


def test_simulated_failure_time_matches_analytic_constant_load_Q0_1_fine_dt():
    A = 3.90362810e-07
    B = 1.42637606e-03
    m = BarrettCallaghan(A=A, B=B)

    F = 3.0 * Units.kN
    N_star = _analytic_cycles_to_failure(A, B, F, Q0=1.0)

    dt = 0.05
    n_steps = int(np.ceil(5 * N_star / dt)) + 1000

    N_hat = cycles_to_failure_from_simulation(m, F, Q0=1.0, dt=dt, n_steps=n_steps)

    assert np.isfinite(N_hat)
    assert N_hat == pytest.approx(N_star, rel=0.02, abs=3 * dt)


def test_generalized_cycles_to_failure_Q0():
    # Uses canonical A, B — but Q0 scaling still valid
    A = 3.90362810e-07
    B = 1.42637606e-03
    m = BarrettCallaghan(A=A, B=B)

    F = 3.0 * Units.kN
    Q0 = 0.6
    N_star = _analytic_cycles_to_failure(A, B, F, Q0=Q0)

    dt = 0.05
    n_steps = int(np.ceil(5 * N_star / dt)) + 1000

    N_hat = cycles_to_failure_from_simulation(m, F, Q0=Q0, dt=dt, n_steps=n_steps)

    assert np.isfinite(N_hat)
    assert N_hat == pytest.approx(N_star, rel=0.05, abs=3 * dt)


def test_ict_scales_with_integrity_Q0_times_uct():
    A = 3.90362810e-07
    B = 1.42637606e-03
    m = BarrettCallaghan(A=A, B=B)

    F_uct = m.uct
    Q0 = 0.4

    def Nstar_Q0(F):
        return _analytic_cycles_to_failure(A, B, F, Q0=Q0)

    func = lambda F: Nstar_Q0(F) - 1.0
    F_ict = newton(func, Q0 * F_uct)

    assert F_ict == pytest.approx(Q0 * F_uct, rel=1e-6)


def test_damage_monotone_and_clips_under_constant_force():
    A = 3.90362810e-07
    B = 1.42637606e-03
    m = BarrettCallaghan(A=A, B=B)

    dt = 0.05
    n_steps = 200_000
    forces = np.full(n_steps, float(1.0 * Units.kN), dtype=float)

    D = m.simulate(forces, initstate=0.0, dt=dt, method="rk4")

    assert np.all(np.diff(D) >= -1e-12)
    assert np.all(D <= 1.0 + 1e-12)

    hit = np.where(D >= 1.0)[0]
    if hit.size:
        assert D[hit[0]] == pytest.approx(1.0)
        assert D[-1] == pytest.approx(1.0)
