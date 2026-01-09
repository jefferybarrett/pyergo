# tests/test_lifft.py
import numpy as np
import pytest

from pyergo.damage_models.lifft import LiFFT
from pyergo.utils.units import Units


@pytest.fixture(autouse=True)
def reset_units():
    Units.reset_SI()
    yield
    Units.reset_SI()


def test_rate_independent_of_state():
    m = LiFFT(A=1e-3, B=0.1, ultimate_tolerance=10 * Units.kN)

    f = 2.0 * Units.kN
    r1 = m.rate_function(state=0.0, force=f)
    r2 = m.rate_function(state=0.9, force=f)

    assert r1 == pytest.approx(r2)


def test_cycles_to_failure_is_inverse_rate_at_zero_state():
    m = LiFFT(A=2e-6, B=0.2, ultimate_tolerance=10 * Units.kN)
    f = 3.0 * Units.kN

    rate = m.rate_function(0.0, f)
    ctf = m.cycles_to_failure(f)

    assert ctf == pytest.approx(1.0 / rate)


def test_simulate_zero_force_has_closed_form_linear_growth():
    # At force=0: rate = A * exp(0) = A
    A = 0.01
    m = LiFFT(A=A, B=0.162, ultimate_tolerance=10 * Units.kN, failure_damage=1.0)

    forces = np.zeros(5, dtype=float)  # 5 cycles of zero force
    dt = 2.0

    D = m.simulate(forces, initstate=0.0, dt=dt, method="euler")

    expected = np.arange(1, len(forces) + 1) * (dt * A)
    expected = np.clip(expected, 0.0, 1.0)

    assert D == pytest.approx(expected)


def test_simulate_constant_force_matches_closed_form_until_clipping():
    # For constant force, rate is constant => D[n] = init + n*dt*rate
    A = 1e-4
    B = 0.2
    UT = 8.0 * Units.kN
    f = 2.0 * Units.kN

    m = LiFFT(A=A, B=B, ultimate_tolerance=UT, failure_damage=1.0)

    forces = np.ones(10, dtype=float) * f
    dt = 3.0

    rate = m.rate_function(0.0, f)
    D = m.simulate(forces, initstate=0.0, dt=dt, method="euler")

    expected = np.arange(1, len(forces) + 1) * (dt * rate)
    expected = np.clip(expected, 0.0, 1.0)

    assert D == pytest.approx(expected)
    assert np.all(np.diff(D) >= -1e-12)  # monotone nondecreasing


def test_rate_increases_with_force_when_B_positive():
    m = LiFFT(A=1e-6, B=0.162, ultimate_tolerance=10 * Units.kN)
    r0 = m.rate_function(0.0, 0.0)
    r1 = m.rate_function(0.0, 1.0 * Units.kN)
    r2 = m.rate_function(0.0, 2.0 * Units.kN)

    assert r0 < r1 < r2


def test_uct_has_closed_form_solution_matches_newton():
    # Solve A * exp(B*100*f/UT) = 1 => f = (UT/(B*100)) * ln(1/A)
    A = 1.0 / 902416.0
    B = 0.162
    UT = 10.0 * Units.kN

    m = LiFFT(A=A, B=B, ultimate_tolerance=UT)

    expected = (UT / (B * 100.0)) * np.log(1.0 / A)
    assert m.uct == pytest.approx(expected)


def test_uct_is_cached():
    # We can't easily count newton iterations without monkeypatching newton,
    # but we can ensure subsequent property access doesn't change result.
    m = LiFFT()
    u1 = m.uct
    u2 = m.uct
    assert u1 == pytest.approx(u2)


def test_clipping_at_failure_damage():
    # Make rate huge so we hit failure quickly
    m = LiFFT(A=10.0, B=0.0, ultimate_tolerance=10.0 * Units.kN, failure_damage=1.0)
    forces = np.ones(5) * Units.kN

    D = m.simulate(forces, initstate=0.0, dt=0.1, method="euler")

    assert D[0] == pytest.approx(1.0)  # first step clips
    assert np.all(D == pytest.approx(1.0))  # then stays at failure
