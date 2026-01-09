# tests/test_miner_palmgren.py
import numpy as np
import pytest

from pyergo.damage_models.miner_palmgren import MinerPalmgren
from pyergo.utils.units import Units


@pytest.fixture(autouse=True)
def reset_units():
    # Keep tests deterministic if Units bases can be changed elsewhere
    Units.reset_SI()
    yield
    Units.reset_SI()


def test_rate_function_is_inverse_sn():
    SN = lambda f: 200.0  # 200 cycles to failure regardless of force
    model = MinerPalmgren(SN)

    # state should not matter for Miner-Palmgren rate
    r1 = model.rate_function(state=0.0, force=5.0 * Units.kN)
    r2 = model.rate_function(state=0.9, force=5.0 * Units.kN)

    assert r1 == pytest.approx(1.0 / 200.0)
    assert r2 == pytest.approx(1.0 / 200.0)


def test_simulate_constant_sn_matches_closed_form():
    # If SN is constant Nf, each cycle adds dt/Nf damage.
    Nf = 50.0
    SN = lambda f: Nf
    model = MinerPalmgren(SN)

    forces = np.array([1, 2, 3, 4, 5], dtype=float) * Units.kN  # doesn't matter
    dt = 1.0

    D = model.simulate(forces, initstate=0.0, dt=dt, method="euler")
    expected = np.arange(1, len(forces) + 1) * (dt / Nf)

    assert D == pytest.approx(expected)


def test_simulate_variable_forces_matches_sum_of_increments():
    # Choose an SN curve that varies with force but is still deterministic
    # SN(force) = 100 + force/kN (dimensionless-ish)
    SN = lambda f: 100.0 + (f / (1.0 * Units.kN))
    model = MinerPalmgren(SN)

    forces = np.array([0.0, 1.0, 2.0, 3.0]) * Units.kN
    dt = 2.0

    D = model.simulate(forces, initstate=0.0, dt=dt, method="euler")

    increments = np.array([dt / SN(f) for f in forces], dtype=float)
    expected = np.cumsum(increments)

    assert D == pytest.approx(expected)


def test_uct_for_linear_sn_has_closed_form_solution():
    # SN(force) = a*force + b; solve SN=1 => force = (1-b)/a
    a = 2.0  # cycles per N (in whatever consistent force units you use)
    b = -3.0

    SN = lambda f: a * f + b
    model = MinerPalmgren(SN)

    # closed form root
    expected = (1.0 - b) / a
    assert model.uct == pytest.approx(expected)


def test_uct_is_cached_and_calls_sn_multiple_times_but_estimates_once():
    calls = {"n": 0}

    def SN(f):
        calls["n"] += 1
        # root at f = 7*kN (in SI base, Units.kN is 1000.0, so be explicit)
        return (f / (1.0 * Units.kN)) - 7.0 + 1.0  # => SN=1 at 7*kN

    model = MinerPalmgren(SN)

    u1 = model.uct
    u2 = model.uct

    assert u1 == pytest.approx(7.0 * Units.kN)
    assert u2 == pytest.approx(7.0 * Units.kN)

    # cached_property => only one estimation run, but Newton will still call SN multiple times.
    # So we can't assert exact call count, but we can assert it didn't double.
    assert calls["n"] > 1
    n_after_first = calls["n"]

    _ = model.uct
    assert calls["n"] == n_after_first  # no more calls after cached result


def test_failure_damage_clipping_inherited_from_base():
    # Miner-Palmgren rate doesn't depend on D, so it will keep increasing,
    # but the base class clips at failure_damage inside step().
    SN = lambda f: 1.0  # rate = 1 -> damage increases fast
    model = MinerPalmgren(SN, failure_damage=1.0)

    forces = np.ones(10) * Units.kN
    D = model.simulate(forces, initstate=0.0, dt=0.2, method="euler")

    # It should reach 1.0 and then stay there
    assert np.all(D <= 1.0 + 1e-12)
    assert D[-1] == pytest.approx(1.0)
    # monotonic nondecreasing
    assert np.all(np.diff(D) >= -1e-12)
