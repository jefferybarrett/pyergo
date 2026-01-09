# tests/test_damage_base.py
import numpy as np
import pytest
from pyergo.damage_models.base import DamageModel


class ConstantRateModel(DamageModel):
    """rate_function ignores state/force and returns a constant."""

    def __init__(self, rate: float, *, failure_damage: float = 1.0):
        super().__init__(failure_damage=failure_damage)
        self.rate = float(rate)
        self._uct_calls = 0

    def rate_function(self, state: float, force: float) -> float:
        return self.rate

    def _estimate_uct(self) -> float:
        self._uct_calls += 1
        return 123.0


class LinearInForceModel(DamageModel):
    """rate_function = a * force (simple, deterministic)."""

    def __init__(self, a: float = 1.0, *, failure_damage: float = 1.0):
        super().__init__(failure_damage=failure_damage)
        self.a = float(a)

    def rate_function(self, state: float, force: float) -> float:
        return self.a * float(force)

    def _estimate_uct(self) -> float:
        return 1.0


def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        DamageModel()  # abstract


def test_step_euler_constant_rate():
    m = ConstantRateModel(rate=0.1)
    d0 = 0.0
    d1 = m.step(d0, force=999.0, dt=1.0, method="euler")
    assert d1 == pytest.approx(0.1)

    d2 = m.step(d1, force=0.0, dt=2.0, method="euler")
    assert d2 == pytest.approx(0.1 + 2.0 * 0.1)


def test_step_rk4_constant_rate_matches_euler():
    # For constant derivative, RK4 should equal Euler exactly (up to float noise)
    m = ConstantRateModel(rate=0.25)
    d0 = 0.0
    dt = 3.0
    euler = m.step(d0, force=1.0, dt=dt, method="euler")
    rk4 = m.step(d0, force=1.0, dt=dt, method="rk4")
    assert rk4 == pytest.approx(euler)


def test_step_unknown_method_raises():
    m = ConstantRateModel(rate=0.1)
    with pytest.raises(ValueError):
        _ = m.step(0.0, force=1.0, method="beepboop")


def test_step_clips_at_failure_damage():
    m = ConstantRateModel(rate=0.6, failure_damage=1.0)
    # dt=2 would try to add 1.2, but should clip to 1.0
    d1 = m.step(0.0, force=1.0, dt=2.0, method="euler")
    assert d1 == pytest.approx(1.0)


def test_step_holds_constant_after_failure():
    m = ConstantRateModel(rate=0.9, failure_damage=1.0)
    d_fail = m.step(0.2, force=1.0, dt=2.0, method="euler")
    assert d_fail == pytest.approx(1.0)

    # Once failed, should not change
    d_next = m.step(d_fail, force=1e9, dt=100.0, method="rk4")
    assert d_next == pytest.approx(1.0)


def test_simulate_shape_dtype_and_values_euler_constant_rate():
    m = ConstantRateModel(rate=0.1)
    forces = np.array([10.0, 20.0, 30.0], dtype=float)

    out = m.simulate(forces, initstate=0.0, dt=1.0, method="euler")
    assert out.shape == (3,)
    assert out.dtype == float

    # cumulative: 0.1, 0.2, 0.3
    assert out[0] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.2)
    assert out[2] == pytest.approx(0.3)


def test_simulate_respects_initstate_and_clipping():
    m = ConstantRateModel(rate=0.5, failure_damage=1.0)
    forces = [1.0, 1.0, 1.0]

    out = m.simulate(forces, initstate=0.6, dt=1.0, method="euler")
    # steps: 0.6+0.5=1.0 clipped, then remain at 1.0
    assert np.allclose(out, [1.0, 1.0, 1.0])


def test_simulate_linear_in_force():
    m = LinearInForceModel(a=0.1)
    forces = np.array([0.0, 2.0, 4.0])

    out = m.simulate(forces, initstate=0.0, dt=1.0, method="euler")
    # increments: 0, 0.2, 0.4 => cumulative: 0, 0.2, 0.6
    assert out.tolist() == pytest.approx([0.0, 0.2, 0.6])


def test_uct_is_cached():
    m = ConstantRateModel(rate=0.1)

    # First access calls _estimate_uct
    v1 = m.uct
    assert v1 == pytest.approx(123.0)
    assert m._uct_calls == 1

    # Second access should not call again
    v2 = m.uct
    assert v2 == pytest.approx(123.0)
    assert m._uct_calls == 1
