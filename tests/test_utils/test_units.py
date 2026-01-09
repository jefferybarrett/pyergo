# test_units.py
import math
import pytest
import numpy as np
from pyergo.utils.units import Units, Unit, UnitsError


def approx(a, b, rel=1e-12, abs_=0.0):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


@pytest.fixture(autouse=True)
def reset_units():
    # Ensure tests don't leak base changes into each other
    Units.reset_SI()
    yield
    Units.reset_SI()


def test_default_base_is_SI():
    # 1 m in base units should be 1.0 (since base length is meters)
    assert 1.0 * Units.m == pytest.approx(1.0)
    # 1 s in base units should be 1.0
    assert 1.0 * Units.s == pytest.approx(1.0)
    # 1 kg in base units should be 1.0
    assert 1.0 * Units.kg == pytest.approx(1.0)


def test_ndarray_interaction():
    test_array = np.array([1, 2, 3]) * Units.m
    assert test_array[0] == pytest.approx(1)


def test_length_conversions_in_SI_base():
    assert (1000.0 * Units.mm) == pytest.approx(1.0)  # 1000 mm = 1 m
    assert (100.0 * Units.cm) == pytest.approx(1.0)  # 100 cm = 1 m
    assert (1.0 * Units.km) == pytest.approx(1000.0)  # 1 km = 1000 m
    assert (12.0 * Units.inch) == pytest.approx(0.3048)  # 12 in = 0.3048 m
    assert (1.0 * Units.ft) == pytest.approx(0.3048)  # 1 ft = 0.3048 m


def test_change_base_length_to_feet():
    # After switching base length to feet, "1 meter" should be ~3.28084 base-lengths
    Units.change_base_length(Units.ft)

    one_meter_in_base = 1.0 * Units.m
    assert one_meter_in_base == pytest.approx(1.0 / 0.3048)

    # And 1 ft should be exactly 1 base-length now
    assert (1.0 * Units.ft) == pytest.approx(1.0)

    # cm should now be expressed in feet
    assert (30.48 * Units.cm) == pytest.approx(1.0)  # 30.48 cm = 1 ft


def test_change_base_time():
    # Switch base time to minutes: 1 second should be 1/60 in base units
    Units.change_base_time(Units.minute)

    assert (1.0 * Units.minute) == pytest.approx(1.0)  # 1 minute is base
    assert (60.0 * Units.s) == pytest.approx(1.0)  # 60 s = 1 minute
    assert (1.0 * Units.s) == pytest.approx(1.0 / 60.0)


def test_change_base_mass():
    # Switch base mass to grams: 1 kg should be 1000 in base units
    Units.change_base_mass(Units.g)

    assert (1.0 * Units.g) == pytest.approx(1.0)
    assert (1.0 * Units.kg) == pytest.approx(1000.0)
    assert (500.0 * Units.g) == pytest.approx(500.0)


def test_velocity_expression_changes_with_base_length_only():
    v_si = 1.0 * Units.m / Units.s  # base is m/s => 1.0

    Units.change_base_length(Units.ft)  # now base is ft/s
    v_ft_per_s = 1.0 * Units.m / Units.s

    assert v_si == pytest.approx(1.0)
    assert v_ft_per_s == pytest.approx(1.0 / 0.3048)  # ~3.28084 ft/s


def test_unit_algebra_dims_and_scaling():
    # Construct speed unit explicitly
    u = Units.m / Units.s
    assert isinstance(u, Unit)
    assert u.dims == (1, -1, 0)
    assert u.si_scale == pytest.approx(1.0)

    # Area
    area_u = Units.m**2
    assert area_u.dims == (2, 0, 0)
    assert (2.0 * area_u) == pytest.approx(2.0)  # 2 m^2 in SI base

    # Acceleration unit check: m/s^2
    acc_u = Units.m / (Units.s**2)
    assert acc_u.dims == (1, -2, 0)
    assert (9.80665 * acc_u) == pytest.approx(9.80665)


def test_force_unit_newton_is_consistent():
    # In SI base, 1 N should be 1.0 in base force units if base is SI (since N is derived from base)
    assert (1.0 * Units.N) == pytest.approx(1.0)

    # If we change base length to feet, Newton should scale accordingly (since force dims include L)
    Units.change_base_length(Units.ft)
    # base force unit becomes kg*ft/s^2, so 1 N (= kg*m/s^2) is 1/0.3048 in that base
    assert (1.0 * Units.N) == pytest.approx(1.0 / 0.3048)


def test_pressure_energy_power_units_scale_with_base_length():
    # Pa dims: (-1, -2, 1) => scales with L^-1
    Units.change_base_length(Units.ft)
    pa_in_ft_base = 1.0 * Units.Pa
    # Since base L is bigger (ft), Pa should numerically be smaller by factor (m/ft)^{-1} = ft/m = 0.3048?
    # Derivation: factor = si_scale / base_L^(-1) = 1 / (0.3048^-1) = 0.3048
    assert pa_in_ft_base == pytest.approx(0.3048)

    Units.reset_SI()
    Units.change_base_length(Units.ft)

    # J dims: (2, -2, 1) => scales with L^2
    j_in_ft_base = 1.0 * Units.J
    assert j_in_ft_base == pytest.approx((1.0 / 0.3048) ** 2)

    # W dims: (2, -3, 1) => scales with L^2
    w_in_ft_base = 1.0 * Units.W
    assert w_in_ft_base == pytest.approx((1.0 / 0.3048) ** 2)


def test_degree_radian_conversion():
    assert (180.0 * Units.deg) == pytest.approx(math.pi)
    assert (1.0 * Units.rad) == pytest.approx(1.0)


def test_light_minute_in_meters():
    # 1 light-minute = c * 60 meters
    c = 299_792_458.0
    assert (1.0 * Units.light_minute) == pytest.approx(c * 60.0)
    assert (8.5 * Units.light_minutes) == pytest.approx(8.5 * c * 60.0)


def test_light_minute_in_feet_after_base_change():
    Units.change_base_length(Units.ft)
    c = 299_792_458.0

    one_light_minute_ft = 1.0 * Units.light_minute
    assert one_light_minute_ft == pytest.approx((c * 60.0) / 0.3048)


def test_invalid_base_changes_raise():
    with pytest.raises(UnitsError):
        Units.change_base_length(Units.s)  # wrong dims
    with pytest.raises(UnitsError):
        Units.change_base_time(Units.m)  # wrong dims
    with pytest.raises(UnitsError):
        Units.change_base_mass(Units.minute)  # wrong dims


def test_rtruediv_behavior():
    # 1 / second => in base, should be 1.0 (since base time is seconds)
    assert (1.0 / Units.s) == pytest.approx(1.0)

    Units.change_base_time(Units.minute)
    # Now base time is minutes, so 1 / second should be 60 (per base-time)
    assert (1.0 / Units.s) == pytest.approx(60.0)
