# units.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

Dims = Tuple[int, int, int]  # (L, T, M) minimal dims for mechanics


class UnitsError(ValueError):
    pass


@dataclass(frozen=True)
class Unit:
    """A unit with SI scale and simple dimensions (L, T, M).

    si_scale: multiplier to convert a quantity in this unit into SI
             (meters, seconds, kilograms)
    dims: exponents for (L, T, M)
    """

    name: str
    si_scale: float
    dims: Dims

    def __mul__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit(
            name=f"({self.name}*{other.name})",
            si_scale=self.si_scale * other.si_scale,
            dims=(self.dims[0] + other.dims[0], self.dims[1] + other.dims[1], self.dims[2] + other.dims[2]),
        )

    def __truediv__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit(
            name=f"({self.name}/{other.name})",
            si_scale=self.si_scale / other.si_scale,
            dims=(self.dims[0] - other.dims[0], self.dims[1] - other.dims[1], self.dims[2] - other.dims[2]),
        )

    def __pow__(self, p: int) -> "Unit":
        if not isinstance(p, int):
            raise UnitsError("Only integer powers are supported.")
        return Unit(
            name=f"({self.name}^{p})",
            si_scale=self.si_scale**p,
            dims=(self.dims[0] * p, self.dims[1] * p, self.dims[2] * p),
        )

    def __rmul__(self, value: float) -> float:
        # Returns value expressed in *current base units*
        return float(value) * Units._to_base_factor(self)

    def __rtruediv__(self, value: float) -> float:
        # value / unit -> value expressed in inverse-base units
        return float(value) / Units._to_base_factor(self)


class Units:
    """Minimal unit registry with changeable base units.

    Base system defaults to SI: m, s, kg.

    Usage:
        v = 1.0 * Units.m / Units.s
        d = 8.5 * Units.light_minutes
        Units.change_base_length(Units.ft)
    """

    # --- current bases, expressed in SI scale ---
    _base_L = 1.0  # meters per base-length unit
    _base_T = 1.0  # seconds per base-time unit
    _base_M = 1.0  # kilograms per base-mass unit

    @classmethod
    def _to_base_factor(cls, u: Unit) -> float:
        """Convert 1*u into current base units (a pure scale factor)."""
        denom = (cls._base_L ** u.dims[0]) * (cls._base_T ** u.dims[1]) * (cls._base_M ** u.dims[2])
        return u.si_scale / denom

    @staticmethod
    def _assert_pure(u: Unit, dims: Dims, label: str) -> None:
        if u.dims != dims:
            raise UnitsError(f"{label} base must have dims {dims}, got {u.dims} from {u.name}.")

    # --- base changers ---
    @classmethod
    def change_base_length(cls, u: Unit) -> None:
        cls._assert_pure(u, (1, 0, 0), "Length")
        cls._base_L = u.si_scale

    @classmethod
    def change_base_time(cls, u: Unit) -> None:
        cls._assert_pure(u, (0, 1, 0), "Time")
        cls._base_T = u.si_scale

    @classmethod
    def change_base_mass(cls, u: Unit) -> None:
        cls._assert_pure(u, (0, 0, 1), "Mass")
        cls._base_M = u.si_scale

    @classmethod
    def reset_SI(cls) -> None:
        cls._base_L, cls._base_T, cls._base_M = 1.0, 1.0, 1.0

    # --- base units (SI primitives) ---
    s = Unit("s", 1.0, (0, 1, 0))
    m = Unit("m", 1.0, (1, 0, 0))
    kg = Unit("kg", 1.0, (0, 0, 1))

    # --- angles (dimensionless primitives) ---
    rad = Unit("rad", 1.0, (0, 0, 0))
    deg = Unit("deg", math.pi / 180.0, (0, 0, 0))

    # --- prefixes (scalars) ---
    milli = 1e-3
    micro = 1e-6
    nano = 1e-9
    k = 1e3
    M = 1e6
    G = 1e9


# -----------------------
# Derived units (defined after class creation)
# -----------------------

# time
Units.minute = Unit("min", 60.0 * Units.s.si_scale, (0, 1, 0))
Units.hour = Unit("h", 60.0 * Units.minute.si_scale, (0, 1, 0))
Units.day = Unit("day", 24.0 * Units.hour.si_scale, (0, 1, 0))

# length
Units.mm = Unit("mm", Units.milli * Units.m.si_scale, (1, 0, 0))
Units.cm = Unit("cm", 1e-2 * Units.m.si_scale, (1, 0, 0))
Units.km = Unit("km", Units.k * Units.m.si_scale, (1, 0, 0))

Units.inch = Unit("in", 0.0254 * Units.m.si_scale, (1, 0, 0))
Units.ft = Unit("ft", 0.3048 * Units.m.si_scale, (1, 0, 0))
Units.yd = Unit("yd", 0.9144 * Units.m.si_scale, (1, 0, 0))

# mass
Units.g = Unit("g", 1e-3 * Units.kg.si_scale, (0, 0, 1))
Units.mg = Unit("mg", 1e-3 * Units.g.si_scale, (0, 0, 1))

# mechanics
# Force: N = kg*m/s^2
Units.N = Unit("N", (Units.kg * Units.m / (Units.s**2)).si_scale, (1, -2, 1))
Units.kN = Unit("kN", 1e3 * Units.N.si_scale, (1, -2, 1))
Units.MN = Unit("MN", 1e6 * Units.N.si_scale, (1, -2, 1))
Units.GN = Unit("GN", 1e9 * Units.N.si_scale, (1, -2, 1))

# Pressure: Pa = N/m^2
Units.Pa = Unit("Pa", (Units.N / (Units.m**2)).si_scale, (-1, -2, 1))
Units.kPa = Unit("kPa", 1e3 * Units.Pa.si_scale, (-1, -2, 1))
Units.MPa = Unit("MPa", 1e6 * Units.Pa.si_scale, (-1, -2, 1))
Units.GPa = Unit("GPa", 1e9 * Units.Pa.si_scale, (-1, -2, 1))

# Energy: J = N*m
Units.J = Unit("J", (Units.N * Units.m).si_scale, (2, -2, 1))
Units.kJ = Unit("kJ", 1e3 * Units.J.si_scale, (2, -2, 1))
Units.MJ = Unit("MJ", 1e6 * Units.J.si_scale, (2, -2, 1))

# Power: W = J/s
Units.W = Unit("W", (Units.J / Units.s).si_scale, (2, -3, 1))
Units.kW = Unit("kW", 1e3 * Units.W.si_scale, (2, -3, 1))
Units.MW = Unit("MW", 1e6 * Units.W.si_scale, (2, -3, 1))

# Impulse: N*s
Units.Ns = Unit("Ns", (Units.N * Units.s).si_scale, (1, -1, 1))
Units.kNs = Unit("kNs", 1e3 * Units.Ns.si_scale, (1, -1, 1))
Units.MNs = Unit("MNs", 1e6 * Units.Ns.si_scale, (1, -1, 1))

# Astronomy-ish convenience (length units)
Units._c = 299_792_458.0  # m/s (scalar in SI)
Units.light_second = Unit("light_second", Units._c * Units.m.si_scale, (1, 0, 0))
Units.light_minute = Unit("light_minute", (60.0 * Units._c) * Units.m.si_scale, (1, 0, 0))
Units.light_minutes = Units.light_minute  # alias


__all__ = [
    "Unit",
    "Units",
    "UnitsError",
]
