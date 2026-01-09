import numpy as np
from functools import cached_property
from abc import ABC, abstractmethod


class DamageModel(ABC):
    """Cumulative damage model: define rate_function(D, force) and (optionally) uct.

    Conventions:
    - D in [0, 1] is cumulative damage (1 = failure)
    - dt is the "cycle duration" in whatever units your model assumes
    """

    def __init__(self, *, failure_damage=1.0):
        self.failure_damage = float(failure_damage)

    @cached_property
    def uct(self):
        """Ultimate tolerance (force where expected cycles-to-failure ~= 1)."""
        return self._estimate_uct()

    @abstractmethod
    def rate_function(self, state: float, force: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def _estimate_uct(self) -> float:
        raise NotImplementedError

    def step(self, d: float, force: float, *, dt: float = 1.0, method: str = "euler") -> float:
        """One integration step. Returns d_next (clipped to failure_damage)."""
        if d >= self.failure_damage:
            return d

        if method == "euler":
            d_next = d + dt * self.rate_function(d, force)
        elif method == "rk4":
            k1 = self.rate_function(d, force)
            k2 = self.rate_function(d + 0.5 * dt * k1, force)
            k3 = self.rate_function(d + 0.5 * dt * k2, force)
            k4 = self.rate_function(d + dt * k3, force)
            d_next = d + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'euler' or 'rk4'.")

        # clip at failure
        return min(float(d_next), self.failure_damage)

    def simulate(self, force_timeseries, initstate: float = 0.0, *, dt: float = 1.0, method: str = "euler"):
        """Simulate cumulative damage over a force time series."""
        d = float(initstate)
        out = np.empty(len(force_timeseries), dtype=float)
        for i, f in enumerate(force_timeseries):
            d = self.step(d, float(f), dt=dt, method=method)
            out[i] = d
        return out
