from pyergo.damage_models.base import DamageModel
from pyergo.utils.units import Units
from scipy.optimize import newton
from scipy.special import exp1
import numpy as np


class BarrettCallaghan(DamageModel):
    """Barrett-Callaghan model (upcoming manuscript)."""

    def __init__(self, A=2.47e-11, B=0.00203, *, failure_damage=1.0):
        super().__init__(failure_damage=failure_damage)
        self.A, self.B = float(A), float(B)

    def rate_function(self, state, force):
        d = state
        if d >= self.failure_damage:
            return 0.0
        # guard against division by zero as d -> 1
        eps = 1e-15
        denom = max(1.0 - d, eps)
        return self.A * denom * np.exp(self.B * force / denom)

    def cycles_to_failure(self, applied_force):
        return (1.0 / self.A) * exp1(self.B * applied_force)

    def _estimate_uct(self):
        func = lambda f: self.cycles_to_failure(f) - 1.0
        return newton(func, 1.0 * Units.kN)
