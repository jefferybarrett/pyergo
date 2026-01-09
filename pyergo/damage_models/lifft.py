from pyergo.damage_models.base import DamageModel
from pyergo.utils.units import Units
from scipy.optimize import newton
import numpy as np


class LiFFT(DamageModel):
    """LiFFT cumulative damage model (Gallagher et al., 2017)."""

    def __init__(self, A=1 / 902416.0, B=0.162, ultimate_tolerance=10 * Units.kN, *, failure_damage=1.0):
        super().__init__(failure_damage=failure_damage)
        self.A, self.B = float(A), float(B)
        self.ultimate_tolerance = float(ultimate_tolerance)

    def rate_function(self, state, force):
        # classic LiFFT form (as you had it)
        return self.A * np.exp(self.B * 100.0 * force / self.ultimate_tolerance)

    def cycles_to_failure(self, force):
        return 1.0 / self.rate_function(0.0, force)

    def _estimate_uct(self):
        func = lambda f: self.cycles_to_failure(f) - 1.0
        return newton(func, 1.0 * Units.kN)
