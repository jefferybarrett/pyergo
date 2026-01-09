from pyergo.damage_models.base import DamageModel
from pyergo.utils.units import Units
from scipy.optimize import newton


class MinerPalmgren(DamageModel):
    def __init__(self, SN_curve, *, failure_damage=1.0):
        super().__init__(failure_damage=failure_damage)
        self.SN_curve = SN_curve

    def rate_function(self, state, force):
        return 1.0 / self.SN_curve(force)

    def _estimate_uct(self):
        func = lambda force: self.SN_curve(force) - 1.0
        return newton(func, 1.0 * Units.kN)
