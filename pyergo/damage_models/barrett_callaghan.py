from pyergo.damage_models.base import DamageModel
from pyergo.utils.units import Units
from scipy.optimize import newton
from scipy.special import exp1
import numpy as np


class BarrettCallaghan(DamageModel):
    """
    Barrett-Callaghan cumulative damage model.

    This model implements a phenomenological damage accumulation law in which the
    instantaneous damage rate depends on both the applied force and the remaining
    undamaged capacity of the system. Damage accelerates as the damage state approaches
    the failure threshold, producing a finite-time failure behavior.

    The model is parameterized by two scalar coefficients, ``A`` and ``B``, which control
    the baseline damage rate and the force sensitivity, respectively. The formulation is
    intended for cumulative fatigue or overload-style damage processes and is described
    in detail in an upcoming manuscript by Barrett & Callaghan.

    Notes
    -----
    - Damage is monotonic and non-healing in this implementation.
    - As ``D → failure_damage``, the damage rate increases rapidly; a small numerical
      guard is used internally to avoid division by zero.
    - The interpretation and units of ``force`` must be consistent with the values of
      ``A`` and ``B``.

    Parameters
    ----------
    A : float, optional
        Baseline damage-rate coefficient. Smaller values correspond to slower damage
        accumulation. Defaults to ``2.47e-11``.
    B : float, optional
        Force sensitivity coefficient controlling how strongly damage accelerates with
        increasing force. Defaults to ``0.00203``.
    failure_damage : float, optional
        Damage threshold treated as failure. Defaults to 1.0.
    """

    def __init__(self, A=2.47e-11, B=0.00203, *, failure_damage=1.0):
        super().__init__(failure_damage=failure_damage)
        self.A, self.B = float(A), float(B)

    def rate_function(self, state, force):
        """
        Compute the instantaneous damage accumulation rate.

        Parameters
        ----------
        state : float
            Current damage state ``D``.
        force : float
            Scalar applied force/load.

        Returns
        -------
        float
            Damage rate ``dD/dt`` (or per-cycle rate), consistent with the chosen ``dt``.

        Notes
        -----
        The rate increases exponentially with force and is modulated by the remaining
        undamaged fraction ``(1 - D)``. A small numerical epsilon is used to prevent
        division by zero as ``D`` approaches ``failure_damage``.
        """
        d = state
        if d >= self.failure_damage:
            return 0.0
        eps = 1e-15
        denom = max(1.0 - d, eps)
        return self.A * denom * np.exp(self.B * force / denom)

    def cycles_to_failure(self, applied_force):
        """
        Estimate the expected number of cycles to failure at a constant force.

        Parameters
        ----------
        applied_force : float
            Constant scalar force/load applied each cycle.

        Returns
        -------
        float
            Expected number of cycles to failure under constant loading.

        Notes
        -----
        This closed-form expression corresponds to the special case of constant force and
        zero initial damage.
        """
        return (1.0 / self.A) * exp1(self.B * applied_force)

    def _estimate_uct(self):
        """
        Estimate the model's ultimate tolerance.

        The ultimate tolerance is defined here as the force level at which the expected
        number of cycles to failure is approximately one.

        Returns
        -------
        float
            Estimated ultimate tolerance (force) for this model.
        """
        func = lambda f: self.cycles_to_failure(f) - 1.0
        return newton(func, 1.0 * Units.kN)
