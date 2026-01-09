from pyergo.damage_models.base import DamageModel
from pyergo.utils.units import Units
from scipy.optimize import newton
import numpy as np


class LiFFT(DamageModel):
    """
    LiFFT cumulative damage model.

    This class implements the LiFFT (Lifting Fatigue Failure Tool) damage formulation
    described by Gallagher et al. (2017), which models cumulative fatigue damage as an
    exponential function of applied load relative to an ultimate tolerance.

    The model assumes monotonic damage accumulation with no healing and is typically
    applied in repetitive lifting or load-bearing scenarios where force is the primary
    driver of fatigue-related failure.

    Notes
    -----
    - Damage accumulation is independent of the current damage state in this formulation;
      the rate depends only on the applied force.
    - The model is calibrated such that higher forces lead to exponentially fewer cycles
      to failure.
    - The interpretation of ``force`` must be consistent with the assumed ultimate
      tolerance and calibration constants.

    References
    ----------
    Gallagher, S., Marras, W. S., Davis, K. G., & Waters, T. R. (2017).
    *The Lifting Fatigue Failure Tool (LiFFT): A fatigue-based method for cumulative
    lifting risk assessment.* Human Factors, 59(3), 418–431.

    Parameters
    ----------
    A : float, optional
        Baseline damage-rate coefficient. Defaults to ``1 / 902416.0``.
    B : float, optional
        Exponential force sensitivity coefficient. Defaults to ``0.162``.
    ultimate_tolerance : float, optional
        Ultimate tolerance force used to normalize the applied force. Defaults to
        ``10 * Units.kN``.
    failure_damage : float, optional
        Damage threshold treated as failure. Defaults to 1.0.
    """

    def __init__(
        self,
        A=1 / 902416.0,
        B=0.162,
        ultimate_tolerance=10 * Units.kN,
        *,
        failure_damage=1.0,
    ):
        super().__init__(failure_damage=failure_damage)
        self.A, self.B = float(A), float(B)
        self.ultimate_tolerance = float(ultimate_tolerance)

    def rate_function(self, state, force):
        """
        Compute the instantaneous damage accumulation rate.

        Parameters
        ----------
        state : float
            Current damage state ``D``. Included for API consistency but not used in
            this model.
        force : float
            Scalar applied force/load.

        Returns
        -------
        float
            Damage rate ``dD/dt`` (or per-cycle rate), consistent with the chosen ``dt``.

        Notes
        -----
        This is the classic LiFFT formulation, where damage grows exponentially with
        force normalized by the ultimate tolerance. The factor of ``100`` reflects the
        original model scaling.
        """
        return self.A * np.exp(self.B * 100.0 * force / self.ultimate_tolerance)

    def cycles_to_failure(self, force):
        """
        Estimate the expected number of cycles to failure at a constant force.

        Parameters
        ----------
        force : float
            Constant scalar applied force/load.

        Returns
        -------
        float
            Expected number of cycles to failure under constant loading.

        Notes
        -----
        This expression assumes zero initial damage and constant force across cycles.
        """
        return 1.0 / self.rate_function(0.0, force)

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
