from pyergo.damage_models.base import DamageModel
from pyergo.utils.units import Units
from scipy.optimize import newton


class MinerPalmgren(DamageModel):
    """
    Miner–Palmgren cumulative damage model.

    This class implements the classical linear cumulative damage hypothesis commonly
    attributed to Miner and Palmgren. Damage accumulates linearly as the fraction of
    expended life relative to an S–N (stress–life) curve, with failure occurring when
    cumulative damage reaches the specified threshold.

    In this formulation, the S–N curve is supplied by the user as a callable mapping
    applied load (or stress proxy) to the expected number of cycles to failure.

    Notes
    -----
    - Damage accumulation is linear and independent of the current damage state.
    - Load interaction effects and sequence effects are not captured (as in the classic
      Miner–Palmgren hypothesis).
    - The quality of predictions depends entirely on the validity of the supplied S–N
      curve.

    Parameters
    ----------
    SN_curve : callable
        Function mapping ``force`` (or stress/load proxy) to the expected number of
        cycles to failure, ``N(force)``.
    failure_damage : float, optional
        Damage threshold treated as failure. Defaults to 1.0.
    """

    def __init__(self, SN_curve, *, failure_damage=1.0):
        super().__init__(failure_damage=failure_damage)
        self.SN_curve = SN_curve

    def rate_function(self, state, force):
        """
        Compute the instantaneous damage accumulation rate.

        Parameters
        ----------
        state : float
            Current damage state ``D``. Included for API consistency but not used in
            this linear damage model.
        force : float
            Scalar applied force/load.

        Returns
        -------
        float
            Damage rate ``dD/dt`` (or per-cycle rate).

        Notes
        -----
        The rate is defined as the reciprocal of the S–N curve value, corresponding to
        consuming ``1 / N(force)`` units of damage per cycle.
        """
        return 1.0 / self.SN_curve(force)

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
        func = lambda force: self.SN_curve(force) - 1.0
        return newton(func, 1.0 * Units.kN)
