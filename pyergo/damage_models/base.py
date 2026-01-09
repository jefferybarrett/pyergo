import numpy as np
from functools import cached_property
from abc import ABC, abstractmethod


class DamageModel(ABC):
    """
    Abstract base class for cumulative damage models.

    A damage model describes how an internal damage state :math:`D` evolves in response
    to an applied load (here represented as a scalar ``force``). Subclasses implement a
    damage accumulation law via :meth:`rate_function` and provide an estimate of an
    "ultimate tolerance" via :meth:`_estimate_uct`.

    Conventions
    -----------
    - ``D`` is cumulative damage. By default, ``D`` is interpreted on ``[0, 1]`` where
      ``1`` corresponds to failure, but the failure threshold is configurable via
      ``failure_damage``.
    - ``force`` is a scalar load measure used by the model (units are model-specific).
    - ``dt`` is the integration step size / cycle duration in whatever time or cycle
      units the model assumes. The meaning of ``dt`` is entirely determined by the
      subclass (e.g., seconds, cycles, blocks, strides).

    Notes
    -----
    This class provides two simple explicit integrators:
    - Euler (first-order, faster, less accurate/stable)
    - RK4 (fourth-order, more accurate, higher cost)

    Subclasses should ensure :meth:`rate_function` returns a non-negative rate when
    damage is expected to be monotonic. If your model permits healing (negative rate),
    consider whether and how you want to clip the state.

    Parameters
    ----------
    failure_damage : float, optional
        Damage threshold treated as "failure". When ``D >= failure_damage``,
        :meth:`step` returns the current state unchanged and :meth:`simulate` will
        plateau at that value. Defaults to 1.0.
    """

    def __init__(self, *, failure_damage=1.0):
        self.failure_damage = float(failure_damage)

    @cached_property
    def uct(self):
        """
        Ultimate tolerance (UCT) for the model.

        Returns a scalar "ultimate tolerance" (often interpreted as a force level) that
        corresponds to approximately one expected cycle to failure under the model.

        The exact meaning is model-specific and depends on how the subclass defines
        :meth:`_estimate_uct`. The value is cached after the first computation.

        Returns
        -------
        float
            Estimated ultimate tolerance for this model.
        """
        return self._estimate_uct()

    @abstractmethod
    def rate_function(self, state: float, force: float) -> float:
        """
        Instantaneous damage accumulation rate.

        Subclasses implement the damage law here.

        Parameters
        ----------
        state : float
            Current damage state ``D``.
        force : float
            Scalar load/force driving damage accumulation.

        Returns
        -------
        float
            Damage rate ``dD/dt`` (or per-cycle increment rate), consistent with the
            meaning of ``dt`` used in :meth:`step`.
        """
        raise NotImplementedError

    @abstractmethod
    def _estimate_uct(self) -> float:
        """
        Estimate the model's ultimate tolerance.

        This method is called by :attr:`uct` and should return a scalar force/load level
        (or analogous quantity) that corresponds to roughly one expected cycle to
        failure.

        Returns
        -------
        float
            Estimated ultimate tolerance.
        """
        raise NotImplementedError

    def step(
        self,
        d: float,
        force: float,
        *,
        dt: float = 1.0,
        method: str = "euler",
    ) -> float:
        """
        Advance the damage state by one integration step.

        Parameters
        ----------
        d : float
            Current damage state.
        force : float
            Scalar load/force applied during this step.
        dt : float, optional
            Step size (time increment or cycle duration). Defaults to 1.0.
        method : {"euler", "rk4"}, optional
            Explicit integration method. Defaults to "euler".

        Returns
        -------
        float
            Updated damage state ``d_next``. The result is clipped to
            ``failure_damage``. If ``d >= failure_damage``, ``d`` is returned unchanged.

        Raises
        ------
        ValueError
            If ``method`` is not one of {"euler", "rk4"}.
        """
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

        return min(float(d_next), self.failure_damage)

    def simulate(
        self,
        force_timeseries,
        initstate: float = 0.0,
        *,
        dt: float = 1.0,
        method: str = "euler",
    ):
        """
        Simulate cumulative damage over a force/load time series.

        Parameters
        ----------
        force_timeseries : array-like
            Sequence of scalar force/load values. Each entry is assumed to apply for one
            integration interval of length ``dt``.
        initstate : float, optional
            Initial damage state ``D0``. Defaults to 0.0.
        dt : float, optional
            Step size (time increment or cycle duration). Defaults to 1.0.
        method : {"euler", "rk4"}, optional
            Explicit integration method. Defaults to "euler".

        Returns
        -------
        numpy.ndarray
            Array of damage states with the same length as ``force_timeseries``.
            The output is clipped at ``failure_damage`` once failure is reached.
        """
        d = float(initstate)
        out = np.empty(len(force_timeseries), dtype=float)
        for i, f in enumerate(force_timeseries):
            d = self.step(d, float(f), dt=dt, method=method)
            out[i] = d
        return out
