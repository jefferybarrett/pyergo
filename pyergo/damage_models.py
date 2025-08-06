"""
DamageModels

This is a collection of objects that are
used to translate between cumulative load
and cumulative damage.

Implemented models are:
 - MinerPalmgren
 - BarrettCallaghan
 - LiFFTrepo
"""

import numpy as np
from scipy.special import exp1
from scipy.optimize import newton
import pyergo.Units as Units
from functools import cached_property
from abc import ABC


class DamageModel(ABC):
    """A DamageModel is a class that houses
    one of the cumulative damage models in
    the literature
    """

    def __init__(self):
        pass

    @cached_property
    def uct(self):
        return self._estimate_uct()

    def rate_function(self, state, force):
        return 0.0

    def _estimate_uct(self):
        raise NotImplementedError

    def simulate(self, force_timeseries, initstate=0.0):
        D = [initstate]  # the damage that will be accumulated
        for force in force_timeseries:
            D += [D[-1] + self.rate_function(D[-1], force)]
        return np.array(D[1:])


class MinerPalmgren(DamageModel):

    def __init__(self, SN_curve):
        self.SN_curve = SN_curve

    def rate_function(self, state, force):
        D = state
        return 1 / self.SN_curve(force)

    def _estimate_uct(self):
        """Gives an estimate for the UCT
        given the provided SN_curve
        """
        func = lambda force: self.SN_curve(force) - 1.0
        return newton(func, 1.0 * Units.kN)


class BarrettCallaghan(DamageModel):
    """
    BarrettCallaghan implements the damage model in
    an upcoming manuscript.

    Here we will assume that t is measured in units
    of lift-durations, for which we have calibrated A and B
    values.
    """

    def __init__(self, A=2.47e-11, B=0.00203):
        self.A, self.B = A, B

    def rate_function(self, state, force):
        D = state
        if D <= 1.0:
            return self.A * (1 - D) * np.exp(self.B * force / (1 - D))
        else:
            return 0.0

    def cycles_to_failure(self, applied_force):
        """This is the "time" to failure of the damage model
        Noting that we are using 'time' measured in units of
        lift duration; so it is equivalent to the number of cycles.
        """
        return (1 / self.A) * exp1(self.B * applied_force)

    def simulate(self, forcetimeseries, initstate=0.0):
        D = super().simulate(forcetimeseries, initstate)
        return np.clip(D, 0.0, 1.0)

    def _estimate_uct(self):
        """The ultimate tolerance is the
        load that leads to an estimated time
        to failure being less than 1.0
        (i.e. it tolerates less than one
              cycle)
        """
        func = lambda f: (self.cycles_to_failure(f) - 1.0)
        return newton(func, 1.0 * Units.kN)


class LiFFT(DamageModel):
    """This implements the LiFFT cumulative
    damage model described in Gallagher et al. (2017)

    In this model, the damage

    Example:
    from pyergo.DamageModels import LiFFT
    import pyergo.Units as Units
    import matplotlib.pyplot as plt

    # let's simulate 50 cycles at 5 kilonewtons!
    repetitions = np.arange(50) + 1
    myForce = (5*Units.kN) * np.linspace(0, 1, 50)

    # now let's initialize the model
    myModel = LiFFT() # loads with default parameters

    # Simulate using the force
    damage = myModel.simulate(myForce)

    # plot the result
    plt.plot(repetitions, damage, label = "LiFFT")
    plt.legend()
    plt.show()

    Remarks:
    (1) This is the same as:
        A = some param
        B = some param
        test = MinerPalmgren(SN_curve = lambda force: (1/A)*np.exp(-B*force))
    """

    def __init__(self, A=1 / 902416.0, B=0.162, ultimate_tolerance=10 * Units.kN):
        self.A, self.B = A, B
        self.ultimate_tolerance = ultimate_tolerance

    def rate_function(self, state, force):
        D = state
        return self.A * np.exp(self.B * 100.0 * force / self.ultimate_tolerance)

    def cycles_to_failure(self, force):
        return 1.0 / self.rate_function(0.0, force)

    def _estimate_uct(self):
        func = lambda f: (self.cycles_to_failure(f) - 1.0)
        return newton(func, 1.0 * Units.kN)
