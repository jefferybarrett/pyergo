from pyergo.damage_models import BarrettCallaghan, LiFFT
import numpy as np


def test_barrett_callaghan():
    model = BarrettCallaghan()

    applied_force = model.uct * np.ones(3)
    predicted = model.simulate(applied_force)
    expected = [0.0, 1.0, 1.0]


def test_lifft():
    pass


if __name__ == "__main__":
    test_barrett_callaghan()
