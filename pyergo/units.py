# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 02:27:17 2024

@author: jeffm
"""


class Units:
    _instance = None

    def __init__(self):
        if self._instance is None:
            self._instance = super().__init__()
        else:
            return self._instance

    # Time
    s = 1.0
    minute = 60 * s
    hour = 60 * minute
    day = 24 * hour

    # Force
    N = 1.0
    kN = 1e3 * N
    MN = 1e6 * N
    GN = 1e9 * N
    mN = 1e-3 * N
    uN = 1e-6 * N
    nN = 1e-9 * N

    # Distance / Length
    m = 1.0
    km = 1e3 * m
    cm = 1e-2 * m
    mm = 1e-3 * m
    um = 1e-6 * m
    nm = 1e-9 * m
    inch = 0.0254 * m
    ft = 0.3048 * m


    # SI Volume equivalents
    m3 = m*m*m                 # cubic metre
    cm3 = 1e-6 * m3               # cubic centimetre (same as 1 mL)
    mm3 = 1e-9 * m3               # cubic millimetre

    # Volumes
    L = 1e-3 * m3               # litre
    mL = cm3                    # millilitre
    uL = 1e-6 * L                # microlitre
    nL = 1e-9 * L                # nanolitre

    # US volumes (optional)
    gallon = 3.78541 * L
    quart = 0.946353 * L
    pint = 0.473176 * L
    cup = 0.24 * L
    tablespoon = 0.0147868 * L
    teaspoon = 0.00492892 * L

    # Mass
    kg = 1.0
    g = 1e-3 * kg
    mg = 1e-6 * kg
    ug = 1e-9 * kg
    tonne = 1e3 * kg
    lb = 0.453592 * kg

    # Molarity
    mol = 1.0
    mmol = 1e-3 * mol
    umol = 1e-6 * mol
    nmol = 1e-9 * mol
    mol_per_L = mol / L  # mol / L = mol / 0.001 m³

    # Energy
    J = N * m
    kJ = 1e3 * J
    MJ = 1e6 * J
    cal = 4.184 * J
    kcal = 1e3 * cal

    # Power
    W = J / s
    kW = 1e3 * W
    MW = 1e6 * W

    # Pressure
    Pa = N / m**2
    kPa = 1e3 * Pa
    MPa = 1e6 * Pa
    atm = 101325 * Pa
    bar = 1e5 * Pa
    mmHg = 133.322 * Pa
    psi = 6894.76 * Pa

    # Cumulative load
    MNs = MN * s
    GNs = GN * s
