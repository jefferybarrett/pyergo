# pyergo

`pyergo` is a lightweight Python package for ergonomics research, with an emphasis on
**mechanistic models of cumulative damage arising from repeated or sustained loading**.

The package is designed to support:
- reproducible fatigue and injury-risk modelling,
- transparent links between analytic theory and numerical simulation,
- ergonomic and biomechanical research workflows where clarity and extensibility matter.

At present, `pyergo` focuses on cumulative damage models derived from the fatigue and mechanobiology literature. The longer-term vision is a broader ecosystem of ergonomics and exposure-modelling tools built on the same principles.

---

## Installation

You can install from `pip` simply with
```bash
pip install pyergo
```
(or, equivalently, if you use `uv`):

```bash
uv venv .venv --python 3.12
uv add pyergo
```

Alternatively you can install a version of it in development mode right from source.

```bash
git clone https://github.com/jefferybarrett/pyergo.git
cd pyergo
pip install -e .
```

---

## A Primer on Cumulative Damage Models

Many ergonomic risk metrics are based on **cumulative load**, such as total force exposure
or integrated joint moments. While useful, cumulative load is *not* the same as cumulative
damage.

Cumulative damage models instead describe how tissue integrity evolves over time as a
function of applied load, often via a differential equation of the form:

$$
\frac{dD}{dN} = f(D, F)
$$

where:
- $D \in [0,1]$ is cumulative damage (with $D=1$ indicating failure),
- $F$ is the applied force or load,
- $N$ is the number of cycles or repetitions.

Such models naturally capture:
- nonlinear load sensitivity,
- history dependence,
- threshold-like behavior near failure,
- differences between short-term tolerance and long-term fatigue.

---

## Implemented Damage Models

All damage models inherit from a common base class:

```python
from pyergo.damage_models.base import DamageModel
```

### Miner–Palmgren

```python
from pyergo.damage_models import MinerPalmgren
```

A classic linear damage accumulation rule, often written as:

$$
\frac{dD}{dN} = \frac{1}{N^*(F)}
$$

where $N^*(F)$ is an empirical S–N curve.

This model is simple and interpretable, but does not capture nonlinear degradation
or load–damage interactions.

---

### Barrett–Callaghan Model

```python
from pyergo.damage_models import BarrettCallaghan
```

This model implements a nonlinear cumulative damage law derived from a
Tobolsky–Eyring–type rate process:

$$
\frac{dQ}{dN} = -A Q \exp\left(\frac{B F}{Q}\right),
\quad D = 1 - Q
$$

Key features:
- analytic S–N curve via the exponential integral,
- well-defined ultimate and instantaneous tolerances,
- threshold-like behavior near catastrophic failure,
- clear distinction between cumulative load and cumulative damage.

This model is described and analyzed in detail in the associated paper (see below).

---

### LiFFT

```python
from pyergo.damage_models import LiFFT
```

An implementation of the LiFFT cumulative damage model originally introduced by
Gallagher et al. (2017). In this framework, damage accumulates exponentially with load
relative to an ultimate tolerance.

The LiFFT model can be interpreted as a special case of a Miner–Palmgren rule with
an exponential S–N curve, and is included here for comparison and compatibility with
existing ergonomic tools.

---

## Simulation Interface

All models support a common simulation interface:

```python
damage = model.simulate(
    force_timeseries,
    initstate=0.0,
    dt=1.0,
    method="rk4",
)
```

Features:
- Euler or RK4 time integration,
- automatic clipping at failure (`D = 1`),
- reproducible and testable numerical behavior.

For constant loading, analytic expressions for cycles to failure are also available
where applicable.

---

## Units System

`pyergo` includes a minimal, lightweight units system designed for numerical ergonomics
workflows.

```python
from pyergo import Units, UnitsError
```

Example usage:

```python
force = 2.0 * Units.kN
time  = 5.0 * Units.minute
speed = 3.0 * Units.m / Units.s
```

Key design goals:
- quantities evaluate to plain floats in the current base unit system,
- base units (length, time, mass) can be changed globally,
- no runtime overhead from tracking units during numerical computation.

This approach keeps simulations fast while preserving clarity at model boundaries.

---

## Citation

If you use the Barrett–Callaghan damage model or related analyses, please cite:

```latex
@article{BarrettCallaghan2026,
  author  = {Barrett, Jeff M. and Callaghan, Jack P.},
  title   = {From cumulative exposure to failure: a unifying modelling framework for nonlinear tissue fatigue in ergonomics},
  journal = {Safety Science},
  volume  = {196},
  year    = {2026},
  pages   = {107099},
  doi     = {10.1016/j.ssci.2025.107099},
  url     = {https://doi.org/10.1016/j.ssci.2025.107099},
}
```

Similarly, for LiFFT the citation that describes this method is Gallager et al. (2017), whose reference is:

```latex
@article{Gallagher2017LiFFT,
  author  = {Gallagher, Sean and Sesek, Richard F. and Schall Jr., Michael C. and Huangfu, Rui},
  title   = {Development and validation of an easy-to-use risk assessment tool for cumulative low back loading: The Lifting Fatigue Failure Tool (LiFFT)},
  journal = {Applied Ergonomics},
  volume  = {63},
  year    = {2017},
  pages   = {142--150},
  doi     = {10.1016/j.apergo.2017.04.016},
  url     = {https://doi.org/10.1016/j.apergo.2017.04.016},
}
```

---

## Scope and Philosophy

`pyergo` is intentionally:
- **mechanistic**, not purely empirical,
- **explicit**, not heuristic-heavy,
- **numerically conservative**, favoring clarity over cleverness.

The goal is not to replace existing ergonomic tools, but to provide a transparent
foundation for research, comparison, and theory-driven model development.
