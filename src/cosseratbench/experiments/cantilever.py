"""A clamped beam with a small transverse load at its tip.

Reference: Euler-Bernoulli tip deflection, F L^3 / (3 E I). The load is small
enough (deflection ~2% of length) and the beam slender enough (r/L = 0.02) that
geometric nonlinearity and shear each change the answer by under 0.1%.
"""

from __future__ import annotations

import numpy as np

from cosseratbench.experiment import Experiment
from cosseratbench.metrics import settling_residual
from cosseratbench.scenario import EndCondition, Material, PointLoad, Rod, Scenario
from cosseratbench.trajectory import Trajectory

LENGTH = 1.0
FORCE = 7.5e-3


def reference_tip_deflection(rod: Rod) -> float:
    return FORCE * rod.length**3 / (3.0 * rod.material.youngs_modulus * rod.second_moment_of_area)


def reference_curve(scenario: Scenario, n_points: int = 201) -> np.ndarray:
    rod = scenario.rods[0]
    stiffness = rod.material.youngs_modulus * rod.second_moment_of_area  # EI
    x = np.linspace(0.0, rod.length, n_points)
    z = -FORCE * x**2 * (3.0 * rod.length - x) / (6.0 * stiffness)
    return np.stack([x, np.zeros_like(x), z], axis=1)


def tip_deflection_error(scenario: Scenario, trajectory: Trajectory) -> float:
    """Relative error of the final tip deflection."""
    reference = reference_tip_deflection(scenario.rods[0])
    deflection = -trajectory.positions[0][-1, -1, 2]
    return float(abs(deflection - reference) / reference)


def build() -> Scenario:
    return Scenario(
        rods=(
            Rod(
                centerline=((0.0, 0.0, 0.0), (LENGTH, 0.0, 0.0)),
                radius=0.02,
                material=Material(youngs_modulus=1e6, shear_modulus=1e6 / 3.0, density=1000.0),
                start=EndCondition.CLAMPED,
                loads=(PointLoad(force=(0.0, 0.0, -FORCE)),),
            ),
        ),
        duration=10.0,
        quasi_static=True,
    )


cantilever = Experiment(
    name="cantilever",
    description="Clamped beam under a small tip load, against Euler-Bernoulli beam theory.",
    build=build,
    reference=reference_curve,
    metrics={
        "tip_deflection_error": tip_deflection_error,
        "settling_residual": settling_residual,
    },
    n_elements=20,
    notes=(
        "A beam clamped at one end bends under a small weight at the other. It "
        "isolates bending stiffness. Both solvers hold the clamped element rigid, so "
        "their error halves only as the element count doubles: vary the resolution "
        "to see it."
    ),
)
