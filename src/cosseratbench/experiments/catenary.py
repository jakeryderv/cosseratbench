"""A slender cable pinned at both ends, sagging under gravity.

Reference: the elastic catenary, the closed-form equilibrium of a perfectly
flexible cable that stretches under its own tension. Stretch deepens the sag of
this cable by about 1%, so it cannot be neglected; bending stiffness, which the
reference does neglect, matters far less because pinned ends carry no moment.
"""

from __future__ import annotations

import numpy as np

from cosseratbench.experiment import Experiment
from cosseratbench.metrics import max_distance_to_curve, settling_residual
from cosseratbench.scenario import EndCondition, Material, Rod, Scenario
from cosseratbench.trajectory import Trajectory

LENGTH = 1.0
SPAN = 0.8


def _bisect(f, lo: float, hi: float) -> float:
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def _circular_arc(length: float, span: float, n_points: int = 401) -> np.ndarray:
    """An arc of the given length hanging below a chord of the given span: a smooth
    starting shape that is deliberately not the answer."""
    half_angle = _bisect(lambda phi: np.sin(phi) / phi - span / length, 1e-9, np.pi)
    radius = length / (2.0 * half_angle)
    alpha = np.linspace(-half_angle, half_angle, n_points)
    x = span / 2.0 + radius * np.sin(alpha)
    z = radius * (np.cos(half_angle) - np.cos(alpha))
    return np.stack([x, np.zeros_like(x), z], axis=1)


def reference_curve(
    rod: Rod, gravity: float, span: float = SPAN, n_points: int = 2001
) -> np.ndarray:
    """Elastic catenary between supports at the same height, parametrised by unstretched arc length."""
    length = rod.length
    stiffness = rod.material.youngs_modulus * rod.area  # EA
    weight = rod.material.density * rod.area * gravity  # per unstretched length
    support = weight * length / 2.0  # vertical reaction at each end
    s = np.linspace(0.0, length, n_points)

    def x(tension: float) -> np.ndarray:  # tension: the constant horizontal component
        hang = np.arcsinh((weight * s - support) / tension) + np.arcsinh(support / tension)
        return tension * s / stiffness + tension / weight * hang

    tension = _bisect(lambda h: x(h)[-1] - span, 1e-6 * weight * length, 1e6 * weight * length)
    z = (weight * s**2 / 2.0 - support * s) / stiffness + (
        np.hypot(tension, weight * s - support) - np.hypot(tension, support)
    ) / weight
    return np.stack([x(tension), np.zeros_like(s), z], axis=1)


def _reference(scenario: Scenario) -> np.ndarray:
    return reference_curve(scenario.rods[0], float(np.linalg.norm(scenario.gravity)))


def shape_error(scenario: Scenario, trajectory: Trajectory) -> float:
    """Largest distance from a final node to the reference catenary, in rod lengths."""
    return max_distance_to_curve(trajectory.positions[0][-1], _reference(scenario)) / LENGTH


def sag_error(scenario: Scenario, trajectory: Trajectory) -> float:
    """Relative error of the lowest point's depth."""
    reference = _reference(scenario)[:, 2].min()
    return float(abs(trajectory.positions[0][-1][:, 2].min() - reference) / abs(reference))


catenary = Experiment(
    name="catenary",
    description="Cable pinned at both ends sagging under gravity, against the analytical catenary.",
    scenario=Scenario(
        rods=(
            Rod(
                centerline=tuple(map(tuple, _circular_arc(LENGTH, SPAN))),
                radius=0.005,
                material=Material(youngs_modulus=1e6, shear_modulus=1e6 / 3.0, density=1000.0),
                normal=(0.0, 1.0, 0.0),
                start=EndCondition.PINNED,
                end=EndCondition.PINNED,
            ),
        ),
        duration=5.0,
        gravity=(0.0, 0.0, -9.81),
        quasi_static=True,
    ),
    reference=_reference,
    metrics={
        "shape_error": shape_error,
        "sag_error": sag_error,
        "settling_residual": settling_residual,
    },
)
