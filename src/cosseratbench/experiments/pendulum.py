"""A cable hanging from a pin, released from rest in its first mode and left to swing.

Reference: linear theory of a hanging cable, whose tension at each point is the
weight below it, with bending stiffness and stretch included. Its lowest mode has
no closed form once bending is included, so it is solved numerically, to far
better accuracy than any solver here reaches. Without bending or stretch it is
Bernoulli's hanging chain, which the tests check it against.

The cable starts stretched as it hangs in equilibrium. Released unstretched, it
would bounce along its length, and that bounce pumps sideways vibration near half
its frequency (parametric resonance) that the linear reference does not describe.

This is the first experiment where the motion itself is the result, so no solver
may add damping. A solver that loses or gains energy shows up as a change in
amplitude. The swing is small because the reference is linear: at 2 cm the real
period is longer than it predicts by 2e-5, which bounds how small a period error
can meaningfully be.
"""

from __future__ import annotations

import functools

import numpy as np

from cosseratbench.experiment import Experiment, Parameter
from cosseratbench.scenario import EndCondition, Material, Rod, Scenario
from cosseratbench.trajectory import Trajectory

LENGTH = 1.0
RADIUS = 0.002
DENSITY = 1000.0
GRAVITY = 9.81
# Of the free end. The reference is linear; the real swing slows with amplitude,
# by 0.051 A^2 (A in metres, measured with both solvers), so 2e-5 at this amplitude.
AMPLITUDE = 0.02


def static_strain(s: np.ndarray, length: float, material: Material, gravity: float) -> np.ndarray:
    """Axial strain of a cable hanging in equilibrium, at unstretched distance ``s`` from the pin."""
    return material.density * gravity * (length - s) / material.youngs_modulus


@functools.cache
def first_mode(
    length: float,
    radius: float,
    material: Material,
    gravity: float,
    n_elements: int = 200,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Lowest mode of small sideways vibration of a rod hanging in equilibrium,
    pinned at the top and free at the bottom.

    In unstretched height z above the free end, it solves
    (EI u'')'' - (T / (1 + T/EA) u')' = rho A omega^2 u, with T = rho A g z, using
    cubic Hermite finite elements. Returns the angular frequency, and the sideways
    displacement and its slope du/dz at ``n_elements + 1`` evenly spaced heights
    from the free end to the pin, scaled so the free end moves by one.
    """
    area = np.pi * radius**2
    bending = material.youngs_modulus * np.pi * radius**4 / 4.0  # EI
    axial = material.youngs_modulus * area  # EA
    mass = material.density * area  # per unstretched length
    h = length / n_elements

    # Three-point Gauss quadrature, exact for the polynomial integrands. Each node's
    # unknowns are its displacement and h times its slope, both in metres.
    points, weights = np.polynomial.legendre.leggauss(3)
    xi, weights = (points + 1.0) / 2.0, weights / 2.0
    shape = np.array(
        [1 - 3 * xi**2 + 2 * xi**3, xi - 2 * xi**2 + xi**3, 3 * xi**2 - 2 * xi**3, xi**3 - xi**2]
    )
    slope = (
        np.array(
            [6 * xi**2 - 6 * xi, 1 - 4 * xi + 3 * xi**2, 6 * xi - 6 * xi**2, 3 * xi**2 - 2 * xi]
        )
        / h
    )
    curvature = np.array([12 * xi - 6, 6 * xi - 4, 6 - 12 * xi, 6 * xi - 2]) / h**2
    element_bending = bending * h * (curvature * weights) @ curvature.T
    element_mass = mass * h * (shape * weights) @ shape.T

    n_dofs = 2 * (n_elements + 1)  # a displacement and a slope at each node
    stiffness = np.zeros((n_dofs, n_dofs))
    inertia = np.zeros((n_dofs, n_dofs))
    for e in range(n_elements):
        tension = mass * gravity * (e + xi) * h
        # Sideways, tension acts along the stretched cable, whose slope is u' / (1 + strain).
        effective = tension / (1.0 + tension / axial)
        dofs = slice(2 * e, 2 * e + 4)
        stiffness[dofs, dofs] += element_bending + h * (slope * weights * effective) @ slope.T
        inertia[dofs, dofs] += element_mass

    free = np.arange(n_dofs) != n_dofs - 2  # the pin holds the top node's displacement
    k, m = stiffness[np.ix_(free, free)], inertia[np.ix_(free, free)]
    # The slowest mode of (k, m) is the fastest of (m, k). Solved that way round it is
    # the largest eigenvalue, which a symmetric eigensolver finds to full relative
    # precision; the smallest would carry roundoff scaled by the stiffest bending mode.
    cholesky = np.linalg.cholesky(k)
    symmetric = np.linalg.solve(cholesky, np.linalg.solve(cholesky, m).T).T
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    mode = np.zeros(n_dofs)
    mode[free] = np.linalg.solve(cholesky.T, eigenvectors[:, -1])
    mode /= mode[0]
    return float(1.0 / np.sqrt(eigenvalues[-1])), mode[0::2], mode[1::2] / h


def reference_period(scenario: Scenario) -> float:
    rod = scenario.rods[0]
    gravity = float(np.linalg.norm(scenario.gravity))
    frequency, _, _ = first_mode(rod.length, rod.radius, rod.material, gravity)
    return 2.0 * np.pi / frequency


def _material(youngs_modulus: float) -> Material:
    return Material(youngs_modulus, youngs_modulus / 3.0, DENSITY)


def _initial_state(material: Material) -> tuple[np.ndarray, np.ndarray]:
    """The cable hanging from the origin in equilibrium, displaced into its first mode:
    the positions of material points, and their unstretched distances from the pin."""
    _, displacement, slope = first_mode(LENGTH, RADIUS, material, GRAVITY)
    s = np.linspace(0.0, LENGTH, len(displacement))
    strain = static_strain(s, LENGTH, material, GRAVITY)
    # The mode runs from the free end up; the rod runs from the pin down, so d/ds = -d/dz.
    x = AMPLITUDE * displacement[::-1]
    dx_ds = -AMPLITUDE * slope[::-1]
    # Stretched, each unstretched ds spans (1 + strain) ds, sideways and down together.
    drop = np.sqrt((1.0 + strain) ** 2 - dx_ds**2)
    z = -np.concatenate(([0.0], np.cumsum((drop[1:] + drop[:-1]) / 2.0 * np.diff(s))))
    return np.stack([x, np.zeros_like(x), z], axis=1), s


def _fit(t: np.ndarray, x: np.ndarray, omega: float) -> tuple[float, float]:
    """Least-squares amplitude of x(t) = a cos(omega t) + b sin(omega t) + c, and its residual."""
    basis = np.stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)], axis=1)
    coefficients, *_ = np.linalg.lstsq(basis, x, rcond=None)
    residual = x - basis @ coefficients
    return float(np.hypot(coefficients[0], coefficients[1])), float(residual @ residual)


def measured_period(scenario: Scenario, trajectory: Trajectory) -> float:
    """Period of the single sinusoid that best fits the free end's sideways motion.

    A fit uses every frame, so a small fast vibration riding on the swing averages
    out, where it would shift the individual times the swing crosses the vertical.
    """
    t, x = trajectory.times, trajectory.positions[0][:, -1, 0]
    reference = reference_period(scenario)

    def cost(period: float) -> float:
        return _fit(t, x, 2.0 * np.pi / period)[1]

    # A coarse scan finds the right minimum; each refinement narrows around it.
    periods = reference * np.geomspace(0.5, 2.0, 301)
    for _ in range(6):
        best = int(np.argmin([cost(p) for p in periods]))
        low, high = periods[max(best - 1, 0)], periods[min(best + 1, len(periods) - 1)]
        periods = np.linspace(low, high, 41)
    return float(periods[int(np.argmin([cost(p) for p in periods]))])


def period_error(scenario: Scenario, trajectory: Trajectory) -> float:
    """Relative error of the swing period."""
    reference = reference_period(scenario)
    return abs(measured_period(scenario, trajectory) - reference) / reference


def amplitude_change(scenario: Scenario, trajectory: Trajectory) -> float:
    """Relative change in the free end's swing amplitude from the first period to the
    last. Negative means the swing lost energy; the physics says zero."""
    period = measured_period(scenario, trajectory)
    t, x = trajectory.times, trajectory.positions[0][:, -1, 0]
    first = t <= t[0] + period
    last = t >= t[-1] - period
    omega = 2.0 * np.pi / period
    return _fit(t[last], x[last], omega)[0] / _fit(t[first], x[first], omega)[0] - 1.0


def build(youngs_modulus: float) -> Scenario:
    material = _material(youngs_modulus)
    centerline, rest_arc_length = _initial_state(material)
    return Scenario(
        rods=(
            Rod(
                centerline=tuple(map(tuple, centerline)),
                rest_arc_length=tuple(rest_arc_length),
                radius=RADIUS,
                material=material,
                normal=(0.0, 1.0, 0.0),
                start=EndCondition.PINNED,
            ),
        ),
        duration=5.0,
        gravity=(0.0, 0.0, -GRAVITY),
        self_contact=False,  # a nearly straight cable swinging 2 cm cannot reach itself
    )


pendulum = Experiment(
    name="pendulum",
    description="Cable hanging from a pin, swinging in its first mode, against linear vibration theory.",
    build=build,
    parameters=(
        Parameter(
            "youngs_modulus",
            default=1e7,
            values=(1e6, 1e7, 1e8),
            unit="Pa",
            description=(
                "Stiffer cables swing faster as bending takes a larger share, and cost "
                "more to simulate; the reference accounts for both bending and stretch."
            ),
        ),
    ),
    metrics={"period_error": period_error, "amplitude_change": amplitude_change},
    notes=(
        "A cable hanging from a pin swings in its first mode with no damping, so the "
        "motion itself is the result. Watch the period against theory and whether the "
        "swing keeps its size: a solver that loses or gains energy shows it as a "
        "change in amplitude."
    ),
)
