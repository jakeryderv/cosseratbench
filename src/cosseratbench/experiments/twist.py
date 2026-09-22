"""Rods under tension, clamped at both ends, each twisted and held just short of or
past the twist at which a straight rod buckles.

The twisted end may slide along the rod's axis, so the tension stays what the
load makes it: the tabletop version is a rod hanging from a clamp that turns,
with a weight on the bottom clamp. Twisting a straight rod stores torque without
changing its shape, until at a critical twist the straight shape becomes unstable.

Reference: Greenhill's problem, the loss of stability of a straight rod under a
twisting moment M and tension T with both ends clamped. Linearised, a sideways
displacement w = x + iy obeys B w'''' - i M w''' - T w'' = 0 with w = w' = 0 at
both ends, B the bending stiffness; the smallest M with a nonzero solution is
the critical moment. The critical twist follows from the torsional stiffness.
Without tension this is Greenhill's M L / B = 2.861 pi, and as the tension grows
it tends to the long-rod limit M^2 = 4 B T; the tests check both.

Measurement: several identical rods, side by side and independent, are each
twisted smoothly to a different fraction of the reference's critical twist and
held there. A rod held past its critical twist is unstable, and a small sideways
disturbance on it grows exponentially, faster the further past it is. For one
unstable mode under damping proportional to mass, the growth rate s satisfies
s^2 + c s = a (twist - critical) exactly, whatever the damping c, so fitting
that line to the rods that grow gives the twist where growth would start.

Why not twist one rod steadily until it buckles: a turning rod under damping is
pushed sideways by the damping itself, so it strays early, and near the threshold
it responds too slowly to follow a ramp; a ramp read off where the rod leaves
straight lands about 20% late. Held, neither happens.

The rods start very slightly bent to seed the disturbance; a solver with no
numerical noise would otherwise stay straight however unstable.
"""

from __future__ import annotations

import numpy as np

from cosseratbench.experiment import Experiment, Parameter
from cosseratbench.scenario import EndCondition, Material, Motion, PointLoad, Rod, Scenario
from cosseratbench.trajectory import Trajectory

LENGTH = 1.0
RADIUS = 0.01
MATERIAL = Material(youngs_modulus=1e6, shear_modulus=1e6 / 3.0, density=1000.0)
IMPERFECTION = 1e-4  # height of the initial bend, in rod lengths
LEVELS = (0.96, 0.98, 1.00, 1.02, 1.04, 1.06)  # held twists, as fractions of the reference's
SPACING = 0.25  # between neighbouring rods, in metres; they do not touch
RAMP = 6.0  # s to reach the held twist, easing in and out
SETTLE = 2.0  # s after the ramp before growth has settled to its exponential rate
DURATION = 16.0
GROWING = 0.05  # 1/s; a rod growing faster than this is past its critical twist
LINEAR = 1e-2  # m; beyond this sideways distance, growth is no longer exponential


def _clamped_determinant(m: float, tau: float) -> float:
    """A real function of the dimensionless moment m = M L / B, zero where a clamped rod
    under dimensionless tension tau = T L^2 / B can buckle."""
    root = np.sqrt(complex(4.0 * tau - m * m))
    l1, l2 = (1j * m + root) / 2.0, (1j * m - root) / 2.0
    e1, e2 = np.exp(l1), np.exp(l2)
    matrix = np.array(
        [[1, 0, 1, 1], [0, 1, l1, l2], [1, 1, e1, e2], [0, 1, l1 * e1, l2 * e2]], dtype=complex
    )
    # w = a + b z + c e^(l1 z) + d e^(l2 z); rows are w and w' at each end. Dividing out
    # l1 - l2 removes a spurious zero where the two exponentials coincide, and the
    # phase factor leaves a real function whose sign changes mark the roots.
    return float((np.linalg.det(matrix) / (l1 - l2) * np.exp(-(l1 + l2) / 2.0)).real)


def critical_moment(tau: float) -> float:
    """The dimensionless critical moment M L / B for tension tau = T L^2 / B > 0."""
    low = 2.0 * np.sqrt(tau)  # no rod, however long, buckles below this
    moments = np.linspace(low + 1e-9, low + 20.0, 4001)
    signs = np.sign([_clamped_determinant(m, tau) for m in moments])
    i = int(np.flatnonzero(signs[:-1] * signs[1:] < 0)[0])
    a, b = moments[i], moments[i + 1]
    sign_a = signs[i]
    for _ in range(60):
        middle = (a + b) / 2.0
        if np.sign(_clamped_determinant(middle, tau)) == sign_a:
            a = middle
        else:
            b = middle
    return (a + b) / 2.0


def critical_twist(rod: Rod, tau: float) -> float:
    """Twist, in radians, at which the straight rod buckles."""
    bending = rod.material.youngs_modulus * rod.second_moment_of_area
    torsion = rod.material.shear_modulus * 2.0 * rod.second_moment_of_area  # G J
    return critical_moment(tau) * bending / torsion


def _sideways(rod: Rod, positions: np.ndarray) -> np.ndarray:
    """Largest distance of any node from the line through the rod's clamps, per frame."""
    start, end = np.asarray(rod.centerline[0]), np.asarray(rod.centerline[-1])
    axis = (end - start) / np.linalg.norm(end - start)
    offsets = positions - start
    across = offsets - (offsets @ axis)[..., None] * axis
    return np.linalg.norm(across, axis=2).max(axis=1)


def growth_rates(scenario: Scenario, trajectory: Trajectory) -> np.ndarray:
    """Exponential growth rate of each rod's sideways distance while held, in 1/s;
    NaN where too few frames fall where growth is still exponential."""
    rates = []
    held = trajectory.times >= RAMP + SETTLE
    for rod, positions in zip(scenario.rods, trajectory.positions):
        sideways = _sideways(rod, positions)
        usable = held & (sideways > 0) & (sideways < LINEAR)
        if usable.sum() < 5:
            rates.append(np.nan)
            continue
        rates.append(np.polyfit(trajectory.times[usable], np.log(sideways[usable]), 1)[0])
    return np.array(rates)


def measured_critical_twist(scenario: Scenario, trajectory: Trajectory) -> float:
    """The twist, as a fraction of the reference's, at which growth would start;
    NaN unless at least three rods grew.

    Fitted to the three growing rods nearest the threshold: further out the line
    curves, and the fit drifts by about half a percent. That, and how it moves
    with the rods chosen, puts the measurement's own uncertainty near 0.5%.
    """
    rates = growth_rates(scenario, trajectory)
    levels = np.array(LEVELS)
    growing = np.flatnonzero(np.isfinite(rates) & (rates > GROWING))[:3]
    if len(growing) < 3:
        return float("nan")
    s, level = rates[growing], levels[growing]
    # s^2 = -c s + a level + b, so growth starts where s = 0: level = -b / a.
    basis = np.stack([-s, level, np.ones_like(s)], axis=1)
    (_, a, b), *_ = np.linalg.lstsq(basis, s**2, rcond=None)
    return float(-b / a)


def critical_twist_error(scenario: Scenario, trajectory: Trajectory) -> float:
    """Relative error of the twist at which the straight rod becomes unstable."""
    return abs(measured_critical_twist(scenario, trajectory) - 1.0)


def _rod(level: float, offset: float, tau: float) -> Rod:
    bending = MATERIAL.youngs_modulus * np.pi * RADIUS**4 / 4.0
    tension = tau * bending / LENGTH**2
    s = np.linspace(0.0, LENGTH, 201)
    bump = IMPERFECTION * LENGTH * (1.0 - np.cos(2.0 * np.pi * s / LENGTH)) / 2.0
    centerline = np.stack([s, offset + bump, np.zeros_like(s)], axis=1)
    straight = Rod(((0.0, 0.0, 0.0), (LENGTH, 0.0, 0.0)), RADIUS, MATERIAL)
    target = level * critical_twist(straight, tau)
    t = np.linspace(0.0, RAMP, 121)
    turn = target * (0.5 - 0.5 * np.cos(np.pi * t / RAMP))
    twist = Motion(
        times=tuple(t),
        displacement=tuple((0.0, 0.0, 0.0) for _ in t),
        rotation=tuple((float(a), 0.0, 0.0) for a in turn),
        slides_along=(1.0, 0.0, 0.0),
    )
    return Rod(
        centerline=tuple(map(tuple, centerline)),
        radius=RADIUS,
        material=MATERIAL,
        normal=(0.0, 0.0, 1.0),
        start=EndCondition.CLAMPED,
        end=EndCondition.CLAMPED,
        end_motion=twist,
        loads=(PointLoad(force=(tension, 0.0, 0.0)),),
    )


def build(tension: float) -> Scenario:
    return Scenario(
        rods=tuple(_rod(level, i * SPACING, tension) for i, level in enumerate(LEVELS)),
        duration=DURATION,
        quasi_static=True,
    )


twist = Experiment(
    name="twist",
    description="Rods under tension twisted past buckling, against Greenhill's critical twist.",
    build=build,
    parameters=(
        Parameter(
            "tension",
            default=10.0,
            values=(1.0, 10.0, 30.0),
            description=(
                "Dimensionless tension T L^2 / B. More tension needs more twist to "
                "buckle the rod, which the reference accounts for."
            ),
        ),
    ),
    metrics={"critical_twist_error": critical_twist_error},
    n_frames=321,
    notes=(
        "Six rods under tension are each twisted to a fraction of the twist at which "
        "a straight rod buckles, from 0.96 to 1.06, and held. Rods past the threshold "
        "bow out and grow; the ones below stay straight. The critical twist is fitted "
        "from how fast the growing rods grow. MuJoCo cannot run it: its clamps give "
        "way under this much twist."
    ),
)
