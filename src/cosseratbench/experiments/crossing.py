"""Two ropes crossing at right angles, the upper one resting on the lower one.

Each rope is pinned at both ends and sags under its own weight. The upper rope's
supports sit lower than the lower rope's, so on the way down it meets the other
rope and stops: it is held up in the middle, and it presses the rope below it
down. The contact is frictionless, and between two rods crossing at a right angle
it is a single point, with the force straight up.

This is the first experiment where rods touch each other rather than an obstacle,
so the question is whether a solver carries the right force through that point.
The answer is the pair of shapes in which each rope is in equilibrium under its
own weight and one contact force, and the two surfaces just touch.

Reference. A rope with a point load at its middle cannot be treated as perfectly
flexible: the ideal kink is rounded off by bending over a length sqrt(EI/T),
about 3 cm here, which lifts the middle by 6 mm -- more than the ropes' radius --
and so changes the contact force outright. The reference therefore solves the planar
elastica -- bending, stretch and the weight of the rope -- as a boundary value
problem, and finds the contact force that leaves exactly one diameter between the
two centrelines. It neglects shear, worth about 1e-4 rod lengths here, which is
the accuracy the experiment can resolve.

The ropes start held a centimetre further apart than they finish and settle down
into contact, so a solver has to find the contact force rather than be handed it.
They cannot start already touching: at the resolutions solvers run, a rope's
polyline cuts the corner where the load presses on it, so ropes one diameter
apart at the crossing have segments that already overlap.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from cosseratbench.experiment import Experiment, Parameter
from cosseratbench.metrics import max_distance_to_curve, settling_residual
from cosseratbench.scenario import EndCondition, Material, Rod, Scenario
from cosseratbench.solver import Capability
from cosseratbench.trajectory import Trajectory

LENGTH = 1.0  # m of rope, each
SPAN = 0.8  # m between each rope's pins
RADIUS = 0.005  # m
DENSITY = 1000.0  # kg/m^3
GRAVITY = 9.81
DURATION = 8.0  # s; the ropes have settled by about 3 s
CLEARANCE = 0.01  # m further apart at the start than they finish
_TOLERANCE = 1e-9  # on the boundary value solve; it resolves the shapes to about 1e-9 m


def _properties(youngs_modulus: float) -> tuple[float, float, float]:
    """Weight per unstretched metre, axial stiffness EA, bending stiffness EI."""
    area = np.pi * RADIUS**2
    return DENSITY * area * GRAVITY, youngs_modulus * area, youngs_modulus * area * RADIUS**2 / 4


def _bisect(f, lo: float, hi: float, steps: int = 200) -> float:
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def flexible(youngs_modulus: float, load: float, points: int = 2001) -> tuple[float, np.ndarray]:
    """Half a perfectly flexible rope, support to middle: its horizontal tension, and
    the curve as columns (unstretched arc length, along, up).

    An elastic catenary carrying a point ``load`` downward at its middle, which it
    takes as a sharp kink.
    """
    weight, axial, _ = _properties(youngs_modulus)
    reaction = (weight * LENGTH + load) / 2.0
    s = np.linspace(0.0, LENGTH / 2.0, points)

    def along(tension: float) -> np.ndarray:
        hang = np.arcsinh((weight * s - reaction) / tension) + np.arcsinh(reaction / tension)
        return tension * s / axial + tension / weight * hang

    tension = _bisect(
        lambda t: along(t)[-1] - SPAN / 2.0, 1e-9 * weight * LENGTH, 1e9 * weight * LENGTH
    )
    up = (weight * s**2 / 2.0 - reaction * s) / axial + (
        np.hypot(tension, weight * s - reaction) - np.hypot(tension, reaction)
    ) / weight
    return tension, np.stack([s, along(tension), up], axis=1)


@functools.lru_cache(maxsize=32)
def _elastica(youngs_modulus: float, load: float) -> tuple[np.ndarray, np.ndarray]:
    """Half a real rope, support to middle: unstretched arc length, and the curve as
    columns (along, up).

    The state is (along, up, angle, bending moment) against unstretched arc length,
    with the constant horizontal tension as an unknown parameter. The rope is pinned,
    so it carries no moment at its support, and level at its middle by symmetry.

    The load is walked up from nothing, each step started from the last. That is not
    only for convergence: a rope held at its middle has several equilibrium shapes,
    and this picks the one the rope reaches as the load comes on.
    """
    weight, axial, bending = _properties(youngs_modulus)

    def system(load: float):
        reaction = (weight * LENGTH + load) / 2.0

        def rates(s: np.ndarray, state: np.ndarray, parameters: np.ndarray) -> np.ndarray:
            tension = parameters[0]
            angle, moment = state[2], state[3]
            across = weight * s - reaction  # the vertical force the rope carries at s
            cos, sin = np.cos(angle), np.sin(angle)
            stretch = 1.0 + (tension * cos + across * sin) / axial
            return np.vstack(
                [
                    stretch * cos,
                    stretch * sin,
                    moment / bending,
                    # The moment balance, dM/ds = -(dr/ds x n): a rope pulled down at
                    # its middle bends down there. With the sign the other way the rod
                    # would ripple around the flexible answer instead of settling onto
                    # it over a boundary layer.
                    -stretch * (across * cos - tension * sin),
                ]
            )

        def ends(start: np.ndarray, middle: np.ndarray, _: np.ndarray) -> np.ndarray:
            return np.array([start[0], start[1], start[3], middle[2], middle[0] - SPAN / 2.0])

        return rates, ends

    mesh = np.linspace(0.0, LENGTH / 2.0, 600)
    tension, hanging = flexible(youngs_modulus, 0.0)
    angle = np.arctan2(weight * mesh - weight * LENGTH / 2.0, tension)
    guess = np.vstack(
        [
            np.interp(mesh, hanging[:, 0], hanging[:, 1]),
            np.interp(mesh, hanging[:, 0], hanging[:, 2]),
            angle,
            np.zeros_like(mesh),
        ]
    )

    def attempt(at: float):
        rates, ends = system(at)
        return solve_bvp(rates, ends, mesh, guess, p=[tension], tol=_TOLERANCE, max_nodes=20000)

    answer = attempt(0.0)  # the rope hanging free, which the guess is already close to
    reached, step = 0.0, load
    while abs(reached - load) > 1e-12:
        guess, tension = answer.sol(mesh), answer.p[0]
        tried = attempt(reached + step)
        if not tried.success:
            step /= 2.0  # too far for one Newton solve; creep up on it instead
            if abs(step) < abs(load) * 1e-6:
                raise RuntimeError(f"no equilibrium found under a {load} N load")
            continue
        answer, reached = tried, reached + step
        step = np.clip(1.5 * step, -abs(load - reached), abs(load - reached))
    s = np.linspace(0.0, LENGTH / 2.0, 1001)
    return s, answer.sol(s)[:2].T


def _mid_height(youngs_modulus: float, load: float) -> float:
    """Height of a rope's middle above its supports, under a downward point load."""
    return float(_elastica(youngs_modulus, load)[1][-1, 1])


def _contact_force(youngs_modulus: float, drop: float, mid, apart: float = 2 * RADIUS) -> float:
    """The force that would hold the ropes' centrelines ``apart`` at the crossing.

    At one diameter apart this is the force they really press on each other with. The
    lower rope is pushed down by it and the upper rope held up, so the gap grows with
    the force and there is one answer.
    """

    def gap(force: float) -> float:
        return -drop + mid(youngs_modulus, -force) - mid(youngs_modulus, force) - apart

    upper = max(drop, 0.1) * _properties(youngs_modulus)[0]
    while gap(upper) < 0.0:
        upper *= 2.0
        if upper > 1e4:
            raise RuntimeError("the ropes cannot be pressed apart; check the geometry")
    return float(brentq(gap, 0.0, upper, xtol=1e-10, rtol=1e-12))


def _mirrored(arc: np.ndarray, along: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A whole rope from half of one: arc length from 0 to LENGTH, and (along, up) with
    ``along`` measured from the left support."""
    whole_arc = np.concatenate([arc, LENGTH - arc[-2::-1]])
    whole_along = np.concatenate([along, SPAN - along[-2::-1]])
    return whole_arc, np.stack([whole_along, np.concatenate([up, up[-2::-1]])], axis=1)


def _shapes(
    youngs_modulus: float, drop: float, apart: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unstretched arc length along each rope, and both centrelines in 3D, lower one
    first, held ``apart`` where they cross. The lower rope runs along x, the upper
    along y."""
    force = _contact_force(youngs_modulus, drop, _mid_height, apart)
    planes, arc = [], None
    for load, height in ((force, 0.0), (-force, -drop)):
        half_arc, curve = _elastica(youngs_modulus, load)
        arc, plane = _mirrored(half_arc, curve[:, 0], curve[:, 1])
        planes.append((plane[:, 0] - SPAN / 2.0, np.zeros_like(plane[:, 0]), plane[:, 1] + height))
    lower = np.stack(planes[0], axis=1)
    upper = np.stack([planes[1][1], planes[1][0], planes[1][2]], axis=1)
    return arc, lower, upper


def _parameters(scenario: Scenario) -> tuple[float, float]:
    return scenario.rods[0].material.youngs_modulus, -scenario.rods[1].centerline[0][2]


def reference_curves(scenario: Scenario) -> tuple[np.ndarray, ...]:
    return _shapes(*_parameters(scenario), 2 * RADIUS)[1:]


def build(youngs_modulus: float, drop: float) -> Scenario:
    material = Material(youngs_modulus, youngs_modulus / 3.0, DENSITY)
    # The ropes start held a centimetre further apart than they finish, and settle down
    # into contact. Starting them already touching does not work: at the resolutions
    # solvers run, a rope's polyline cuts the corner where the load presses on it, so
    # ropes one diameter apart at the crossing have segments that already overlap, and
    # the run would open with a contact impulse nobody asked for.
    arc, lower, upper = _shapes(youngs_modulus, drop, 2 * RADIUS + CLEARANCE)
    keep = np.linspace(0, len(arc) - 1, 401).round().astype(int)
    rods = [
        Rod(
            centerline=tuple(map(tuple, points[keep])),
            rest_arc_length=tuple(arc[keep]),
            radius=RADIUS,
            material=material,
            normal=normal,
            start=EndCondition.PINNED,
            end=EndCondition.PINNED,
        )
        for points, normal in ((lower, (0.0, 1.0, 0.0)), (upper, (1.0, 0.0, 0.0)))
    ]
    return Scenario(
        rods=tuple(rods),
        duration=DURATION,
        gravity=(0.0, 0.0, -GRAVITY),
        quasi_static=True,
        self_contact=False,  # neither rope doubles back far enough to reach itself
    )


def shape_error(scenario: Scenario, trajectory: Trajectory) -> float:
    """Largest distance from a final node to its rope's reference shape, in rod lengths,
    over both ropes."""
    curves = reference_curves(scenario)
    return max(
        max_distance_to_curve(positions[-1], curve) / LENGTH
        for positions, curve in zip(trajectory.positions, curves)
    )


def lift_error(scenario: Scenario, trajectory: Trajectory) -> float:
    """Relative error of how far the lower rope holds the upper one up.

    The upper rope's middle, against where it would hang with nothing under it. This
    is the contact force in disguise: a solver that carries too little force through
    the touching point lets the upper rope hang too low.
    """
    youngs_modulus, drop = _parameters(scenario)
    upper = reference_curves(scenario)[1]
    free = _mid_height(youngs_modulus, 0.0)
    reference = float(upper[len(upper) // 2][2]) + drop
    nodes = trajectory.positions[1][-1]
    measured = float(nodes[len(nodes) // 2][2]) + drop
    return float(abs(measured - reference) / abs(reference - free))


crossing = Experiment(
    name="crossing",
    description="Two ropes crossing at right angles, the upper resting on the lower, against the elastica.",
    build=build,
    parameters=(
        Parameter(
            "drop",
            default=0.1,
            values=(0.05, 0.1, 0.2),
            unit="m",
            description=(
                "How far the upper rope's pins sit below the lower rope's: further down "
                "means it pulls harder against the rope holding it up."
            ),
        ),
        Parameter(
            "youngs_modulus",
            default=1e6,
            values=(1e5, 1e6, 1e7),
            unit="Pa",
            description="Stiffer ropes resist the kink at the touching point over a longer stretch.",
        ),
    ),
    requires=frozenset({Capability.ROD_CONTACT}),
    reference=reference_curves,
    metrics={
        "shape_error": shape_error,
        "lift_error": lift_error,
        "settling_residual": settling_residual,
    },
    # Odd, so that neither rope has a node where they touch. With a node there the
    # contact between two segment ends is degenerate, and pushes the lower rope
    # sideways by millimetres; see the findings.
    n_elements=51,
    notes=(
        "Two ropes cross at right angles. The upper one's pins are lower, so it comes "
        "down onto the other and is held there; they press on each other at one point, "
        "with no friction. The reference solves both ropes as elasticas, bending and "
        "all, and finds the force that leaves exactly one diameter between their "
        "centrelines: treating them as perfectly flexible is not good enough here, "
        "since bending rounds off the kink under the load by more than a rope's radius, "
        "and would ask for a third less force between them. The ropes start a centimetre "
        "further apart than they finish and settle into contact, so a solver has to find "
        "that force rather than be handed it. The lift tells you most: it is how far the "
        "lower rope holds the upper one up. Resolution matters more than usual here: with "
        "an even number of elements a node lands exactly where the ropes touch, and the "
        "contact comes out skewed."
    ),
)
