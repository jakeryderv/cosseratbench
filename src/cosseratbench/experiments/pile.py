"""A rope lowered onto a floor faster than it can lie down, so it heaps up.

The rope hangs from a clamp, its free end just above the floor. The clamp descends
at a steady speed until most of the rope has been fed onto the floor, then holds
a little above the heap. Rope arriving at the floor faster than the pile can carry
it away buckles, folds and coils on top of what is already there: a heap of loops
pressed on each other, sliding over each other and over the floor, with the rope
touching itself in many places at once.

This is an exploration experiment (decision 0007), with no reference: a pile is
chaotic in its details, and the point is to see how each solver copes with dense,
persistent self-contact and friction, and what it costs. Solvers that cannot give
rods friction against each other (PyElastica, decision 0011) still run it, since
the floor has friction of its own; their piles spread more, which is the finding.

The rope starts stretched as it hangs, so it does not bounce along its length when
released, and with a bow of a thousandth of its length so that it folds a definite
way rather than by numerical noise. The bow lies in no one plane: given a flat one,
a solver whose arithmetic keeps the symmetry folds the rope back and forth in that
plane for ever and never makes a heap. The clamp stops well clear of the floor:
lowered into the heap, it crushes the standing loop under it and flings the rope
out flat. Metrics leave out the length that still hangs from the clamp at the end.

The heap at rest is the result, so solvers may add dissipation to reach it
(``quasi_static``); without any, an elastic rope with no material damping goes on
jiggling on the floor indefinitely. The dissipation the adapters add is mild next
to the feed: a free rope would fall through it several times faster than the clamp
descends.
"""

from __future__ import annotations

import numpy as np

from cosseratbench.experiment import Experiment, Parameter
from cosseratbench.experiments.pendulum import static_strain
from cosseratbench.metrics import segment_distance, settling_residual
from cosseratbench.scenario import (
    EndCondition,
    Material,
    Motion,
    Plane,
    Rod,
    Scenario,
    neighbour_elements,
)
from cosseratbench.solver import Capability
from cosseratbench.trajectory import Trajectory

LENGTH = 1.5  # m of rope
RADIUS = 0.0025  # m: a 5 mm cord
DENSITY = 1000.0
GRAVITY = 9.81
CLEARANCE = 0.02  # m between the rope's free end and the floor at the start
CLAMP_END = 0.15  # m above the floor where the clamp stops: clear of the heap
SETTLE = 1.0  # s of holding still after the feed
IMPERFECTION = 1e-3  # sideways bow of the hanging rope, in rod lengths, out of any one plane
TAIL = 2.0 * CLAMP_END  # m of rope below the clamp left out of the pile's measurements
TOUCH = 1.1  # elements closer than this many diameters count as touching


def _hanging(youngs_modulus: float) -> tuple[np.ndarray, np.ndarray]:
    """The rope's points as it hangs from the clamp, stretched by its own weight, with
    its free end CLEARANCE above the floor; and the unstretched distance from the clamp
    to each."""
    material = Material(youngs_modulus, youngs_modulus / 3.0, DENSITY)
    s = np.linspace(0.0, LENGTH, 201)
    strain = static_strain(s, LENGTH, material, GRAVITY)
    down = np.concatenate(([0.0], np.cumsum((2.0 + strain[1:] + strain[:-1]) / 2.0 * np.diff(s))))
    top = CLEARANCE + down[-1]
    bow = IMPERFECTION * LENGTH
    across = np.stack([np.sin(np.pi * s / LENGTH), np.sin(2.0 * np.pi * s / LENGTH)], axis=1)
    return np.column_stack([bow * across, top - down]), s


def feed_time(speed: float, youngs_modulus: float) -> float:
    """How long the clamp descends: from where it starts to CLAMP_END above the floor."""
    top = _hanging(youngs_modulus)[0][0, 2]
    return float((top - CLAMP_END) / speed)


def build(speed: float, friction: float, youngs_modulus: float) -> Scenario:
    points, arc = _hanging(youngs_modulus)
    descent = feed_time(speed, youngs_modulus)
    lower = Motion(
        times=(0.0, descent),
        displacement=((0.0, 0.0, 0.0), (0.0, 0.0, -speed * descent)),
        rotation=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    rope = Rod(
        centerline=tuple(map(tuple, points)),
        rest_arc_length=tuple(arc),
        radius=RADIUS,
        material=Material(youngs_modulus, youngs_modulus / 3.0, DENSITY),
        normal=(0.0, 1.0, 0.0),
        start=EndCondition.CLAMPED,
        start_motion=lower,
        friction=friction,
    )
    floor = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), friction=friction)
    # The heap at rest is the result, so solvers may add dissipation to reach it.
    return Scenario(
        rods=(rope,),
        obstacles=(floor,),
        duration=descent + SETTLE,
        gravity=(0.0, 0.0, -GRAVITY),
        quasi_static=True,
        self_contact=True,
    )


def _piled(scenario: Scenario, trajectory: Trajectory) -> np.ndarray:
    """The final positions of the nodes in the pile: all but the TAIL below the clamp."""
    positions = trajectory.positions[0][-1]
    arc = np.linspace(0.0, scenario.rods[0].length, len(positions))
    return positions[arc >= TAIL]


def pile_height(scenario: Scenario, trajectory: Trajectory) -> float:
    """Height of the pile's highest node above the floor, in rod diameters."""
    return float(_piled(scenario, trajectory)[:, 2].max() / (2.0 * RADIUS))


def footprint(scenario: Scenario, trajectory: Trajectory) -> float:
    """Radius of the pile: the furthest any of its nodes lies, horizontally, from
    their centre, in m."""
    horizontal = _piled(scenario, trajectory)[:, :2]
    return float(np.linalg.norm(horizontal - horizontal.mean(axis=0), axis=1).max())


def contacts(scenario: Scenario, trajectory: Trajectory) -> float:
    """How many pairs of the rope's elements touch at the end: closer than TOUCH
    diameters, and further apart along the rope than neighbours can be."""
    rod = scenario.rods[0]
    positions = trajectory.positions[0][-1]
    n = len(positions) - 1
    apart = neighbour_elements(rod.radius, rod.length / n)
    first, second = np.triu_indices(n, k=apart)
    distance = segment_distance(
        positions[first], positions[first + 1], positions[second], positions[second + 1]
    )
    return float((distance < TOUCH * 2.0 * rod.radius).sum())


pile = Experiment(
    name="pile",
    description="Rope lowered onto a floor faster than it can lie down, heaping into a pile.",
    build=build,
    parameters=(
        Parameter(
            "speed",
            default=0.4,
            values=(0.2, 0.4, 0.8),
            unit="m/s",
            description="How fast the clamp descends: rope arrives faster than it can spread.",
        ),
        Parameter(
            "friction",
            default=0.3,
            values=(0.1, 0.3, 0.6),
            description="Coulomb friction of the rope against the floor and against itself.",
        ),
        Parameter(
            "youngs_modulus",
            default=1e6,
            values=(1e5, 1e6, 1e7),
            unit="Pa",
            description="Stiffer rope coils in wider loops and stands taller.",
        ),
    ),
    requires=frozenset({Capability.ROD_CONTACT}),
    metrics={
        "pile_height": pile_height,
        "footprint": footprint,
        "contacts": contacts,
        "settling_residual": settling_residual,
    },
    n_elements=100,
    n_frames=201,
    notes=(
        "A rope hangs from a clamp that descends at a steady speed until most of it "
        "has been fed onto the floor. Rope arriving faster than the pile can carry "
        "it away folds and coils on top of what is already there. There is no reference: "
        "look at the pile itself, how tall and how wide it is, how many places the rope "
        "touches itself, whether it goes on creeping once the clamp stops, and what the "
        "run cost. Overlap shows how far the rope passes into itself. PyElastica has no "
        "friction between rods, only against the floor, so its coils slide over each "
        "other freely."
    ),
)
