"""A rope over a fixed horizontal cylinder, hanging further on one side than the other.

The longer side is heavier and pulls the rope toward it; friction on the cylinder
resists. The capstan equation, T2 <= e^(mu theta) T1 for a rope wrapped through
angle theta, ignores the weight of the rope lying on the cylinder. Here that
weight is not negligible: it presses the rope on, adding friction. With it, a
rope over the top half-turn of a cylinder of radius R (to the rope's centre),
hanging h1 on its light side under its own weight, holds while the heavy side
hangs at most

    h2* = e^(mu pi) h1 + R (2 mu / (1 + mu^2)) (e^(mu pi) + 1),

from tension along the wrap obeying dT/dphi = mu T + w R (cos phi + mu sin phi),
w the rope's weight per length. For this rope that is 1.26 times the plain
capstan ratio at mu = 0.3.

This is an exploration experiment (decision 0007). The overhang is a parameter,
a fraction of h2*, so some variants should hold and some slide, and the result to
look at is how far each rope moved. A rope that should hold but creeps shows how a
solver imitates static friction; penetration, reported on every run, shows how
stiff its contact is.

The rope starts stretched as it hangs, with a tension along the wrap that friction
can hold, so the start is an equilibrium; released unstretched it would jolt. It is
much thinner than the cylinder, as the equation assumes a flexible rope.
"""

from __future__ import annotations

import numpy as np

from cosseratbench.experiment import Experiment, Parameter
from cosseratbench.scenario import Cylinder, Material, Rod, Scenario
from cosseratbench.trajectory import Trajectory

CYLINDER_RADIUS = 0.05
RADIUS = 0.001
MATERIAL = Material(youngs_modulus=3e5, shear_modulus=1e5, density=1000.0)
GRAVITY = 9.81
SHORT_SIDE = 0.15  # m of rope hanging on the light side
DURATION = 2.0


def _wrap_radius() -> float:
    return CYLINDER_RADIUS + RADIUS


def holding_overhang(friction: float, short: float = SHORT_SIDE) -> float:
    """The longest heavy-side overhang, in m, at which the rope still holds."""
    ratio = np.exp(friction * np.pi)
    return float(ratio * short + _wrap_radius() * 2 * friction / (1 + friction**2) * (ratio + 1))


def _wrap_tension(
    phi: np.ndarray, t1: float, t2: float, weight: float, friction: float
) -> np.ndarray:
    """A tension along the wrap, from t1 at the light edge (phi = 0) to t2 at the heavy
    edge (phi = pi), in equilibrium with friction no more than ``friction``: the
    solution of dT/dphi = m T + w R (cos phi + m sin phi) for the m that reaches t2."""
    wr = weight * _wrap_radius()

    def tension(m: float) -> np.ndarray:
        a, b = (1 - m * m) / (1 + m * m), 2 * m / (1 + m * m)
        integral = a * np.exp(-m * phi) * np.sin(phi) - b * np.exp(-m * phi) * np.cos(phi) + b
        return np.exp(m * phi) * (t1 + wr * integral)

    low, high = -friction, friction
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if tension(middle)[-1] < t2 else (low, middle)
    return tension((low + high) / 2)


def _draped(
    short: float, long: float, friction: float, per_metre: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """The rope's material points, hanging `short` on the -x side, over the top of the
    cylinder and `long` on the +x side, stretched as it hangs; and their unstretched
    distances from the light end."""
    wrap = _wrap_radius()
    area = np.pi * RADIUS**2
    weight = MATERIAL.density * area * GRAVITY
    stiffness = MATERIAL.youngs_modulus * area

    # Unstretched coordinate along each part, and the tension there.
    s_left = np.linspace(0.0, short, int(short * per_metre), endpoint=False)
    phi = np.linspace(0.0, np.pi, int(np.pi * wrap * per_metre), endpoint=False)
    s_right = np.linspace(0.0, long, int(long * per_metre) + 1)
    t_left = weight * s_left  # the weight below
    t_wrap = _wrap_tension(phi, weight * short, weight * long, weight, friction)
    t_right = weight * (long - s_right)

    # Stretched lengths place the points; the wrap keeps its radius, so stretch spreads
    # it through more angle, which a tiny shift of the arc's start absorbs.
    stretched_left = np.cumsum(np.diff(s_left, prepend=0.0) * (1 + t_left / stiffness))
    left = np.stack(
        [
            np.full_like(s_left, -wrap),
            0 * s_left,
            stretched_left - stretched_left[-1] - (s_left[1] - s_left[0]),
        ],
        axis=1,
    )
    # Points a hair outside the wrap radius, so the straight pieces between them rest on
    # the cylinder rather than cutting into it.
    bulge = wrap / np.cos((phi[1] - phi[0]) / 2)
    arc = np.stack([-bulge * np.cos(phi), 0 * phi, bulge * np.sin(phi)], axis=1)
    right = np.stack(
        [
            np.full_like(s_right, wrap),
            0 * s_right,
            -np.cumsum(np.diff(s_right, prepend=0.0) * (1 + t_right / stiffness)),
        ],
        axis=1,
    )
    points = np.concatenate([left, arc, right])

    # Unstretched distance: each piece's length over (1 + its strain).
    tension = np.concatenate([t_left, t_wrap, t_right])
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    middle = (tension[1:] + tension[:-1]) / 2
    rest = np.concatenate([[0.0], np.cumsum(gaps / (1 + middle / stiffness))])
    return points, rest


def build(overhang: float, friction: float) -> Scenario:
    long = overhang * holding_overhang(friction)
    points, rest = _draped(SHORT_SIDE, long, friction)
    rope = Rod(
        centerline=tuple(map(tuple, points)),
        rest_arc_length=tuple(rest),
        radius=RADIUS,
        material=MATERIAL,
        normal=(0.0, 1.0, 0.0),
    )
    cylinder = Cylinder(
        center=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        radius=CYLINDER_RADIUS,
        length=0.1,
        friction=friction,
    )
    # The question is whether an equilibrium exists, so solvers may damp their way to it;
    # damping slows a sliding rope but cannot hold one that friction cannot.
    return Scenario(
        rods=(rope,),
        obstacles=(cylinder,),
        duration=DURATION,
        gravity=(0.0, 0.0, -GRAVITY),
        quasi_static=True,
        self_contact=False,  # the rope hangs straight down on both sides of the cylinder
    )


def slide(scenario: Scenario, trajectory: Trajectory) -> float:
    """How far the rope moved toward its heavy side: the most its light end rose, in m.
    A rope that slides all the way off falls, so its final height would mislead."""
    light_end = trajectory.positions[0][:, 0, 2]
    return float(light_end.max() - light_end[0])


capstan = Experiment(
    name="capstan",
    description="Rope over a fixed cylinder, heavier on one side, against the heavy-rope capstan equation.",
    build=build,
    parameters=(
        Parameter(
            "overhang",
            default=0.8,
            values=(0.5, 0.8, 0.95, 1.1, 1.5),
            description=(
                "The heavy side's overhang as a fraction of the longest that holds: "
                "below 1 the rope should hold, above 1 slide."
            ),
        ),
        Parameter(
            "friction",
            default=0.3,
            values=(0.1, 0.3, 0.6),
            description="Coulomb friction coefficient between rope and cylinder.",
        ),
    ),
    metrics={"slide": slide},
    n_elements=100,
    notes=(
        "A rope hangs over a fixed cylinder, longer on one side. Friction holds it "
        "while the long side is short enough; the overhang control sets it as a "
        "fraction of the longest that holds, which counts the weight of rope lying on "
        "the cylinder as well as the capstan equation. Below 1 the rope should stay "
        "put, and any slide is creep: how the solver imitates static friction. Above "
        "1 it should slide off. Penetration shows how stiff each solver's contact is."
    ),
)
