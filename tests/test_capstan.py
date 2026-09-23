import math

import numpy as np
import pytest

from cosseratbench import Trajectory
from cosseratbench.experiments.capstan import (
    CYLINDER_RADIUS,
    RADIUS,
    SHORT_SIDE,
    capstan,
    holding_overhang,
    slide,
)


def integrated_holding_overhang(mu, steps=20000):
    """Tangential equilibrium along the wrap at impending slip, integrated directly:
    dT/dphi = R (w cos phi + mu (T / R + w sin phi)), with w = 1."""
    radius = CYLINDER_RADIUS + RADIUS
    phi = np.linspace(0.0, np.pi, steps + 1)
    h = phi[1] - phi[0]

    def f(p, t):
        return radius * (np.cos(p) + mu * (t / radius + np.sin(p)))

    t = SHORT_SIDE
    for p in phi[:-1]:
        k1 = f(p, t)
        k2 = f(p + h / 2, t + h / 2 * k1)
        k3 = f(p + h / 2, t + h / 2 * k2)
        k4 = f(p + h, t + h * k3)
        t += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t


@pytest.mark.parametrize("mu", [0.0, 0.1, 0.3, 0.6])
def test_holding_overhang_solves_equilibrium_with_the_ropes_own_weight(mu):
    assert holding_overhang(mu) == pytest.approx(integrated_holding_overhang(mu), rel=1e-9)


def test_weight_on_the_cylinder_lets_it_hold_more_than_the_plain_capstan():
    assert holding_overhang(0.0) == pytest.approx(SHORT_SIDE)  # no friction: balanced
    assert holding_overhang(0.3) > math.exp(0.3 * math.pi) * SHORT_SIDE


def test_rope_starts_stretched_as_it_hangs_and_resting_on_the_cylinder():
    rope = capstan.scenario.rods[0]
    points = np.array(rope.centerline)
    drawn = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    assert drawn > rope.length  # stretched by its own weight
    assert drawn / rope.length - 1 < 0.02
    nodes = rope.nodes(capstan.n_elements)
    touching = nodes[:, 2] > 0  # on the top half, over the cylinder
    distance = np.hypot(nodes[touching, 0], nodes[touching, 2])
    assert distance.min() >= CYLINDER_RADIUS + RADIUS - 1e-9  # not sunk in at the start


def test_slide_is_how_far_the_light_end_rose_even_if_the_rope_then_fell_off():
    t = np.linspace(0.0, 2.0, 5)
    heights = np.array([-0.15, -0.05, 0.05, -1.0, -3.0])  # up over the top, then off and down
    positions = np.zeros((5, 3, 3))
    positions[:, 0, 2] = heights
    trajectory = Trajectory(t, (positions,), (np.zeros((5, 2, 3)),))
    assert slide(capstan.scenario, trajectory) == pytest.approx(0.2)
