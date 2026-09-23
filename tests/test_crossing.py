"""The crossing experiment's reference: two ropes pressing on each other, each solved
as an elastica rather than as a perfectly flexible line."""

import itertools

import numpy as np
import pytest

from cosseratbench.experiments.crossing import (
    CLEARANCE,
    LENGTH,
    RADIUS,
    SPAN,
    _contact_force,
    _elastica,
    _mid_height,
    _properties,
    _shapes,
    build,
    crossing,
    flexible,
)
from cosseratbench.metrics import segment_distance


def test_the_flexible_rope_hangs_across_its_span_with_the_right_length():
    tension, curve = flexible(1e6, 0.3)
    assert tension > 0.0
    assert curve[-1, 1] == pytest.approx(SPAN / 2.0, abs=1e-9)
    assert curve[-1, 0] == pytest.approx(LENGTH / 2.0, abs=1e-12)


@pytest.mark.parametrize("load", [-0.4, -0.1, 0.0, 0.1, 0.4])
def test_the_elastica_meets_its_boundary_conditions(load):
    arc, curve = _elastica(1e6, load)
    assert arc[0] == 0.0 and arc[-1] == pytest.approx(LENGTH / 2.0)
    assert curve[0, 0] == pytest.approx(0.0, abs=1e-12)  # starts at its support
    assert curve[-1, 0] == pytest.approx(SPAN / 2.0, abs=1e-7)  # reaches half the span
    # Level at the middle, so the rope's two halves meet smoothly there.
    slope = np.diff(curve[:, 1]) / np.diff(curve[:, 0])
    assert abs(slope[-1]) < np.abs(slope).max() / 50.0


def test_a_heavier_load_pulls_the_middle_further_down():
    heights = [_mid_height(1e6, load) for load in (-0.3, -0.1, 0.0, 0.1, 0.3)]
    assert all(later < earlier for earlier, later in itertools.pairwise(heights))


def _flexible_mid_height(youngs_modulus: float, load: float) -> float:
    """Where the middle would sit if the rope had no bending stiffness at all."""
    return float(flexible(youngs_modulus, load)[1][-1, 2])


def test_bending_holds_the_middle_up_against_a_load_the_flexible_rope_takes_as_a_kink():
    # With no load the two agree closely: a pinned rope carries no moment at its ends,
    # which is why the catenary experiment can use the flexible answer. Under a point
    # load they part company by more than the ropes are thick, which is why this
    # experiment cannot.
    assert abs(_mid_height(1e6, 0.0) - _flexible_mid_height(1e6, 0.0)) < 2e-4
    assert _mid_height(1e6, 0.25) - _flexible_mid_height(1e6, 0.25) > RADIUS
    # Which moves the force they press on each other with by a third.
    bending = _contact_force(1e6, 0.1, _mid_height)
    flexible_force = _contact_force(1e6, 0.1, _flexible_mid_height)
    assert bending > 1.2 * flexible_force


def test_the_reference_leaves_exactly_one_diameter_between_the_ropes():
    for drop in (0.05, 0.1, 0.2):
        _, lower, upper = _shapes(1e6, drop, 2 * RADIUS)
        gap = upper[len(upper) // 2][2] - lower[len(lower) // 2][2]
        assert gap == pytest.approx(2 * RADIUS, abs=1e-9)


def test_pressing_harder_is_what_holds_the_ropes_further_apart():
    forces = [_contact_force(1e6, 0.1, _mid_height, apart) for apart in (0.01, 0.02, 0.03)]
    assert all(later > earlier for earlier, later in itertools.pairwise(forces))


def test_the_ropes_start_clear_of_each_other_at_every_resolution():
    # A rope's polyline cuts the corner where the load presses on it, so starting them
    # one diameter apart would start them overlapping. Check the clearance covers it.
    for values in ({}, {"drop": 0.05}, {"drop": 0.2}, {"youngs_modulus": 1e5}):
        scenario = crossing.scenario_for(**values)
        for n in (crossing.n_elements // 2, crossing.n_elements, crossing.n_elements * 2):
            first, second = (rod.nodes(n) for rod in scenario.rods)
            i, j = (index.ravel() for index in np.meshgrid(np.arange(n), np.arange(n)))
            closest = segment_distance(first[i], first[i + 1], second[j], second[j + 1]).min()
            assert closest > 2 * RADIUS, f"{values} at {n} elements starts overlapping"


def test_the_ropes_start_away_from_the_answer_so_a_solver_has_to_find_it():
    scenario = build(1e6, 0.1)
    _, _, answer = _shapes(1e6, 0.1, 2 * RADIUS)
    started = np.asarray(scenario.rods[1].centerline)
    moved = abs(started[len(started) // 2][2] - answer[len(answer) // 2][2])
    assert CLEARANCE / 4 < moved < CLEARANCE


def test_both_ropes_are_the_length_they_should_be_and_pinned_where_they_should_be():
    scenario = build(1e6, 0.1)
    lower, upper = scenario.rods
    for rod in scenario.rods:
        assert rod.length == pytest.approx(LENGTH, rel=1e-6)
    assert lower.centerline[0][0] == pytest.approx(-SPAN / 2.0)
    assert lower.centerline[-1][0] == pytest.approx(SPAN / 2.0)
    assert upper.centerline[0][1] == pytest.approx(-SPAN / 2.0)
    assert upper.centerline[0][2] == pytest.approx(-0.1)


def test_the_weight_and_stiffnesses_are_the_rods():
    weight, axial, bending = _properties(1e6)
    area = np.pi * RADIUS**2
    assert weight == pytest.approx(1000.0 * area * 9.81)
    assert axial == pytest.approx(1e6 * area)
    assert bending == pytest.approx(1e6 * np.pi * RADIUS**4 / 4)
