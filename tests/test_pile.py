import numpy as np
import pytest

from cosseratbench import Trajectory
from cosseratbench.experiments.pile import (
    CLAMP_END,
    CLEARANCE,
    LENGTH,
    RADIUS,
    SETTLE,
    contacts,
    feed_time,
    footprint,
    pile,
    pile_height,
)


def test_the_rope_hangs_stretched_from_the_clamp_just_above_the_floor():
    scenario = pile.scenario
    rope = scenario.rods[0]
    points = np.asarray(rope.centerline)
    assert rope.length == pytest.approx(LENGTH)
    assert points[-1, 2] == pytest.approx(CLEARANCE)  # free end just above the floor
    assert points[0, 2] > CLEARANCE + LENGTH  # stretched by its own weight
    # The bow, and in no one plane: a flat bow leaves a symmetric solver folding flat.
    assert abs(points[:, 0]).max() == pytest.approx(1e-3 * LENGTH, rel=1e-6)
    assert abs(points[:, 1]).max() == pytest.approx(1e-3 * LENGTH, rel=1e-6)
    assert np.linalg.matrix_rank(points[:, :2] - points[0, :2], tol=1e-6) == 2
    assert scenario.obstacles[0].kind == "plane"
    assert scenario.quasi_static  # the heap at rest is the result


def test_the_clamp_descends_to_just_above_the_floor_and_then_holds():
    scenario = pile.scenario
    rope = scenario.rods[0]
    motion = rope.start_motion
    descent = feed_time(0.4, 1e6)
    assert motion.times[-1] == pytest.approx(descent)
    assert rope.centerline[0][2] + motion.displacement[-1][2] == pytest.approx(CLAMP_END)
    assert scenario.duration == pytest.approx(descent + SETTLE)
    # Feeding faster takes less time but lays down the same rope.
    fast = pile.scenario_for(speed=0.8)
    assert fast.duration < scenario.duration
    assert fast.rods[0].start_motion.displacement[-1] == motion.displacement[-1]


def coiled(
    turns: float, coil_radius: float, rise: float, tail_height: float = RADIUS, n: int = 100
) -> Trajectory:
    """A rope wound ``turns`` times in a helix on the floor, rising ``rise`` per turn,
    with the TAIL below the clamp a straight spoke at ``tail_height`` coming in
    radially from outside the coil to where the pile starts."""
    scenario = pile.scenario
    s = np.linspace(0.0, LENGTH, n + 1)
    angle = 2 * np.pi * turns * s / LENGTH
    positions = np.stack(
        [
            coil_radius * np.cos(angle),
            coil_radius * np.sin(angle),
            RADIUS + rise * angle / (2 * np.pi),
        ],
        axis=1,
    )
    tail = s < 2 * CLAMP_END
    theta = angle[np.flatnonzero(~tail)[0]]
    radius = coil_radius * (1.0 + 5.0 * (1.0 - s[tail] / (2 * CLAMP_END)))  # a long spoke
    positions[tail] = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full(tail.sum(), tail_height)], axis=1
    )
    times = np.array([0.0, scenario.duration])
    return Trajectory(times, (np.stack([positions, positions]),), (np.zeros((2, n, 3)),))


def test_pile_metrics_leave_out_the_tail_hanging_from_the_clamp():
    # The tail sits far above the pile; it must not count as its height.
    trajectory = coiled(turns=3.0, coil_radius=0.05, rise=2 * RADIUS, tail_height=0.5)
    scenario = pile.scenario
    # Three turns rising a diameter each: the top of the pile is three diameters up,
    # plus the half diameter the rope's centre sits above the floor.
    assert pile_height(scenario, trajectory) == pytest.approx(3.5, abs=0.05)
    assert footprint(scenario, trajectory) == pytest.approx(0.05, abs=0.01)


def test_contacts_count_the_element_pairs_that_touch():
    scenario = pile.scenario
    # Turns stacked a diameter apart touch all the way round; spread four apart, none.
    # The tail runs in along the floor, clear of the turns stacked above the coil.
    touching = coiled(turns=3.0, coil_radius=0.05, rise=2 * RADIUS)
    clear = coiled(turns=3.0, coil_radius=0.05, rise=8 * RADIUS)
    assert contacts(scenario, touching) > 50
    assert contacts(scenario, clear) == 0
