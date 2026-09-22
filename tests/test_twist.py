import math

import numpy as np
import pytest

from cosseratbench import Trajectory
from cosseratbench.experiments.twist import (
    LEVELS,
    RAMP,
    critical_moment,
    critical_twist_error,
    growth_rates,
    measured_critical_twist,
    twist,
)


def test_greenhill_without_tension():
    # tan(m / 2) = m / 2: Greenhill's clamped rod, M L / B = 2.861 pi.
    assert critical_moment(1e-6) == pytest.approx(2 * 4.493409457909064, rel=1e-5)


def test_tension_raises_the_threshold_towards_the_long_rod_limit():
    moments = [critical_moment(tau) for tau in (1.0, 10.0, 100.0, 1000.0)]
    assert moments == sorted(moments)
    for tau, m in zip((1.0, 10.0, 100.0, 1000.0), moments):
        assert m > 2 * math.sqrt(tau)  # no finite rod buckles below M^2 = 4 B T
    assert moments[-1] / (2 * math.sqrt(1000.0)) < 1.02


def test_each_rod_is_held_at_its_own_fraction_of_the_critical_twist():
    rods = twist.scenario.rods
    assert len(rods) == len(LEVELS)
    turns = [rod.end_motion.rotation[-1][0] for rod in rods]
    np.testing.assert_allclose(np.array(turns) / turns[LEVELS.index(1.00)], LEVELS)
    for rod in rods:
        assert rod.end_motion.slides_along == (1.0, 0.0, 0.0)
        assert rod.end_motion.times[-1] == RAMP  # then held


def synthetic(critical, damping=1.5, stiffness=60.0, frames=321):
    """Rods whose sideways distance grows as one unstable mode would, past `critical`."""
    t = np.linspace(0.0, twist.scenario.duration, frames)
    positions = []
    for rod, level in zip(twist.scenario.rods, LEVELS):
        k = stiffness * (level - critical)  # s^2 + c s = k
        rate = (-damping + math.sqrt(damping**2 + 4 * k)) / 2 if k > 0 else -0.01
        sideways = 1e-5 * np.exp(rate * np.clip(t - RAMP, 0.0, None))
        ends = np.array(rod.centerline)[[0, -1]]  # a straight rod between the clamps
        nodes = np.stack([ends[0], ends.mean(axis=0), ends[1]])
        frames_ = np.repeat(nodes[None], frames, axis=0)
        frames_[:, 1, 2] = sideways  # the middle node strays along z
        positions.append(frames_)
    return Trajectory(t, tuple(positions))


def test_fit_recovers_the_threshold_whatever_the_damping():
    for critical, damping in ((0.99, 1.1), (1.01, 3.0), (0.97, 0.5)):
        trajectory = synthetic(critical, damping)
        assert measured_critical_twist(twist.scenario, trajectory) == pytest.approx(
            critical, abs=2e-3
        )
    assert critical_twist_error(twist.scenario, synthetic(1.0)) < 2e-3


def test_growth_is_read_only_while_small():
    rates = growth_rates(twist.scenario, synthetic(0.99))
    assert np.isnan(rates).sum() == 0 or rates[np.isfinite(rates)].max() < 10


def test_threshold_is_unmeasurable_when_too_few_rods_grow():
    assert math.isnan(measured_critical_twist(twist.scenario, synthetic(1.05)))
