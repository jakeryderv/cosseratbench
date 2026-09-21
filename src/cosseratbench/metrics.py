"""Building blocks shared by experiment metrics."""

from __future__ import annotations

import numpy as np

from cosseratbench.scenario import Scenario
from cosseratbench.trajectory import Trajectory


def max_distance_to_curve(nodes: np.ndarray, curve: np.ndarray) -> float:
    """Largest distance from any of ``nodes`` [N, 3] to a densely sampled ``curve`` [M, 3]."""
    distances = np.linalg.norm(nodes[:, None, :] - curve[None, :, :], axis=2)
    return float(distances.min(axis=1).max())


def settling_residual(scenario: Scenario, trajectory: Trajectory) -> float:
    """Largest node speed over the last frame interval, in rod lengths per second.

    Near zero when a quasi-static run has reached equilibrium; when it is not,
    the other metrics describe a state still in motion.
    """
    dt = trajectory.times[-1] - trajectory.times[-2]
    return max(
        float(np.linalg.norm(positions[-1] - positions[-2], axis=1).max() / dt / rod.length)
        for rod, positions in zip(scenario.rods, trajectory.positions)
    )
