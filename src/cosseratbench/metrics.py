"""Building blocks shared by experiment metrics."""

from __future__ import annotations

import numpy as np

from cosseratbench.scenario import Scenario
from cosseratbench.trajectory import Trajectory


def segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Shortest distance between the segment a0-a1 and the segment b0-b1.

    The four arrays are [..., 3] and broadcast against each other, so many pairs
    are measured at once. Closest points are found in the interior where the
    segments' common perpendicular lands inside both, and on an endpoint where it
    does not; parallel segments fall back to the endpoint search.
    """
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    b = (d1 * d2).sum(-1)
    c = (d1 * r).sum(-1)
    f = (d2 * r).sum(-1)

    # Zero-length segments and parallel pairs make these denominators vanish; each
    # is replaced by one where it is unused, and the endpoint clamps below decide.
    def safe(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, x, 1.0)

    denominator = a * e - b * b
    parallel = denominator <= 1e-12 * a * e
    s = np.where(parallel, 0.0, (b * f - c * e) / safe(denominator))
    s = np.clip(s, 0.0, 1.0)
    t = (b * s + f) / safe(e)
    s = np.where(
        t < 0.0,
        np.clip(-c / safe(a), 0.0, 1.0),
        np.where(t > 1.0, np.clip((b - c) / safe(a), 0.0, 1.0), s),
    )
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(r + s[..., None] * d1 - t[..., None] * d2, axis=-1)


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
