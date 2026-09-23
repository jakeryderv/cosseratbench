"""The common output format every solver produces and every metric consumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Where every rod is, and which way round it is, at each recorded time.

    ``positions`` are the nodes of each rod. ``directors`` are the material direction
    of each element between them: a unit vector perpendicular to the element that a
    solver carries with the rod's cross-section, the ``d1`` of a Cosserat rod. At
    t = 0 it is ``Rod.directors``, so it is the same material line for every solver.
    Together with the tangent it gives the cross-section's whole orientation, so
    twist is visible and measurable rather than inferred from the centerline.
    """

    times: np.ndarray  # [T] seconds
    positions: tuple[np.ndarray, ...]  # per rod: [T, n_nodes, 3] metres, world frame
    directors: tuple[np.ndarray, ...]  # per rod: [T, n_nodes - 1, 3] unit vectors, world frame

    def __post_init__(self) -> None:
        if len(self.directors) != len(self.positions):
            raise ValueError("expected directors for every rod that has positions")
        for rod, directors in zip(self.positions, self.directors):
            if rod.ndim != 3 or rod.shape[0] != len(self.times) or rod.shape[2] != 3:
                raise ValueError(
                    f"expected positions of shape [{len(self.times)}, n_nodes, 3], got {rod.shape}"
                )
            if directors.shape != (len(self.times), rod.shape[1] - 1, 3):
                raise ValueError(
                    f"expected directors of shape [{len(self.times)}, {rod.shape[1] - 1}, 3], "
                    f"got {directors.shape}"
                )

    def save(self, path: Path) -> None:
        rods = {f"rod{i}": rod for i, rod in enumerate(self.positions)}
        directors = {f"rod{i}_directors": d for i, d in enumerate(self.directors)}
        np.savez_compressed(path, times=self.times, **rods, **directors)

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        with np.load(path) as data:
            n_rods = sum(name.startswith("rod") and name[3:].isdigit() for name in data.files)
            return cls(
                data["times"],
                tuple(data[f"rod{i}"] for i in range(n_rods)),
                tuple(data[f"rod{i}_directors"] for i in range(n_rods)),
            )
