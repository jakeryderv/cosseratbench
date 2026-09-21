"""The common output format every solver produces and every metric consumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray  # [T] seconds
    positions: tuple[np.ndarray, ...]  # per rod: [T, n_nodes, 3] metres, world frame

    def __post_init__(self) -> None:
        for rod in self.positions:
            if rod.ndim != 3 or rod.shape[0] != len(self.times) or rod.shape[2] != 3:
                raise ValueError(
                    f"expected positions of shape [{len(self.times)}, n_nodes, 3], got {rod.shape}"
                )

    def save(self, path: Path) -> None:
        rods = {f"rod{i}": rod for i, rod in enumerate(self.positions)}
        np.savez_compressed(path, times=self.times, **rods)

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        with np.load(path) as data:
            n_rods = len(data.files) - 1
            return cls(data["times"], tuple(data[f"rod{i}"] for i in range(n_rods)))
