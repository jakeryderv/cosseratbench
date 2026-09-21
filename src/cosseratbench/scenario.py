"""Scenario specification: what is being simulated, in physical terms only.

Nothing here may describe *how* a solver discretises or integrates the problem.
Time steps, element counts, contact stiffnesses and damping coefficients belong
to solver adapters. All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Material:
    youngs_modulus: float  # Pa
    shear_modulus: float  # Pa
    density: float  # kg/m^3


class EndCondition(Enum):
    FREE = "free"
    PINNED = "pinned"  # position held, rotation free
    CLAMPED = "clamped"  # position and orientation held


class End(Enum):
    START = "start"
    END = "end"


@dataclass(frozen=True)
class PointLoad:
    """A constant force in the world frame, applied at one end of a rod."""

    force: Vec3  # N
    at: End = End.END


@dataclass(frozen=True)
class Rod:
    """A uniform circular rod whose stress-free shape is straight.

    ``centerline`` is the initial shape as a polyline; its arc length is the
    rod's length. Two points describe a straight rod.
    """

    centerline: tuple[Vec3, ...]
    radius: float  # m
    material: Material
    normal: Vec3 = (0.0, 0.0, 1.0)  # reference for the material frame, not parallel to the rod
    start: EndCondition = EndCondition.FREE
    end: EndCondition = EndCondition.FREE
    loads: tuple[PointLoad, ...] = ()

    @property
    def length(self) -> float:
        points = np.asarray(self.centerline)
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    @property
    def area(self) -> float:
        return float(np.pi * self.radius**2)

    @property
    def second_moment_of_area(self) -> float:
        return float(np.pi * self.radius**4 / 4.0)

    def nodes(self, n_elements: int) -> np.ndarray:
        """Resample the centerline to ``n_elements + 1`` nodes evenly spaced in arc length."""
        points = np.asarray(self.centerline, dtype=float)
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        targets = np.linspace(0.0, arc[-1], n_elements + 1)
        return np.stack([np.interp(targets, arc, points[:, axis]) for axis in range(3)], axis=1)


@dataclass(frozen=True)
class Scenario:
    rods: tuple[Rod, ...]
    duration: float  # s
    gravity: Vec3 = (0.0, 0.0, 0.0)  # m/s^2
    # When True only the final equilibrium matters, and a solver may add whatever
    # dissipation gets it there. When False the dynamics are the result, and a
    # solver must add none beyond what the scenario specifies.
    quasi_static: bool = False

    def slowest_frequency(self) -> float:
        """Rough angular frequency (rad/s) of the slowest mode.

        A scale for solvers choosing dissipation in quasi-static scenarios; not a result.
        """
        g = float(np.linalg.norm(self.gravity))
        estimates = []
        for rod in self.rods:
            m = rod.material
            bending = 3.516 * np.sqrt(
                m.youngs_modulus * rod.second_moment_of_area / (m.density * rod.area)
            )
            estimates.append(max(bending / rod.length**2, np.sqrt(g / rod.length)))
        return float(min(estimates))
