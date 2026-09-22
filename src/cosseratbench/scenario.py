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


def _cross_matrix(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """The rotation a rotation vector (axis times angle, in radians) describes."""
    angle = float(np.linalg.norm(rotation))
    k = _cross_matrix(np.asarray(rotation, dtype=float))
    if angle < 1e-12:
        return np.eye(3) + k
    return np.eye(3) + np.sin(angle) / angle * k + (1.0 - np.cos(angle)) / angle**2 * k @ k


@dataclass(frozen=True)
class Motion:
    """Prescribed motion of a clamped end, relative to where it starts.

    At each of ``times`` the end has moved by ``displacement`` and turned by
    ``rotation`` about its own starting position, both in the world frame; a
    rotation is a rotation vector, axis times angle in radians. Between times both
    are interpolated linearly, and after the last the end holds still. Sample a
    smooth motion finely enough that its corners do not matter.

    With ``slides_along``, the end is not held along that direction: it slides
    freely along it under whatever acts on it, such as a load hanging from it.
    The motion then moves it only across that direction, and turns it only about it.
    """

    times: tuple[float, ...]  # s, from 0
    displacement: tuple[Vec3, ...]  # m
    rotation: tuple[Vec3, ...]  # rad
    slides_along: Vec3 | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        if len(times) < 2 or times[0] != 0.0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must start at 0 and increase")
        if not (len(self.displacement) == len(self.rotation) == len(times)):
            raise ValueError("displacement and rotation need one entry per time")
        if any(self.displacement[0]) or any(self.rotation[0]):
            raise ValueError("a motion starts where the end starts: zero at time 0")
        if self.slides_along is not None:
            axis = self.axis
            along = np.asarray(self.displacement, float) @ axis
            rotation = np.asarray(self.rotation, float)
            across = np.linalg.norm(rotation - np.outer(rotation @ axis, axis), axis=1)
            if np.any(np.abs(along) > 1e-12) or np.any(across > 1e-12):
                raise ValueError(
                    "a sliding end moves only across its slide direction and turns only about it"
                )

    @property
    def axis(self) -> np.ndarray:
        """The unit slide direction."""
        axis = np.asarray(self.slides_along, dtype=float)
        return axis / np.linalg.norm(axis)

    def _segment(self, time: float) -> tuple[int, float]:
        """Index of the interval containing ``time`` and how far through it, in [0, 1]."""
        times = self.times
        if time >= times[-1]:
            return len(times) - 2, 1.0
        i = int(np.searchsorted(times, time, side="right")) - 1
        return i, (time - times[i]) / (times[i + 1] - times[i])

    def pose(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Displacement, and rotation matrix, of the end at ``time``."""
        i, f = self._segment(time)
        d, r = np.asarray(self.displacement, float), np.asarray(self.rotation, float)
        return d[i] + f * (d[i + 1] - d[i]), rotation_matrix(r[i] + f * (r[i + 1] - r[i]))

    def rates(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Velocity, and angular velocity in the world frame, of the end at ``time``."""
        if time >= self.times[-1]:
            return np.zeros(3), np.zeros(3)
        i, f = self._segment(time)
        dt = self.times[i + 1] - self.times[i]
        d, r = np.asarray(self.displacement, float), np.asarray(self.rotation, float)
        velocity = (d[i + 1] - d[i]) / dt
        rotation, turning = r[i] + f * (r[i + 1] - r[i]), (r[i + 1] - r[i]) / dt
        # A rotation vector changing at rate v turns the body at J(r) v, J the left Jacobian.
        angle = float(np.linalg.norm(rotation))
        k = _cross_matrix(rotation)
        if angle < 1e-6:
            jacobian = np.eye(3) + 0.5 * k
        else:
            jacobian = (
                np.eye(3)
                + (1.0 - np.cos(angle)) / angle**2 * k
                + (angle - np.sin(angle)) / angle**3 * k @ k
            )
        return velocity, jacobian @ turning


@dataclass(frozen=True)
class Rod:
    """A uniform circular rod whose stress-free shape is straight.

    ``centerline`` is the initial shape as a polyline. Two points describe a
    straight rod.

    ``rest_arc_length``, if given, is the unstretched distance along the rod to
    each centerline point, so a rod can start stretched: a cable already hanging
    in equilibrium, say, rather than one that stretches the moment gravity acts.
    Without it the rod starts unstretched and its length is the polyline's.

    A clamped end may be driven: ``start_motion`` or ``end_motion`` then moves it
    and turns it over time instead of holding it still.
    """

    centerline: tuple[Vec3, ...]
    radius: float  # m
    material: Material
    normal: Vec3 = (0.0, 0.0, 1.0)  # reference for the material frame, not parallel to the rod
    start: EndCondition = EndCondition.FREE
    end: EndCondition = EndCondition.FREE
    loads: tuple[PointLoad, ...] = ()
    rest_arc_length: tuple[float, ...] | None = None  # m, one per centerline point, from 0
    start_motion: Motion | None = None
    end_motion: Motion | None = None

    def motion(self, end: End) -> Motion | None:
        return self.start_motion if end is End.START else self.end_motion

    def condition(self, end: End) -> EndCondition:
        return self.start if end is End.START else self.end

    def __post_init__(self) -> None:
        for end in End:
            if self.motion(end) is not None and self.condition(end) is not EndCondition.CLAMPED:
                raise ValueError(f"only a clamped end can be driven; the {end.value} is not")
        if self.rest_arc_length is not None:
            rest = np.asarray(self.rest_arc_length)
            if len(rest) != len(self.centerline) or rest[0] != 0.0 or np.any(np.diff(rest) <= 0):
                raise ValueError(
                    "rest_arc_length must start at 0, increase, and match the centerline"
                )

    def _arc(self) -> np.ndarray:
        """Unstretched distance along the rod to each centerline point."""
        if self.rest_arc_length is not None:
            return np.asarray(self.rest_arc_length, dtype=float)
        points = np.asarray(self.centerline, dtype=float)
        return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))

    @property
    def length(self) -> float:
        """Unstretched length."""
        return float(self._arc()[-1])

    @property
    def area(self) -> float:
        return float(np.pi * self.radius**2)

    @property
    def second_moment_of_area(self) -> float:
        return float(np.pi * self.radius**4 / 4.0)

    def nodes(self, n_elements: int) -> np.ndarray:
        """Initial positions of ``n_elements + 1`` material points evenly spaced along the
        unstretched rod, interpolated along the centerline."""
        points = np.asarray(self.centerline, dtype=float)
        arc = self._arc()
        targets = np.linspace(0.0, arc[-1], n_elements + 1)
        return np.stack([np.interp(targets, arc, points[:, axis]) for axis in range(3)], axis=1)


@dataclass(frozen=True)
class Scenario:
    rods: tuple[Rod, ...]
    duration: float  # s
    gravity: Vec3 = (0.0, 0.0, 0.0)  # m/s^2
    # When True the result is equilibrium: the final state, or the states passed
    # through as a load changes slowly, and a solver may add whatever dissipation
    # gets it there. When False the dynamics are the result, and a solver must add
    # none beyond what the scenario specifies.
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
