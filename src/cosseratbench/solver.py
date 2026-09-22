"""The interface a solver adapter implements to take part in the benchmark."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from cosseratbench.scenario import Scenario
from cosseratbench.trajectory import Trajectory


class Capability(Enum):
    """Physics an adapter models. Experiments list what they require; a solver
    missing a requirement is reported as unsupported rather than run."""

    STRETCH = "stretch"
    SHEAR = "shear"


class Diverged(FloatingPointError):
    """Raised by a solver whose simulation broke down numerically."""

    def __init__(self, message: str, time: float | None = None) -> None:
        super().__init__(message)
        self.time = time  # s, when it was first seen, if known


# Solver options are numerical settings that are fair to vary across solvers, kept
# apart from the scenario (decision 0008). An adapter accepts each as a keyword
# argument to its constructor; resolution is passed to run() instead.
TIME_STEP_SCALE = "time_step_scale"  # multiplies the time step the adapter would choose


@runtime_checkable
class Solver(Protocol):
    name: str
    capabilities: frozenset[Capability]

    def run(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> Trajectory:
        """Simulate ``scenario`` with ``n_elements`` elements per rod.

        Returns ``n_frames`` evenly spaced frames, the first at t=0 and the last
        at ``scenario.duration``. Everything else about the numerics (time step,
        integrator, contact and damping parameters) is the adapter's choice.
        """
        ...
