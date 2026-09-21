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
