"""An experiment pairs a scenario with the metrics that judge a solver's trajectory."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from cosseratbench.scenario import Scenario
from cosseratbench.solver import Capability, Solver
from cosseratbench.trajectory import Trajectory

# Metrics see only the scenario and the trajectory, never the solver, so every
# solver is judged by the same code.
Metric = Callable[[Scenario, Trajectory], float]
# The analytical answer as a curve [M, 3], where one exists, for drawing beside the solvers'.
Reference = Callable[[Scenario], np.ndarray]


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    scenario: Scenario
    metrics: Mapping[str, Metric]
    requires: frozenset[Capability] = frozenset()
    n_elements: int = 50  # default resolution
    reference: Reference | None = None

    def save(self, directory: Path) -> None:
        """Write everything a reader of the results needs to interpret them without this code."""
        directory.mkdir(parents=True, exist_ok=True)
        reference = None if self.reference is None else self.reference(self.scenario).tolist()
        summary = {
            "name": self.name,
            "description": self.description,
            "scenario": asdict(self.scenario),
            "rod_lengths": [rod.length for rod in self.scenario.rods],
            "reference": reference,
        }
        text = json.dumps(summary, default=lambda enum: enum.value)
        (directory / "experiment.json").write_text(text + "\n")


@dataclass(frozen=True)
class Result:
    experiment: str
    solver: str
    n_elements: int
    # Capabilities the experiment requires and the solver lacks; if any, it was not run.
    missing: tuple[str, ...] = ()
    failure: str | None = None  # why the run produced no usable trajectory
    wall_time: float | None = None  # s
    metrics: Mapping[str, float] = field(default_factory=dict)
    trajectory: Trajectory | None = None

    @property
    def supported(self) -> bool:
        return not self.missing

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment": self.experiment,
            "solver": self.solver,
            "n_elements": self.n_elements,
            "missing": list(self.missing),
            "failure": self.failure,
            "wall_time": self.wall_time,
            "metrics": dict(self.metrics),
        }
        (directory / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        if self.trajectory is not None:
            self.trajectory.save(directory / "trajectory.npz")


def _divergence(scenario: Scenario, trajectory: Trajectory) -> str | None:
    """A solver can go unstable without producing NaNs or raising, so judge the
    trajectory itself: no real rod here doubles the length of one of its segments."""
    for index, positions in enumerate(trajectory.positions):
        if not np.isfinite(positions).all():
            return f"rod {index} has non-finite positions"
        segments = np.linalg.norm(np.diff(positions, axis=1), axis=2)
        stretch = float((segments / segments[0]).max())
        if stretch > 2.0:
            return f"rod {index} has a segment stretched {stretch:.3g}x"
    return None


def run(
    experiment: Experiment,
    solver: Solver,
    *,
    n_elements: int | None = None,
    n_frames: int = 101,
) -> Result:
    n_elements = n_elements or experiment.n_elements
    missing = experiment.requires - solver.capabilities
    if missing:
        names = tuple(sorted(c.value for c in missing))
        return Result(experiment.name, solver.name, n_elements, missing=names)

    # A few steps of the same problem first, so that one-off costs (JIT compilation,
    # library loading) stay out of the timing.
    warm_up = replace(experiment.scenario, duration=experiment.scenario.duration * 1e-3)
    try:
        solver.run(warm_up, n_elements=n_elements, n_frames=2)
        started = time.perf_counter()
        trajectory = solver.run(experiment.scenario, n_elements=n_elements, n_frames=n_frames)
        failure = _divergence(experiment.scenario, trajectory)
    except FloatingPointError as error:
        return Result(experiment.name, solver.name, n_elements, failure=str(error))
    wall_time = time.perf_counter() - started
    if failure:
        return Result(
            experiment.name, solver.name, n_elements, failure=failure, wall_time=wall_time
        )

    metrics = {
        name: float(m(experiment.scenario, trajectory)) for name, m in experiment.metrics.items()
    }
    return Result(
        experiment.name,
        solver.name,
        n_elements,
        wall_time=wall_time,
        metrics=metrics,
        trajectory=trajectory,
    )
