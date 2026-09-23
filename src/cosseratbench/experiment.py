"""An experiment builds a scenario from its parameters and judges a solver's trajectory."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from cosseratbench.metrics import segment_distance
from cosseratbench.scenario import Cylinder, Scenario, neighbour_elements
from cosseratbench.solver import Capability, Solver
from cosseratbench.trajectory import Trajectory

# Metrics see only the scenario and the trajectory, never the solver, so every
# solver is judged by the same code.
Metric = Callable[[Scenario, Trajectory], float]
# The analytical answer, for drawing beside the solvers': one curve [M, 3] per rod
# it covers, in the scenario's rod order.
Reference = Callable[[Scenario], tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class Parameter:
    """A physical quantity an experiment can be run at, from its ordinary value
    toward harder ones."""

    name: str
    default: float
    values: tuple[float, ...]  # swept, in order; includes the default
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.default not in self.values:
            raise ValueError(f"{self.name}: the default must be one of the swept values")


@dataclass(frozen=True)
class Experiment:
    """A scenario, built from parameter values, and the metrics that judge it.

    ``build`` takes one keyword argument per parameter. ``notes`` says what the
    experiment explores and what to look for; the viewer shows it.
    """

    name: str
    description: str
    build: Callable[..., Scenario]
    metrics: Mapping[str, Metric]
    parameters: tuple[Parameter, ...] = ()
    requires: frozenset[Capability] = frozenset()
    n_elements: int = 50  # default resolution
    n_frames: int = 101  # frames recorded, evenly spaced from start to end
    reference: Reference | None = None
    notes: str = ""

    def values(self, **changes: float) -> dict[str, float]:
        """Every parameter's value: its default, unless changed."""
        known = {p.name for p in self.parameters}
        if unknown := set(changes) - known:
            raise KeyError(
                f"{self.name} has no parameter {sorted(unknown)}; it has {sorted(known)}"
            )
        return {p.name: float(changes.get(p.name, p.default)) for p in self.parameters}

    def scenario_for(self, **changes: float) -> Scenario:
        return self.build(**self.values(**changes))

    @property
    def scenario(self) -> Scenario:
        """The ordinary case: every parameter at its default."""
        return self.scenario_for()

    def save(
        self,
        directory: Path,
        changes: Mapping[str, float] | None = None,
        varied: Mapping[str, float] | None = None,
        defaults: Mapping[str, float] | None = None,
    ) -> None:
        """Write everything a reader of one variant's results needs to interpret them
        without this code: the scenario at ``changes``, and what the variant varies
        (``varied``) from the ordinary case (``defaults``)."""
        directory.mkdir(parents=True, exist_ok=True)
        changes = dict(changes or {})
        scenario = self.scenario_for(**changes)
        reference = (
            None if self.reference is None else [c.tolist() for c in self.reference(scenario)]
        )
        summary = {
            "name": self.name,
            "description": self.description,
            "notes": self.notes,
            "parameters": [asdict(p) for p in self.parameters],
            "values": self.values(**changes),
            "varied": dict(varied or {}),
            "defaults": dict(defaults or {}),
            "scenario": asdict(scenario),
            "rod_lengths": [rod.length for rod in scenario.rods],
            "reference": reference,
        }
        text = json.dumps(summary, default=lambda enum: enum.value)
        (directory / "experiment.json").write_text(text + "\n")


COMPLETED, UNSUPPORTED, DIVERGED = "completed", "unsupported", "diverged"


@dataclass(frozen=True)
class Result:
    experiment: str
    solver: str
    n_elements: int
    values: Mapping[str, float] = field(default_factory=dict)  # the experiment's parameters
    options: Mapping[str, float] = field(default_factory=dict)  # the solver's, as run
    # Capabilities the experiment requires and the solver lacks; if any, it was not run.
    missing: tuple[str, ...] = ()
    failure: str | None = None  # how the run broke down numerically
    diverged_at: float | None = None  # s, when that was first seen, if known
    wall_time: float | None = None  # s
    metrics: Mapping[str, float] = field(default_factory=dict)
    observations: Mapping[str, float] = field(default_factory=dict)
    trajectory: Trajectory | None = None

    @property
    def outcome(self) -> str:
        if self.missing:
            return UNSUPPORTED
        return DIVERGED if self.failure else COMPLETED

    @property
    def supported(self) -> bool:
        return not self.missing

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        def finite(numbers: Mapping[str, float]) -> dict[str, float | None]:
            # JSON has no NaN or infinity; a value that could not be computed is null.
            return {k: v if np.isfinite(v) else None for k, v in numbers.items()}

        summary = {
            "experiment": self.experiment,
            "solver": self.solver,
            "outcome": self.outcome,
            "n_elements": self.n_elements,
            "values": dict(self.values),
            "options": dict(self.options),
            "missing": list(self.missing),
            "failure": self.failure,
            "diverged_at": self.diverged_at,
            "wall_time": self.wall_time,
            "metrics": finite(self.metrics),
            "observations": finite(self.observations),
        }
        (directory / "result.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        if self.trajectory is not None:
            self.trajectory.save(directory / "trajectory.npz")


def _divergence(trajectory: Trajectory) -> tuple[str, float] | None:
    """A solver can go unstable without producing NaNs or raising, so judge the
    trajectory itself: no real rod here doubles the length of one of its segments.
    Returns what gave it away, and the time of the first frame that did."""
    for index, positions in enumerate(trajectory.positions):
        finite = np.isfinite(positions).all(axis=(1, 2))
        segments = np.linalg.norm(np.diff(positions, axis=1), axis=2)
        with np.errstate(invalid="ignore"):
            stretch = (segments / segments[0]).max(axis=1)
        broken = ~finite | ~(stretch <= 2.0)
        if broken.any():
            frame = int(np.argmax(broken))
            what = (
                "non-finite positions"
                if not finite[frame]
                else f"a segment stretched {stretch[frame]:.3g}x"
            )
            return f"rod {index} has {what}", float(trajectory.times[frame])
    return None


def max_strain(scenario: Scenario, trajectory: Trajectory) -> float:
    """Largest stretch or compression of any segment at any time, relative to rest."""
    worst = 0.0
    for rod, positions in zip(scenario.rods, trajectory.positions):
        rest = rod.length / (positions.shape[1] - 1)
        segments = np.linalg.norm(np.diff(positions, axis=1), axis=2)
        worst = max(worst, float(np.abs(segments / rest - 1.0).max()))
    return worst


def max_penetration(scenario: Scenario, trajectory: Trajectory) -> float:
    """Deepest any rod node sinks into any obstacle at any time, in rod radii.

    Nodes, not segments: a straight segment between two nodes on a curved obstacle
    cuts inside it by a depth set by resolution alone, which would hide how stiff a
    solver makes contact. NaN when there is nothing to sink into.
    """
    if not scenario.obstacles:
        return float("nan")
    worst = 0.0
    for rod, positions in zip(scenario.rods, trajectory.positions):
        for obstacle in scenario.obstacles:
            if isinstance(obstacle, Cylinder):
                axis = obstacle.unit_axis
                offset = positions - np.asarray(obstacle.center)
                along = offset @ axis
                across = np.linalg.norm(offset - along[..., None] * axis, axis=2)
                within = np.abs(along) <= obstacle.length / 2
                depth = np.where(within, obstacle.radius + rod.radius - across, 0.0)
            else:
                height = (positions - np.asarray(obstacle.point)) @ obstacle.unit_normal
                depth = rod.radius - height
            worst = max(worst, float(depth.max()) / rod.radius)
    return worst


def max_rod_overlap(scenario: Scenario, trajectory: Trajectory) -> float:
    """Deepest any two rod segments overlap at any time, in radii of the thinner rod:
    how far rods pass into each other, or into themselves.

    Segments close to each other along one rod are skipped, as many as it takes to
    bend back on itself, since those are neighbours rather than contact; it is the
    rule PyElastica's self-contact uses. NaN when that leaves no pair to measure, so
    nothing here could touch anything.
    """
    worst = float("nan")
    rods = tuple(zip(scenario.rods, trajectory.positions))
    for i, (rod, positions) in enumerate(rods):
        for j in range(i, len(rods)):
            other, other_positions = rods[j]
            n, m = positions.shape[1] - 1, other_positions.shape[1] - 1
            first, second = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
            if j == i:
                apart = neighbour_elements(rod.radius, rod.length / n)
                keep = second - first >= apart
                first, second = first[keep], second[keep]
            else:
                first, second = first.ravel(), second.ravel()
            if not len(first):
                continue
            touching = rod.radius + other.radius
            unit = min(rod.radius, other.radius)
            for here, there in zip(positions, other_positions):
                distance = segment_distance(
                    here[first], here[first + 1], there[second], there[second + 1]
                ).min()
                depth = max(touching - float(distance), 0.0) / unit
                worst = depth if np.isnan(worst) else max(worst, depth)
    return worst


# Observed on every completed run, whatever the experiment; NaN where it does not apply.
OBSERVATIONS: Mapping[str, Metric] = {
    "max_strain": max_strain,
    "max_penetration": max_penetration,
    "max_rod_overlap": max_rod_overlap,
}


def _needs(scenario: Scenario) -> frozenset[Capability]:
    """What the scenario itself requires of a solver, beyond what the experiment says."""
    return frozenset(
        Capability.CYLINDER_CONTACT if isinstance(o, Cylinder) else Capability.PLANE_CONTACT
        for o in scenario.obstacles
    )


def run(
    experiment: Experiment,
    solver: Solver,
    *,
    values: Mapping[str, float] | None = None,
    options: Mapping[str, float] | None = None,
    n_elements: int | None = None,
    n_frames: int | None = None,
) -> Result:
    """Run ``solver`` on one variant of ``experiment``: parameters at ``values`` (the
    rest at their defaults). ``options`` records how the solver was configured."""
    values = experiment.values(**(values or {}))
    scenario = experiment.build(**values)
    n_elements = n_elements or experiment.n_elements
    n_frames = n_frames or experiment.n_frames
    identity = {
        "experiment": experiment.name,
        "solver": solver.name,
        "n_elements": n_elements,
        "values": values,
        "options": dict(options or {}),
    }
    missing = (experiment.requires | _needs(scenario)) - solver.capabilities
    if missing:
        return Result(**identity, missing=tuple(sorted(c.value for c in missing)))

    # A few steps of the same problem first, so that one-off costs (JIT compilation,
    # library loading) stay out of the timing.
    warm_up = replace(scenario, duration=scenario.duration * 1e-3)
    try:
        solver.run(warm_up, n_elements=n_elements, n_frames=2)
        started = time.perf_counter()
        trajectory = solver.run(scenario, n_elements=n_elements, n_frames=n_frames)
    except FloatingPointError as error:
        return Result(**identity, failure=str(error), diverged_at=getattr(error, "time", None))
    except NotImplementedError as error:  # something in this scenario is outside its model
        return Result(**identity, missing=(str(error),))
    wall_time = time.perf_counter() - started
    if divergence := _divergence(trajectory):
        failure, when = divergence
        return Result(**identity, failure=failure, diverged_at=when, wall_time=wall_time)

    return Result(
        **identity,
        wall_time=wall_time,
        metrics={name: float(m(scenario, trajectory)) for name, m in experiment.metrics.items()},
        observations={name: float(o(scenario, trajectory)) for name, o in OBSERVATIONS.items()},
        trajectory=trajectory,
    )
