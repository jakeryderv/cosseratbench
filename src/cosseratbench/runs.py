"""Running experiments: which runs a request means, which of them are already saved,
and running the rest, one after another or several at once.

A saved run is reused when everything its trajectory depended on is unchanged
(its provenance, resolution, frames and solver options; see
``cosseratbench.provenance``). Metrics are not part of that: ``rescore``
recomputes them from saved trajectories without simulating anything.

Runs in parallel are timed while competing for the machine, so each records how
many ran at once, and each is held to one thread of numerical work so that they
compete as little as possible.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path

from cosseratbench import registry, variations
from cosseratbench.experiment import (
    COMPLETED,
    DIVERGED,
    UNSUPPORTED,
    Experiment,
    finite,
    run,
    score,
)
from cosseratbench.provenance import provenance, scenario_digest
from cosseratbench.trajectory import Trajectory

Report = Callable[[str], None]

# Libraries that start a pool of threads for numerical work. With several runs at
# once, each keeps to one, unless the user has said otherwise.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass(frozen=True)
class Task:
    """One run: a solver on one variant of an experiment, saved in ``directory``."""

    experiment: str
    variant: variations.Variant
    solver: str
    n_elements: int | None  # None: the experiment's default
    directory: Path
    label: str


def describe(summary: Mapping) -> str:
    """One line for a result, as saved in result.json."""
    if summary["outcome"] == UNSUPPORTED:
        return "unsupported (needs " + ", ".join(summary["missing"]) + ")"
    if summary["outcome"] == DIVERGED:
        when = summary.get("diverged_at")
        at = f" at t = {when:.3g} s" if when is not None else ""
        return f"diverged{at}: {summary['failure']}"
    metrics = "  ".join(
        f"{name}={value:.3e}" if value is not None else f"{name}=n/a"
        for name, value in summary["metrics"].items()
    )
    parallel = f" ({summary['jobs']} at once)" if summary.get("jobs", 1) > 1 else ""
    return f"{summary['wall_time']:6.1f}s{parallel}  {metrics}"


def saved(task: Task, experiment: Experiment, solver: object) -> dict | None:
    """The saved result of ``task``, if there is one its trajectory still stands for."""
    path = task.directory / "result.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    scenario = experiment.scenario_for(**task.variant.parameters)
    expected = {
        "n_elements": task.n_elements or experiment.n_elements,
        "n_frames": experiment.n_frames,
        "options": dict(task.variant.options),
        "provenance": provenance(scenario, solver),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return None
    if summary["outcome"] == COMPLETED and not (task.directory / "trajectory.npz").exists():
        return None
    return summary


def _execute(task: Task, jobs: int) -> str:
    """Run one task and save it; the line describing it. Runs in a worker process when
    several go at once, so it takes names rather than objects."""
    experiment = registry.load_experiment(task.experiment)
    solver = registry.load_solver(task.solver, **task.variant.options)
    result = run(
        experiment,
        solver,
        values=task.variant.parameters,
        options=task.variant.options,
        n_elements=task.n_elements,
    )
    result = replace(result, jobs=jobs)
    result.save(task.directory)
    return describe(result.summary())


@contextlib.contextmanager
def _one_thread_each() -> Iterator[None]:
    """Worker processes inherit the environment: each does its numerical work on one
    thread, unless the user has set otherwise."""
    before = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    for name in _THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def plan(
    experiments: Iterable[Experiment],
    solvers: Iterable[str],
    vary: Iterable[str],
    n_elements: int | None,
    out: Path,
    *,
    force: bool = False,
    report: Report = print,
) -> list[Task]:
    """The runs a request means that are not already saved. Writes each variant's
    experiment.json, and reports the runs it skips: saved, or solver not installed."""
    experiments, solvers, vary = list(experiments), list(solvers), list(vary)
    # Each name applies where it means something; one that means nothing anywhere is a typo.
    sweepable = {e.name: variations.sweepable(e) for e in experiments}
    for name in vary:
        if name != "all" and not any(name in names for names in sweepable.values()):
            raise KeyError(f"nothing to vary called {name!r}; see `cosseratbench list`")

    tasks = []
    for experiment in experiments:
        defaults = variations.defaults(experiment)
        applicable = [n for n in vary if n == "all" or n in sweepable[experiment.name]]
        for variant in variations.variants(experiment, applicable):
            directory = out / experiment.name / variant.key
            experiment.save(directory, variant.parameters, variant.varied, defaults)
            for name in solvers:
                label = f"{experiment.name:12s} {variant.key:24s} {name:12s}"
                try:
                    solver = registry.load_solver(name, **variant.options)
                except ImportError as error:
                    report(f"{label} not installed ({error.name})")
                    continue
                task = Task(
                    experiment=experiment.name,
                    variant=variant,
                    solver=name,
                    n_elements=variant.n_elements or n_elements,
                    directory=directory / name,
                    label=label,
                )
                if not force and (summary := saved(task, experiment, solver)) is not None:
                    report(f"{label} n={summary['n_elements']:<4d} {describe(summary)}  [saved]")
                    continue
                tasks.append(task)
    return tasks


def execute(tasks: list[Task], *, jobs: int = 1, report: Report = print) -> None:
    """Run ``tasks``, ``jobs`` at a time, reporting each as it finishes. A run whose
    adapter raises is reported and the rest carry on; if any did, raises at the end."""
    crashed = []

    def finished(task: Task, line: str | None, error: BaseException | None) -> None:
        n = task.n_elements or registry.load_experiment(task.experiment).n_elements
        if error is None:
            report(f"{task.label} n={n:<4d} {line}")
        else:
            crashed.append(task.label)
            report(f"{task.label} n={n:<4d} crashed: {type(error).__name__}: {error}")

    if jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            try:
                line = _execute(task, 1)
            except Exception as error:  # noqa: BLE001 -- an adapter bug: report it, carry on
                finished(task, None, error)
            else:
                finished(task, line, None)
    else:
        at_once = min(jobs, len(tasks))
        # Spawned, not forked: the parent has solver libraries loaded, some with threads.
        with (
            _one_thread_each(),
            ProcessPoolExecutor(at_once, mp_context=get_context("spawn")) as pool,
        ):
            pending = {pool.submit(_execute, task, at_once): task for task in tasks}
            try:
                for future in as_completed(pending):
                    task = pending[future]
                    try:
                        line = future.result()
                    except Exception as error:  # noqa: BLE001 -- as above
                        finished(task, None, error)
                    else:
                        finished(task, line, None)
            except KeyboardInterrupt:
                pool.shutdown(cancel_futures=True)
                raise
    if crashed:
        raise RuntimeError(f"{len(crashed)} run(s) crashed; see above")


def rescore(results: Path, names: Iterable[str] = (), *, report: Report = print) -> None:
    """Recompute the metrics and observations of saved runs from their trajectories,
    and each variant's experiment.json (reference, notes) with them.

    A run whose scenario the experiment no longer builds is left as it was and
    reported: its trajectory answers a different question, and needs rerunning.
    """
    names = set(names)
    for description in sorted(results.glob("*/*/experiment.json")):
        directory = description.parent
        existing = json.loads(description.read_text())
        if names and existing["name"] not in names:
            continue
        try:
            experiment = registry.load_experiment(existing["name"])
        except KeyError:
            report(f"{existing['name']}: no such experiment now; left as it was")
            continue
        parameters = {p.name for p in experiment.parameters}
        changes = {k: v for k, v in existing["varied"].items() if k in parameters}
        experiment.save(directory, changes, existing["varied"], variations.defaults(experiment))
        scenario = experiment.scenario_for(**changes)
        digest = scenario_digest(scenario)
        for path in sorted(directory.glob("*/result.json")):
            summary = json.loads(path.read_text())
            label = f"{experiment.name:12s} {directory.name:24s} {summary['solver']:12s}"
            if summary["outcome"] != COMPLETED:
                continue
            if summary.get("provenance", {}).get("scenario") != digest:
                report(f"{label} its scenario has changed since it ran; rerun it")
                continue
            trajectory = Trajectory.load(path.parent / "trajectory.npz")
            metrics, observations = score(experiment, scenario, trajectory)
            summary["metrics"], summary["observations"] = finite(metrics), finite(observations)
            path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            report(f"{label} {describe(summary)}")
