"""Discovery of solvers and experiments through Python entry points.

A third-party package takes part by declaring, in its own ``pyproject.toml``::

    [project.entry-points."cosseratbench.solvers"]
    mysolver = "mypackage.adapter:MySolver"

    [project.entry-points."cosseratbench.experiments"]
    myexperiment = "mypackage.experiments:my_experiment"

A solver entry point names a class constructible with no arguments; an
experiment entry point names an ``Experiment`` instance. Entry points are loaded
on demand, so a solver whose backend is not installed costs nothing until used.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

from cosseratbench.experiment import Experiment
from cosseratbench.solver import Solver

SOLVERS = "cosseratbench.solvers"
EXPERIMENTS = "cosseratbench.experiments"


def _find(group: str, name: str) -> EntryPoint:
    matches = entry_points(group=group, name=name)
    if not matches:
        raise KeyError(f"no {group!r} entry point named {name!r}; available: {names(group)}")
    return next(iter(matches))


def names(group: str) -> list[str]:
    return sorted(ep.name for ep in entry_points(group=group))


def load_solver(name: str) -> Solver:
    return _find(SOLVERS, name).load()()


def load_experiment(name: str) -> Experiment:
    return _find(EXPERIMENTS, name).load()
