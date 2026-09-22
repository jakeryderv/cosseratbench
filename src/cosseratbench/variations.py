"""Variants of an experiment: its ordinary case, and one change at a time from it.

A variant changes one thing: a physical parameter the experiment declares, or a
solver option (decision 0008). Changing one thing at a time keeps the number of
runs proportional to the values swept, and each difference attributable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from cosseratbench.experiment import Experiment
from cosseratbench.solver import TIME_STEP_SCALE

RESOLUTION = "resolution"  # elements per rod
OPTIONS = (RESOLUTION, TIME_STEP_SCALE)
_TIME_STEP_SCALES = (2.0, 4.0)  # past the default of 1: toward and beyond stability limits


@dataclass(frozen=True)
class Variant:
    parameters: Mapping[str, float] = field(default_factory=dict)  # changed from default
    n_elements: int | None = None  # None: the experiment's default
    options: Mapping[str, float] = field(default_factory=dict)  # solver constructor options

    @property
    def varied(self) -> dict[str, float]:
        """What this variant changes, by name: empty for the ordinary case."""
        varied = dict(self.parameters) | dict(self.options)
        if self.n_elements is not None:
            varied[RESOLUTION] = self.n_elements
        return varied

    @property
    def key(self) -> str:
        """A directory name for the variant."""
        return ",".join(f"{k}={v:g}" for k, v in sorted(self.varied.items())) or "default"


def sweepable(experiment: Experiment) -> list[str]:
    return [p.name for p in experiment.parameters] + list(OPTIONS)


def variants(experiment: Experiment, vary: Iterable[str] = ()) -> list[Variant]:
    """The ordinary case, then each non-default value of each name in ``vary``, which
    may name the experiment's parameters, the solver options, or be "all"."""
    names = list(vary)
    if "all" in names:
        names = sweepable(experiment)
    found = [Variant()]
    parameters = {p.name: p for p in experiment.parameters}
    for name in names:
        if name in parameters:
            p = parameters[name]
            found += [Variant(parameters={name: v}) for v in p.values if v != p.default]
        elif name == RESOLUTION:
            n = experiment.n_elements
            found += [Variant(n_elements=m) for m in (max(n // 2, 2), n * 2)]
        elif name == TIME_STEP_SCALE:
            found += [Variant(options={name: s}) for s in _TIME_STEP_SCALES]
        else:
            raise KeyError(
                f"{experiment.name} cannot vary {name!r}; it can vary {sweepable(experiment)}"
            )
    return found


def defaults(experiment: Experiment) -> Mapping[str, float]:
    """The value each sweepable name takes in the ordinary case."""
    values = {p.name: p.default for p in experiment.parameters}
    return values | {RESOLUTION: experiment.n_elements, TIME_STEP_SCALE: 1.0}
