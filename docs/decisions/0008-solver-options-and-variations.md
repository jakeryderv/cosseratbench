# 0008: Variations sweep physics and solver options separately

**Status:** Accepted

## Context

The project should let people change conditions and watch behaviour develop:
stiffer, faster, heavier, tighter, more contact, and more demanding numerical
settings. Scenarios describe physics only ([0001](0001-scenarios-describe-physics-only.md)),
and the viewer is static ([0004](0004-static-viewer.md)).

## Decision

- An experiment declares **parameters**: named physical quantities with an
  ordinary value and a range toward harder cases. Its scenario is built from
  them.
- Numerical settings that are fair to vary across solvers (resolution, a time
  step scale, contact stiffness where a solver has one) are **solver options**,
  kept apart from the scenario and passed to adapters.
- The runner sweeps chosen values of both and records each run with the values
  that made it. The viewer picks among precomputed runs with controls for each
  swept parameter.

## Consequences

- One experiment covers its ordinary case and its stress cases, and the
  parameter sweep in the project's table becomes a property of every experiment
  rather than an experiment of its own.
- Results gain an extra level: each run is identified by its parameter and
  option values, not only by experiment and solver.
- Sweeps multiply run time; choosing a small set of values per parameter is part
  of designing an experiment.
- As built, variants change one thing at a time from the ordinary case, which
  keeps runs proportional to the values swept and each difference attributable;
  interactions between parameters are not explored. The solver options are
  resolution and `time_step_scale`, a multiplier on each adapter's own choice of
  time step.
