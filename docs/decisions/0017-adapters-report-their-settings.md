# 0017: Adapters report their numerical settings

**Status:** Accepted

## Context

Decision [0001](0001-scenarios-describe-physics-only.md) leaves every numerical
choice to the solver adapter: time step, contact stiffness, damping, integrator.
Many findings turned on those choices (MuJoCo's contact softness, PyElastica's
friction regularisation, dismech's Newton tolerance), yet they lived only in
adapter code. A reader of a result could not see what a solver was run with, so
comparisons between solvers lacked the context the vision asks for.

## Decision

- An adapter may define `settings(scenario, *, n_elements, n_frames) -> dict`:
  the numerical choices it makes for that run, as JSON values. Each built-in
  adapter computes them with the same code its `run` uses, so what is reported
  is what was run.
- The runner records them in every result it simulates (`settings` in
  result.json), and the viewer lists them under the metrics, per solver.

## Consequences

- Settings are derived from the scenario, options and adapter code, which the
  provenance already covers ([0016](0016-reusing-runs-and-running-in-parallel.md)),
  so they add nothing to what makes a run reusable.
- Third-party adapters without `settings` still run; their results show none.
