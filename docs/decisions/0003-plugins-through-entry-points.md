# 0003: Solvers and experiments are plugins

**Status:** Accepted

## Context

The project should grow beyond the solvers and experiments it ships with, and
people should be able to compare their own solver without forking it.

## Decision

Solvers and experiments are discovered through Python entry points
(`cosseratbench.solvers`, `cosseratbench.experiments`). The built-ins register
the same way as third-party packages. Entry points load on demand, so a solver
whose backend is not installed costs nothing until it is used.

## Consequences

- A solver is a small adapter class implementing `run(scenario, n_elements,
  n_frames) -> Trajectory` and declaring its capabilities.
- The adapter interface is a public contract; changing it breaks other people's
  packages, so it should change rarely and deliberately.
