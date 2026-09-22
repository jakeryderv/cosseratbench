# 0002: Metrics judge trajectories, not solvers

**Status:** Accepted

## Context

If each solver reported its own measurements, differences in how they measure
would be indistinguishable from differences in what they simulate.

## Decision

A solver returns a `Trajectory`: node positions over time, in a common format.
Every metric is a function of the scenario and the trajectory alone, computed by
the benchmark, so all solvers are judged by the same code. The runner also checks
each trajectory for divergence (non-finite values, a segment stretched past
twice its length) rather than trusting the solver to notice.

## Consequences

- Anything a metric needs must be recoverable from positions. Forces are not:
  tension is visible only through stretch, which an inextensible model does not
  have. Twist is not visible at all, which limits the viewer and datasets; adding
  material frames to trajectories is a likely extension.
- Metrics must be robust to motion they were not written for. The pendulum's
  first period measurement, from zero crossings, broke on a small fast vibration
  and was replaced by a fit over every frame.
- Results carry a failure reason when a run produced no usable trajectory, so a
  diverged solver is a reported result, not a crash.
