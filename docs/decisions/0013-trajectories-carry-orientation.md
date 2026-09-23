# 0013: Trajectories carry orientation

**Status:** Accepted

## Context

A trajectory was node positions over time and nothing else. That is enough to
judge shape, but a rod's cross-section can turn without its centerline moving,
and positions alone cannot show it. Twist was invisible in the viewer, and the
twist experiment could only infer buckling from sideways growth of the
centerline rather than measure the twist a rod holds. Decision
[0009](0009-kinds-of-outcome.md) promised energy drift as a standard
observation, which positions alone cannot supply either.

Both built-in solvers carry a material frame with every element: PyElastica as
directors, MuJoCo as the orientation of each segment's body. Adding a third
solver would mean a third adapter to change, so the format had to be settled
first.

## Decision

A trajectory carries, for each rod, one unit **director** per element alongside
the node positions: the material direction `d1` of a Cosserat rod, perpendicular
to the element. At t = 0 it is `Rod.directors`: the rod's `normal` projected
perpendicular to the first element and carried along the initial centerline by
parallel transport, so a rod starts untwisted however it is bent, and every
solver reports the same material line. With the tangent it gives the whole
orientation of the cross-section.

Velocities are not carried. Energy drift stays deferred; when it is wanted it
will need velocities and angular velocities too, and that is a separate change.

The viewer lays each rod's tube out from its directors and draws a darker stripe
along them, so twist is visible. `metrics.twist_angles` measures twist from them:
the turn of the director from one element to the next beyond what the bend at
the node carries it through, whose sum along a rod is the rod's total twist.

## Consequences

- The adapter contract changes: `Trajectory` takes `directors` as well as
  `positions`, and every solver must provide them. Results written before this
  cannot be loaded; rerun them.
- A solver with no material frame (a chain of point masses, say) can only
  synthesise directors, by parallel transport along each frame's centerline,
  and will report no twist. That is a limit of its model and should be said in
  its adapter.
- Saved trajectories gain `rod{i}_directors`; the site's binary holds every rod's
  positions and then every rod's directors.
- Twist can now be measured directly. The twist experiment keeps its
  growth-rate measurement, which is what the reference predicts, but the
  directors let a held twist be checked against what the clamp imposed.
