# 0006: Clamped ends can be driven

**Status:** Accepted

## Context

Twisting, pulling, feeding and whipping are all prescribed motions of a rod's
end. The alternatives considered were named drive types (`Twist(rate)`,
`Pull(speed)`), which need a new type and adapter support for every experiment,
and Python functions of time, which cannot be written to JSON or shown in the
viewer.

## Decision

A clamped end may carry a `Motion`: sampled times, each with the end's
displacement and rotation (a rotation vector) relative to where it started,
interpolated linearly between samples and held after the last. With
`slides_along`, the end is instead free along one axis, which it may only turn
about, so a load can set the tension on a twisted rod.

## Consequences

- Motions are data: serialisable, shown in the viewer, and independent of any
  solver's discretisation. Smooth motions are sampled finely.
- PyElastica drives the end node and element frame with matching velocities.
  MuJoCo welds the end to a mocap body it moves each step. MuJoCo's welds turned
  out to be the weak point for large twists; see the findings.
