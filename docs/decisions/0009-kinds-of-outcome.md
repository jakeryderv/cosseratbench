# 0009: Outcomes say what kind of thing happened

**Status:** Accepted

## Context

The project treats failures as evidence, and asks investigations to tell apart
numerical breakdown, physically plausible events, and behaviour a model cannot
represent. A result currently says only that a run was unsupported (missing
capabilities) or failed (with a message). Each kind has already occurred:

- **Numerical breakdown:** MuJoCo diverging when a doubly clamped cable holds
  about 5 rad of twist.
- **Physically plausible event:** an unstretched cable released under gravity
  bouncing and pumping sideways vibration, which first looked like a solver bug.
- **Not representable:** MuJoCo's cable cannot stretch, so it cannot show that
  bounce or buckle a rod held at fixed length.

## Decision

A result's outcome is one of: **completed**, **unsupported** (the solver's model
cannot represent something the experiment needs), or **diverged** (numerical
breakdown, with when and how). Physically plausible surprises are not outcomes
but findings, documented with the experiment. Completed runs also report
standard **violations** where they apply: penetration of obstacles or of the rod
itself, stretch in a rod that should not stretch, and energy drift where none is
expected.

## Consequences

- The viewer can show at a glance why a solver has no result.
- Violations make contact stiffness and similar solver choices visible rather
  than trusted, which is the fairness rule for contact.
- As built, results record `outcome` and, for a divergence, the time it was first
  seen. Observations are `max_strain`, the largest stretch or compression of
  any segment, `max_penetration` on runs with obstacles, and `max_rod_overlap`
  ([0011](0011-contact-between-rods.md)). Energy drift is not yet observed:
  trajectories carry positions and directors ([0013](0013-trajectories-carry-orientation.md))
  but not velocities, so it is deferred until they do.
