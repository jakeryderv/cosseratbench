# 0005: Rods can start stretched

**Status:** Accepted

## Context

Scenarios originally gave a rod's initial shape and assumed it unstretched. For
the pendulum, that meant releasing a cable at its rest length with gravity
already acting: it bounced along its length, and the bounce pumped sideways
vibration by parametric resonance. That was correct physics for the scenario as
written, but not the problem the reference describes, which is a cable already
hanging in equilibrium.

## Decision

`Rod.rest_arc_length` optionally gives the unstretched distance along the rod to
each centerline point, so the initial shape can be stretched relative to rest.
Without it, behaviour is unchanged.

## Consequences

- Experiments can start in the equilibrium their reference assumes.
- Adapters build rods from their unstretched geometry and then place them. This
  also fixed PyElastica taking rest lengths from chords of a curved initial shape,
  which had made curved rods slightly short.
- An inextensible solver is built in the stretched shape, with its density scaled
  to keep the rod's mass.
