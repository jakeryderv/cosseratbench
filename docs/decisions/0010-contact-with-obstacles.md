# 0010: Contact with obstacles, and how it is judged

**Status:** Accepted

## Context

Most of the project's scenarios involve contact: wrapping, pulleys, obstacle
courses, knots. Solvers model contact very differently (PyElastica with penalty
springs and viscous-plus-Coulomb friction, MuJoCo with soft constraints and
friction cones), and each needs tuning numbers the physics does not supply, such
as contact stiffness. Fixing those numbers in the scenario would break decision
[0001](0001-scenarios-describe-physics-only.md); leaving them unexamined would
make comparisons unfair in ways no one could see.

## Decision

- A scenario may list fixed obstacles. The first kind is a `Cylinder` (centre,
  axis, radius, length) with a Coulomb friction coefficient between it and rods.
- Each adapter chooses its own contact parameters, and says how in its code. Both
  built-in adapters make contact as stiff as their time step allows and make
  static friction as sticky as they can.
- The benchmark then measures the effects instead of trusting the choices:
  `max_penetration`, how deep any rod node sinks into an obstacle in rod radii, is
  observed on every run with obstacles, and experiments show creep where a rod
  should hold still.

## Consequences

- Contact quality is part of the comparison: a solver that lets rods sink, or
  imitates static friction loosely, shows it.
- Rod-rod contact and self-contact are not yet part of scenarios; MuJoCo's adapter
  keeps cable segments from touching each other, and PyElastica's rod-rod contact
  has no friction.
- Penetration is measured at nodes. A straight segment between two nodes on a
  curved obstacle cuts inside it by a depth set by resolution alone; measuring
  there would hide the solver's contact stiffness behind geometry.
