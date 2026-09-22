# 0001: Scenarios describe physics only

**Status:** Accepted

## Context

Comparing solvers on "the same problem" is only meaningful if the problem is
stated independently of any solver. Solvers differ in everything numerical: time
integration, discretisation, how contact and damping are modelled. A scenario
written in one solver's terms (its time step, its contact stiffness) quietly
favours that solver and cannot be run faithfully by another.

## Decision

A `Scenario` states the physical problem only, in SI units: geometry, material,
end conditions, loads, driven motions, gravity, duration, and whether the result
is an equilibrium (`quasi_static`) or the motion itself. Time steps, element
counts, integrators, contact stiffness and added damping belong to each solver
adapter. Resolution is passed to a run separately, as the one numerical input
every solver shares.

## Consequences

- Adapters must translate physics into their own numerics, and document choices
  that affect results: `quasi_static` lets a solver add damping, and each adapter
  says how much.
- Some physics a solver cannot represent at all (an inextensible cable cannot
  stretch). That is declared through capabilities and reported, not worked around.
- Pushing solvers through their numerical settings is still wanted; it happens in
  a separate layer of solver options, never in the scenario
  ([0008](0008-solver-options-and-variations.md)).
