# 0012: SciPy for references that have no closed form

**Status:** Accepted

## Context

The project's references have so far been closed-form (the elastic catenary,
Euler-Bernoulli deflection, Greenhill's twist) or small hand-written solves (a
Hermite eigenproblem for the pendulum, a Runge-Kutta integration for the
capstan). The core package depended on NumPy alone.

The crossing experiment broke that. A rope with a point load on it is a boundary
value problem with a boundary layer a few centimetres wide across a half-metre
span, which makes shooting hopeless: an error at the support grows by about 1e6
by the middle. Three hand-written attempts — single shooting, multiple shooting,
and a collocation solve with an analytic Jacobian — each converged for some loads
and wandered onto the wrong branch for others.

## Decision

The core package depends on SciPy. References may use it; `solve_bvp` and
`brentq` do the crossing experiment's work.

## Consequences

- One more install for everyone, including anyone who only wants to register a
  solver. SciPy is a standard scientific dependency and PyElastica already
  requires it, so in practice it is already present.
- The alternative was keeping about a hundred lines of fragile numerics of our
  own. Correct references matter more here than a short dependency list: a wrong
  reference does not look wrong, it looks like every solver has the same error.
- References that need a solve should still be checked against something
  independent. The crossing experiment's was checked two ways: against a
  hand-written shooting solution, which agrees to nine digits where both
  converge, and against PyElastica's catenary, whose sag moves toward it, not
  toward the perfectly flexible answer, as resolution rises.
