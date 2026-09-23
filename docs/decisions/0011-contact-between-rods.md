# 0011: Contact between rods, and saying when it can happen

**Status:** Accepted

## Context

Decision [0010](0010-contact-with-obstacles.md) gave rods fixed obstacles to
touch. Six of the scenarios the project is growing toward — plectonemes, piles,
two ropes, knots, coiling, packing — need rods to touch each other and
themselves instead.

Two things made this more than switching a flag on.

Friction differs between the solvers. PyElastica's `RodRodContact` and
`RodSelfContact` take a stiffness and a damping and nothing else: rods that touch
each other slide freely however rough they are. MuJoCo's segments are ordinary
geoms and carry friction like anything else. An experiment that needs rods to
grip therefore cannot be run by both.

Self-contact is expensive. PyElastica checks every pair of a rod's elements every
step with no broadphase, which cost 2 to 5 times the rest of the step on
experiments where nothing ever touched. Since the benchmark reports wall time,
paying that everywhere would misreport what a solver costs, and would only fall
on the solver whose contact code has no broadphase.

## Decision

- Rods always collide with each other. Nothing in a scenario turns that off.
- `Scenario.self_contact` says whether a rod in this scenario can reach *itself*.
  It describes the scenario, not the numerics: a cable hanging between two
  supports cannot double back on itself, and saying so saves a solver looking.
- `Rod.friction` is the Coulomb coefficient where a rod touches another rod or
  itself. Where two things that carry friction touch, the larger applies.
- Two capabilities go with this. `ROD_CONTACT` is claimed by both built-in
  solvers; `ROD_FRICTION` only by MuJoCo, so an experiment that needs rods to
  grip reports PyElastica as unsupported rather than quietly running it
  frictionless.
- `max_rod_overlap` is observed on every run, whether or not the scenario asked
  for self-contact: the deepest two rod segments pass into each other, in radii
  of the thinner one.
- What counts as a rod touching itself is one shared rule,
  `scenario.neighbour_elements`: elements closer together along a rod than it
  takes to bend back through a half-turn of its own radius are neighbours, not
  contact. It is PyElastica's rule; the MuJoCo adapter excludes the same pairs, so
  the two solvers and the measurement agree.

## Consequences

- A scenario that gets `self_contact` wrong is caught rather than hidden:
  `max_rod_overlap` is measured either way, so a rod that passes through itself
  shows up as an overlap of about two radii. That is how the twist experiment was
  found to grow a plectoneme late in its run, which it now simulates properly at
  about a third more cost.
- Experiments that need friction between rods — knots, piles — will be scored
  against MuJoCo alone until PyElastica grows it, or until the benchmark ships its
  own contact for rods that lack it.
- Both adapters now make rod contact as stiff as their step allows, as they
  already did for obstacles. At MuJoCo's default a rope sank most of a radius into
  the rope it rested on.
