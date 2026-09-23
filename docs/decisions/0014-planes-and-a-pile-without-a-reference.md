# 0014: Planes to rest on, and a pile without a reference

**Status:** Accepted

## Context

Six experiments in, every one was a validation against a reference except the
capstan, and nothing exercised self-contact: a rod touching itself in many
places at once, persistently, under friction. The scenarios the project is for
(piles, knots, packing, winding) are all of that kind, and none of them has a
reference. Decision [0007](0007-validation-and-exploration.md) allows
experiments without one, but the machinery — observations instead of scores,
self-contact in both adapters, `max_rod_overlap` — had not been tried on a
problem that needs it.

A pile needs a floor. Rods could only touch cylinders
([0010](0010-contact-with-obstacles.md)).

## Decision

- `Plane` joins `Cylinder` as an obstacle: a fixed, unbounded plane with a
  point, a normal toward the side the rods are on, and a friction coefficient
  under the same rule as cylinders. `max_penetration` measures sinking into
  planes as it does into cylinders. Saved scenarios say which shape each
  obstacle is (`kind`), so the viewer and any other reader can tell them apart.
- The **pile** experiment lowers a rope from a clamp onto a floor faster than
  it can lie down, and looks at the heap. It has no reference and no score.
  Its metrics describe the pile: how tall, how wide, how many places the rope
  touches itself, and whether it is still moving at the end. The standard
  observations say how far the rope passed into itself and into the floor.
- The rope is fed from a moving clamp rather than dropped. A dropped rope is
  chaotic from the first bounce, so its pile says little about the solver; a
  fed rope buckles and coils where it meets the floor, a phenomenon with a
  literature of its own, and gives piles that can be compared. A small bow in
  the hanging rope decides which way it folds first, so that solvers are not
  compared on their rounding noise.
- The heap at rest is the result (`quasi_static=True`), so solvers may add
  dissipation to reach it. Run without any, an elastic rope with no material
  damping went on jiggling on the floor at a third of a rod length per second
  after the clamp had stopped, and its pile metrics described a rope still in
  motion. The dissipation the adapters add is mild next to the feed.
- The clamp stops well clear of the floor. Lowered into the heap it crushed the
  standing loop under it and flung the rope out flat at over a metre per second,
  which said nothing about piles.

## Consequences

- Experiments that need friction between rods (knots) are still MuJoCo-only
  ([0011](0011-contact-between-rods.md)); the pile does not require it, so
  PyElastica runs it with its coils sliding freely over each other, and the
  difference is a finding rather than a missing result.
- PyElastica's floor contact has real static friction (an element slower than a
  threshold is held), unlike its cylinder contact, which regularises it. The
  adapter says so.
- Pile metrics leave out the length of rope still hanging from the clamp at the
  end, since the clamp stops a little above the floor.
