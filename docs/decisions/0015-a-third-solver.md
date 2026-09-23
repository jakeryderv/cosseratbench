# 0015: A third solver, and which

**Status:** Accepted

## Context

With two solvers, most findings were "MuJoCo cannot do this": its cable is
inextensible, shearless, and diverges under moderate twist at clamped ends.
That is real information, but a benchmark of two, one of them a poor fit,
reads as a showcase for the other. The vision names discrete elastic rods
(DER), SOFA's Cosserat plugin, Project Chrono and others as candidates.

None of the DER implementations is on PyPI. SOFA and Chrono need binary
installs outside `pip`. [dismech-python](https://github.com/StructuresComp/dismech-python)
is a pure-Python DER (Bergou et al.) with implicit integration and the
implicit contact model (IMC) with friction between rods, installable from git.
It is pre-1.0 with no documentation, requires Python 3.13, and is GPL-3.0.

## Decision

- dismech-python is the third solver, through an adapter like the other two.
- It is not a dependency of this package: a git dependency cannot be published
  to PyPI, and a GPL library should not be pulled in by an Apache-2.0 package's
  extras. It is a `uv` dependency group (`--group dismech`), pinned to a commit,
  installed by those who want it; the adapter is registered regardless and
  reports itself not installed otherwise. The project's own environment and CI
  move to Python 3.13 so that it is tested.
- Contact with each kind of fixed obstacle is a capability (`CYLINDER_CONTACT`,
  `PLANE_CONTACT`) that a scenario requires by having one. dismech has a level
  floor and no cylinders, so the capstan reports it as unsupported rather than
  failing.
- An adapter may raise `NotImplementedError` from `run` for a scenario within its
  capabilities but outside its model (a floor that is not level, rods of two
  radii), and the run is reported as unsupported with that reason.

## Consequences

- Wherever dismech's model lacks something (shear, cylinders, a tilted floor),
  the result says so rather than running something else.
- The adapter corrects two bugs in the pinned commit: gravity applied twice, and
  floor friction that crashes on a rod partly on the floor. It corrects only what
  would otherwise crash or give a plainly wrong answer to a check with a known
  result (a free fall), and says so in the findings; what remains wrong (floor
  friction that cannot stick) is reported, not worked around. The pin is what
  makes the corrections safe, and moving it means checking them again.
- dismech takes whatever shape its stepper is built in as stress-free, so the
  adapter builds every rod straight and unstretched and only then moves it to its
  start. Its contact pairs follow the benchmark's neighbour rule
  ([0011](0011-contact-between-rods.md)), not dismech's own.
- Its cost is different in kind: an implicit step is far larger than an explicit
  one but each needs a Newton solve with a dense Jacobian, so it grows with the
  cube of the number of elements.
