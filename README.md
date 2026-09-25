# cosseratbench

***A tool for exploring, comparing, and stress-testing simulations of slender flexible bodies.***

What works well, what becomes difficult, and what breaks, for which solver,
under which conditions. See the [vision](docs/vision.md) for the idea behind it,
and [findings](docs/findings.md) for what it has shown so far.

The scenarios it is growing toward:

| #  | Experiment                         | What it tests                                                           |
| -- | ---------------------------------- | ----------------------------------------------------------------------- |
| 1  | **Twist → plectoneme / knot**      | twist, bending, buckling, extreme curvature, self-contact               |
| 2  | **Rope drop / pile**               | gravity, friction, chaotic self-contact, many simultaneous contacts     |
| 3  | **Catenary / hanging cable**       | basic gravity, tension, sag, stretch; good sanity/validation case       |
| 4  | **Cantilever bend + twist**        | isolated bending stiffness, torsion, large deformation                  |
| 5  | **Pendulum / swinging cable**      | dynamics, inertia, damping, oscillation                                 |
| 6  | **Snap / whip test**               | very fast motion, high curvature, timestep stability                    |
| 7  | **Cylinder wrap / capstan**        | rod-cylinder contact, friction, sliding, tension transfer               |
| 8  | **Pulley / sheave**                | moving contact, bending around small radius, tension under motion       |
| 9  | **Obstacle course**                | repeated contact against cylinders/planes/spheres, sliding and snagging |
| 10 | **Two-rope interaction**           | rod-rod contact, crossing, rubbing, entanglement                        |
| 11 | **Loop / knot tightening**         | persistent dense self-contact and friction under increasing tension     |
| 12 | **Compression / coiling**          | rope pushed into a confined area; buckling and pile formation           |
| 13 | **Container packing**              | rope fed into a box/cylinder; dense 3D self-contact                     |
| 14 | **Parameter/extreme stress sweep** | deliberately push stiffness, friction, speed, resolution, timestep      |

## Status

Early. Seven experiments run against three solvers
([PyElastica](https://github.com/GazzolaLab/PyElastica),
[MuJoCo](https://mujoco.org)'s cable plugin, and
[dismech-python](https://github.com/StructuresComp/dismech-python)'s discrete
elastic rods): five validations scored against an analytical or high-accuracy
numerical reference (catenary, cantilever, pendulum, twist, crossing) and two
explorations without one, of contact with friction (capstan) and of a rope
heaping up on a floor in dense self-contact (pile). A web viewer plays the
results back. Rod ends can be driven: moved, turned, or left free to slide under
a load. Rods touch fixed cylinders and planes, each other, and themselves. What a
solver's model lacks (friction between rods in PyElastica, cylinders in dismech)
makes an experiment that needs it unsupported for that solver rather than run
with something else.

Known solver limit: MuJoCo cannot run the twist experiment. A cable clamped at
both ends diverges once it holds about 5 rad of twist; see
[findings](docs/findings.md) for this and other solver behaviour.

## Usage

```sh
uv sync --all-extras        # from a clone; installs PyElastica and MuJoCo
uv sync --all-extras --group dismech   # and dismech-python, from git (Python 3.13+, GPL-3.0)
uv run cosseratbench list
uv run cosseratbench run    # every experiment x every solver, saved under results/
uv run cosseratbench run catenary --solver pyelastica --n-elements 100
uv run cosseratbench run catenary --vary span --vary time_step_scale
uv run cosseratbench run --vary all   # every parameter and solver option, one at a time
uv run cosseratbench run --vary all -j 8   # the same, eight simulations at once
uv run cosseratbench rescore   # recompute metrics from saved trajectories, simulating nothing
uv run cosseratbench view   # open the results in a browser
```

A run whose saved result still stands (same scenario, resolution, solver options,
solver version and adapter code) is reused and reported as `[saved]`; `--force`
reruns it. With `-j`, runs compete for the machine, so each is held to one thread
and records how many ran at once, and their wall times should be read with that in
mind. With dismech installed, name the solvers or experiments you want: twist on
dismech takes hours.

Each run writes `results/<experiment>/<variant>/<solver>/result.json` (outcome,
metrics, observations, wall time, and what the trajectory depended on) and
`trajectory.npz` (node positions and element directors over time), next to an
`experiment.json` describing the problem. A variant is the ordinary
case (`default`) or one change from it: a physical parameter the experiment
declares, the resolution, or `time_step_scale`, which multiplies the time step
each solver would choose. `cosseratbench list` shows what each experiment can vary. Wall time excludes a short warm-up run, so one-off costs
such as JIT compilation do not count against a solver.

## Viewer

`cosseratbench view` serves an interactive page for whatever is under `results/`:
3D playback of every solver on one timeline, overlaid or split into panes that
share a camera, with the analytical reference drawn where one exists and a stripe
along each rod that turns with its material, so twist shows; the metrics table;
the speed of the fastest node over time; and the physical scenario.

`cosseratbench site OUT` writes the same page as static files, for hosting
anywhere (GitHub Pages, for example). The page loads three.js from a CDN, so it
needs a network connection.

## Documentation

- [Vision](docs/vision.md): what the project is for and the principles behind it.
- [Findings](docs/findings.md): what the experiments have shown about each solver,
  and about measuring them.
- [Decisions](docs/decisions/README.md): the choices that shape the project, and why.

## How it fits together

- A **scenario** describes the physics and nothing else: geometry, material,
  boundary conditions, loads, gravity, duration, all in SI units. A rod may
  start stretched, for example already hanging in equilibrium. Time steps,
  element counts, contact stiffnesses and damping coefficients are not part of
  it; they are each solver's business.
- A **solver** adapter turns a scenario into a **trajectory**, node positions
  and a material direction per element over time, at a requested resolution. It
  declares the physics it models (`Capability`), and an experiment that needs
  more is reported as unsupported rather than run.
- An **experiment** pairs a scenario with **metrics**. Metrics see only the
  scenario and the trajectory, so every solver is judged by the same code.

## Adding a solver or an experiment

Both are discovered through entry points, so they can live in your own package:

```toml
[project.entry-points."cosseratbench.solvers"]
mysolver = "mypackage.adapter:MySolver"

[project.entry-points."cosseratbench.experiments"]
myexperiment = "mypackage.experiments:my_experiment"
```

```python
from cosseratbench import Capability, Scenario, Trajectory


class MySolver:
    name = "mysolver"
    capabilities = frozenset({Capability.STRETCH})

    def run(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> Trajectory: ...
```

A trajectory holds each rod's node positions, `[n_frames, n_elements + 1, 3]`, and
one unit director per element, `[n_frames, n_elements, 3]`: the material direction
that starts as `Rod.directors(n_elements)`, perpendicular to the element, and
turns with the rod's cross-section. A solver that breaks down should raise
`cosseratbench.solver.Diverged`, with the simulated time if it knows it; the
runner also catches blow-ups a solver misses.
To take part in time step sweeps, accept a `time_step_scale` keyword in the
constructor and multiply your own choice of step by it.

An experiment builds its scenario from its parameters:

```python
from cosseratbench import Experiment, Parameter


def build(youngs_modulus: float) -> Scenario: ...


my_experiment = Experiment(
    name="myexperiment",
    description="One line for listings.",
    build=build,
    parameters=(Parameter("youngs_modulus", default=1e6, values=(1e5, 1e6, 1e7), unit="Pa"),),
    metrics={"my_metric": my_metric},  # functions of (scenario, trajectory)
    notes="What it explores and what to look for; the viewer shows this.",
)
```

The built-in solvers and experiments register the same way; see
`src/cosseratbench/solvers` and `src/cosseratbench/experiments`.

