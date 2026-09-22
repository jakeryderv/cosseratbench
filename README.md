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

Early. Four experiments (catenary, cantilever, pendulum, twist) run against two
solvers ([PyElastica](https://github.com/GazzolaLab/PyElastica) and
[MuJoCo](https://mujoco.org)'s cable plugin), each scored against an analytical
or high-accuracy numerical reference, and a web viewer plays the results back.
Rod ends can be driven: moved, turned, or left free to slide under a load.
Contact is not built yet.

Known solver limit: MuJoCo cannot run the twist experiment. A cable clamped at
both ends diverges once it holds about 5 rad of twist; see
[findings](docs/findings.md) for this and other solver behaviour.

## Usage

```sh
uv sync --all-extras        # from a clone; installs both solver backends
uv run cosseratbench list
uv run cosseratbench run    # every experiment x every solver, saved under results/
uv run cosseratbench run catenary --solver pyelastica --n-elements 100
uv run cosseratbench run catenary --vary span --vary time_step_scale
uv run cosseratbench run --vary all   # every parameter and solver option, one at a time
uv run cosseratbench view   # open the results in a browser
```

Each run writes `results/<experiment>/<variant>/<solver>/result.json` (outcome,
metrics, observations, wall time) and `trajectory.npz` (node positions over time),
next to an `experiment.json` describing the problem. A variant is the ordinary
case (`default`) or one change from it: a physical parameter the experiment
declares, the resolution, or `time_step_scale`, which multiplies the time step
each solver would choose. `cosseratbench list` shows what each experiment can vary. Wall time excludes a short warm-up run, so one-off costs
such as JIT compilation do not count against a solver.

## Viewer

`cosseratbench view` serves an interactive page for whatever is under `results/`:
3D playback of every solver on one timeline, overlaid or split into panes that
share a camera, with the analytical reference drawn where one exists; the metrics
table; the speed of the fastest node over time; and the physical scenario.

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
  over time, at a requested resolution. It declares the physics it models
  (`Capability`), and an experiment that needs more is reported as unsupported
  rather than run.
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

The built-in solvers and experiments register the same way; see
`src/cosseratbench/solvers` and `src/cosseratbench/experiments`.

