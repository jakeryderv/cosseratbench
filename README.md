# cosseratbench

***A visual benchmark suite for evaluating rope, cable, and Cosserat rod simulation across deformation, contact, and dynamic stress cases.***

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

Early. Two experiments (catenary, cantilever) run against two solvers
([PyElastica](https://github.com/GazzolaLab/PyElastica) and
[MuJoCo](https://mujoco.org)'s cable plugin), each scored against an analytical
reference. Contact, driven boundaries and the web viewer are not built yet.

## Usage

```sh
uv sync --all-extras        # from a clone; installs both solver backends
uv run cosseratbench list
uv run cosseratbench run    # every experiment x every solver, saved under results/
uv run cosseratbench run catenary --solver pyelastica --n-elements 100
```

Each run writes `results/<experiment>/<solver>/result.json` (metrics, wall time)
and `trajectory.npz` (node positions over time).

## How it fits together

- A **scenario** describes the physics and nothing else: geometry, material,
  boundary conditions, loads, gravity, duration, all in SI units. Time steps,
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

    def run(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> Trajectory:
        ...
```

The built-in solvers and experiments register the same way; see
`src/cosseratbench/solvers` and `src/cosseratbench/experiments`.

