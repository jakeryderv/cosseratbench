"""Adapter for PyElastica: Cosserat rods with an explicit symplectic integrator."""

from __future__ import annotations

import math

import elastica as ea
import numpy as np

from cosseratbench.scenario import End, EndCondition, Rod, Scenario
from cosseratbench.solver import Capability
from cosseratbench.trajectory import Trajectory

# Timoshenko shear coefficient PyElastica uses for circular cross-sections.
_SHEAR_COEFFICIENT = 27.0 / 28.0


class _Simulator(ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping):
    pass


class _PointForce(ea.NoForces):
    def __init__(self, force: np.ndarray, node: int) -> None:
        super().__init__()
        self.force = force
        self.node = node

    def apply_forces(self, system, time=0.0) -> None:
        system.external_forces[:, self.node] += self.force


def _directors(nodes: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Material frames [3, 3, n_elements] carried along the centerline by parallel
    transport, starting from ``normal`` projected perpendicular to the first tangent."""
    tangents = np.diff(nodes, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    d1 = normal - np.dot(normal, tangents[0]) * tangents[0]
    d1 /= np.linalg.norm(d1)

    directors = np.empty((3, 3, len(tangents)))
    for i, tangent in enumerate(tangents):
        if i > 0:
            axis = np.cross(tangents[i - 1], tangent)
            sin, cos = np.linalg.norm(axis), np.dot(tangents[i - 1], tangent)
            if sin > 1e-12:
                axis /= sin
                d1 = d1 * cos + np.cross(axis, d1) * sin + axis * np.dot(axis, d1) * (1.0 - cos)
        directors[0, :, i] = d1
        directors[1, :, i] = np.cross(tangent, d1)
        directors[2, :, i] = tangent
    return directors


def _stable_time_step(rod: Rod, n_elements: int) -> float:
    """Upper bound on the explicit step from the fastest axial, bending, shear and twisting modes."""
    m = rod.material
    dl = rod.length / n_elements
    axial_speed = math.sqrt(m.youngs_modulus / m.density)
    shear_speed = math.sqrt(_SHEAR_COEFFICIENT * m.shear_modulus / m.density)
    twist_speed = math.sqrt(m.shear_modulus / m.density)
    return min(
        dl / axial_speed,
        4.0 * dl**2 / (math.pi**2 * rod.radius * axial_speed),
        rod.radius / shear_speed,
        dl / twist_speed,
    )


class PyElasticaSolver:
    name = "pyelastica"
    capabilities = frozenset({Capability.STRETCH, Capability.SHEAR})

    def __init__(self, time_step_safety: float = 0.5) -> None:
        self.time_step_safety = time_step_safety

    def run(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> Trajectory:
        frame_interval = scenario.duration / (n_frames - 1)
        limit = self.time_step_safety * min(_stable_time_step(r, n_elements) for r in scenario.rods)
        steps_per_frame = math.ceil(frame_interval / limit)
        dt = frame_interval / steps_per_frame

        simulator = _Simulator()
        rods = [self._add_rod(simulator, scenario, spec, n_elements, dt) for spec in scenario.rods]
        simulator.finalize()

        stepper = ea.PositionVerlet()
        frames = [[rod.position_collection.T.copy()] for rod in rods]
        time = 0.0
        for _ in range(n_frames - 1):
            for _ in range(steps_per_frame):
                time = stepper.step(simulator, time, dt)
            for rod, history in zip(rods, frames):
                history.append(rod.position_collection.T.copy())

        if not all(np.isfinite(history[-1]).all() for history in frames):
            raise FloatingPointError("PyElastica simulation diverged")
        times = np.linspace(0.0, scenario.duration, n_frames)
        return Trajectory(times, tuple(np.stack(history) for history in frames))

    @staticmethod
    def _add_rod(
        simulator: _Simulator, scenario: Scenario, spec: Rod, n_elements: int, dt: float
    ) -> ea.CosseratRod:
        nodes = spec.nodes(n_elements)
        tangents = np.diff(nodes, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        # PyElastica takes rest lengths, and from them masses and inertias, from the
        # positions it is built with. Build it unstretched, then move it to where it starts.
        rest = nodes[0] + np.concatenate(
            ([np.zeros(3)], np.cumsum(tangents * spec.length / n_elements, axis=0))
        )
        rod = ea.CosseratRod.straight_rod(
            n_elements,
            start=nodes[0],
            direction=tangents[0],
            normal=np.asarray(spec.normal, dtype=float),
            base_length=spec.length,
            base_radius=spec.radius,
            density=spec.material.density,
            youngs_modulus=spec.material.youngs_modulus,
            shear_modulus=spec.material.shear_modulus,
            position=rest.T.copy(),
            directors=_directors(nodes, np.asarray(spec.normal, dtype=float)),
        )
        rod.position_collection[:] = nodes.T
        simulator.append(rod)

        if any(scenario.gravity):
            simulator.add_forcing_to(rod).using(
                ea.GravityForces, acc_gravity=np.asarray(scenario.gravity, dtype=float)
            )
        for load in spec.loads:
            simulator.add_forcing_to(rod).using(
                _PointForce,
                force=np.asarray(load.force, dtype=float),
                node=0 if load.at is End.START else -1,
            )

        held = [(0, spec.start), (-1, spec.end)]
        positions = tuple(i for i, c in held if c is not EndCondition.FREE)
        orientations = tuple(i for i, c in held if c is EndCondition.CLAMPED)
        if positions:
            simulator.constrain(rod).using(
                ea.FixedConstraint,
                constrained_position_idx=positions,
                constrained_director_idx=orientations,
            )

        if scenario.quasi_static:
            # Critical damping of the slowest mode: fastest route to equilibrium.
            simulator.dampen(rod).using(
                ea.AnalyticalLinearDamper,
                uniform_damping_constant=2.0 * scenario.slowest_frequency(),
                time_step=dt,
            )
        return rod
