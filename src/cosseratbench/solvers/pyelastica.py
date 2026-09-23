"""Adapter for PyElastica: Cosserat rods with an explicit symplectic integrator."""

from __future__ import annotations

import math

import elastica as ea
import numpy as np

from cosseratbench.scenario import Cylinder, End, EndCondition, Motion, Plane, Rod, Scenario
from cosseratbench.solver import Capability, Diverged
from cosseratbench.trajectory import Trajectory

# Timoshenko shear coefficient PyElastica uses for circular cross-sections.
_SHEAR_COEFFICIENT = 27.0 / 28.0


class _Simulator(ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.Contact):
    pass


class _PointForce(ea.NoForces):
    def __init__(self, force: np.ndarray, node: int) -> None:
        super().__init__()
        self.force = force
        self.node = node

    def apply_forces(self, system, time=0.0) -> None:
        system.external_forces[:, self.node] += self.force


class _DrivenClamp(ea.ConstraintBase):
    """Moves and turns one end of a rod as a Motion prescribes: its node, and the
    frame of the element it ends."""

    def __init__(self, *args, node: int, motion: Motion, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node, self.element, self.motion = node, node if node == 0 else -1, motion
        self.start_position = self.system.position_collection[:, node].copy()
        self.start_directors = self.system.director_collection[:, :, self.element].copy()

    def _held(self, prescribed: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """The prescribed vector, except along a slide direction, where the rod's own stands."""
        if self.motion.slides_along is None:
            return prescribed
        axis = self.motion.axis
        return prescribed + axis * (axis @ (actual - prescribed))

    def constrain_values(self, system, time) -> None:
        displacement, rotation = self.motion.pose(float(time))
        target = self.start_position + displacement
        system.position_collection[:, self.node] = self._held(
            target, system.position_collection[:, self.node]
        )
        # Directors are the rows of the frame; turning them by R turns the frame by R^T.
        system.director_collection[:, :, self.element] = self.start_directors @ rotation.T

    def constrain_rates(self, system, time) -> None:
        velocity, angular_velocity = self.motion.rates(float(time))
        system.velocity_collection[:, self.node] = self._held(
            velocity, system.velocity_collection[:, self.node]
        )
        # PyElastica keeps angular velocity in the element's own frame.
        frame = system.director_collection[:, :, self.element]
        system.omega_collection[:, self.element] = frame @ angular_velocity


def _frames(spec: Rod, n_elements: int) -> np.ndarray:
    """Material frames [3, 3, n_elements] of the rod as it starts: the scenario's
    directors, the tangents, and the direction that completes each right-handed frame.
    Directors are the rows of the frame, as PyElastica keeps them."""
    nodes = spec.nodes(n_elements)
    tangents = np.diff(nodes, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    d1 = spec.directors(n_elements)
    return np.stack([d1, np.cross(tangents, d1), tangents], axis=0).transpose(0, 2, 1)


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
    capabilities = frozenset(
        {Capability.STRETCH, Capability.SHEAR, Capability.ROD_CONTACT}
    )  # PyElastica's rod-rod and self-contact are frictionless: no ROD_FRICTION

    def __init__(self, time_step_scale: float = 1.0) -> None:
        # Half the estimated stability limit, times any scale asked for.
        self.time_step_safety = 0.5 * time_step_scale

    def run(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> Trajectory:
        frame_interval = scenario.duration / (n_frames - 1)
        limit = self.time_step_safety * min(_stable_time_step(r, n_elements) for r in scenario.rods)
        steps_per_frame = math.ceil(frame_interval / limit)
        dt = frame_interval / steps_per_frame

        simulator = _Simulator()
        rods = [self._add_rod(simulator, scenario, spec, n_elements, dt) for spec in scenario.rods]
        for obstacle in scenario.obstacles:
            if isinstance(obstacle, Plane):
                plane = ea.Plane(
                    plane_origin=np.asarray(obstacle.point, dtype=float),
                    plane_normal=obstacle.unit_normal,
                )
                simulator.append(plane)
                for spec, rod in zip(scenario.rods, rods):
                    self._add_plane_contact(simulator, rod, plane, spec, obstacle, n_elements, dt)
            else:
                cylinder = self._add_cylinder(simulator, obstacle)
                for spec, rod in zip(scenario.rods, rods):
                    self._add_contact(simulator, rod, cylinder, spec, obstacle, n_elements, dt)
        self._add_rod_contact(simulator, scenario, rods, n_elements, dt)
        simulator.finalize()

        stepper = ea.PositionVerlet()
        frames = [[rod.position_collection.T.copy()] for rod in rods]
        # The first row of each element's frame is its d1, the scenario's director.
        turned = [[rod.director_collection[0].T.copy()] for rod in rods]
        time = 0.0
        for _ in range(n_frames - 1):
            for _ in range(steps_per_frame):
                time = stepper.step(simulator, time, dt)
            for rod, history, directors in zip(rods, frames, turned):
                history.append(rod.position_collection.T.copy())
                directors.append(rod.director_collection[0].T.copy())
            if not all(np.isfinite(history[-1]).all() for history in frames):
                raise Diverged("PyElastica simulation diverged", time=float(time))
        times = np.linspace(0.0, scenario.duration, n_frames)
        return Trajectory(
            times,
            tuple(np.stack(history) for history in frames),
            tuple(np.stack(directors) for directors in turned),
        )

    @staticmethod
    def _add_cylinder(simulator: _Simulator, obstacle: Cylinder) -> ea.Cylinder:
        axis = obstacle.unit_axis
        normal = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
        cylinder = ea.Cylinder(
            start=np.asarray(obstacle.center) - axis * obstacle.length / 2,
            direction=axis,
            normal=normal / np.linalg.norm(normal),
            base_length=obstacle.length,
            base_radius=obstacle.radius,
            density=1000.0,  # held still, so its mass does not matter
        )
        simulator.append(cylinder)
        simulator.constrain(cylinder).using(
            ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )
        return cylinder

    @staticmethod
    def _node_mass(spec: Rod, n_elements: int) -> float:
        return spec.material.density * spec.area * spec.length / n_elements

    @staticmethod
    def _spring(mass: float, dt: float) -> tuple[float, float]:
        """Penalty stiffness and damping for a contact: as stiff as the explicit step
        allows, a contact vibration of a node of ``mass`` no faster than 0.5 / dt,
        damped near critically."""
        stiffness = mass * (0.5 / dt) ** 2
        return stiffness, float(np.sqrt(stiffness * mass))

    @classmethod
    def _add_rod_contact(
        cls,
        simulator: _Simulator,
        scenario: Scenario,
        rods: list[ea.CosseratRod],
        n_elements: int,
        dt: float,
    ) -> None:
        """Rods against each other and against themselves. PyElastica's contact here is
        a normal penalty spring only: it has no friction between rods, so rods that
        touch slide freely over each other whatever their friction says. Experiments
        that need it require Capability.ROD_FRICTION, which this solver does not claim.

        A pair's spring is sized by the lighter of the two rods' nodes, so the contact
        vibration stays inside the step for both.
        """
        masses = [cls._node_mass(spec, n_elements) for spec in scenario.rods]
        for i, rod in enumerate(rods):
            if scenario.self_contact:
                # PyElastica checks every pair of a rod's elements every step, with no
                # broadphase, which costs several times the rest of the step; a scenario
                # whose rods cannot reach themselves says so and skips it.
                k, nu = cls._spring(masses[i], dt)
                simulator.detect_contact_between(rod, rod).using(ea.RodSelfContact, k=k, nu=nu)
            for j in range(i + 1, len(rods)):
                k, nu = cls._spring(min(masses[i], masses[j]), dt)
                simulator.detect_contact_between(rod, rods[j]).using(ea.RodRodContact, k=k, nu=nu)

    @staticmethod
    def _add_contact(
        simulator: _Simulator,
        rod: ea.CosseratRod,
        cylinder: ea.Cylinder,
        spec: Rod,
        obstacle: Cylinder,
        n_elements: int,
        dt: float,
    ) -> None:
        # Contact is a penalty spring on each element. Make it as stiff as the explicit
        # step allows, a contact vibration of a node no faster than 0.5 / dt, and damp
        # it near critically. Static friction is imitated by viscous friction, the lesser
        # of it and Coulomb's, so a strong viscous term only stops rods creeping: Coulomb's
        # cap keeps the force bounded. The benchmark reports how deep rods sink and
        # whether they creep, so these choices are visible rather than trusted.
        node_mass = PyElasticaSolver._node_mass(spec, n_elements)
        stiffness, damping = PyElasticaSolver._spring(node_mass, dt)
        simulator.detect_contact_between(rod, cylinder).using(
            ea.RodCylinderContact,
            k=stiffness,
            nu=damping,
            velocity_damping_coefficient=1e3 * node_mass / dt,
            friction_coefficient=max(obstacle.friction, spec.friction),
        )

    @staticmethod
    def _add_plane_contact(
        simulator: _Simulator,
        rod: ea.CosseratRod,
        plane: ea.Plane,
        spec: Rod,
        obstacle: Plane,
        n_elements: int,
        dt: float,
    ) -> None:
        # The same penalty spring as against a cylinder. Against a plane PyElastica has
        # real static friction: an element moving slower than a threshold is held by up
        # to mu times the normal force, and one moving faster gets kinetic friction. The
        # threshold is a small fraction of a rod length per second. The same coefficient
        # applies along the rod, against it, and sideways, as the scenario states.
        node_mass = PyElasticaSolver._node_mass(spec, n_elements)
        stiffness, damping = PyElasticaSolver._spring(node_mass, dt)
        friction = max(obstacle.friction, spec.friction)
        if friction == 0.0:
            simulator.detect_contact_between(rod, plane).using(
                ea.RodPlaneContact, k=stiffness, nu=damping
            )
            return
        simulator.detect_contact_between(rod, plane).using(
            ea.RodPlaneContactWithAnisotropicFriction,
            k=stiffness,
            nu=damping,
            slip_velocity_tol=1e-4 * spec.length,
            static_mu_array=np.full(3, friction),
            kinetic_mu_array=np.full(3, friction),
        )

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
            directors=_frames(spec, n_elements),
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

        held = [(0, End.START), (-1, End.END)]
        for node, end in held:
            if (motion := spec.motion(end)) is not None:
                simulator.constrain(rod).using(_DrivenClamp, node=node, motion=motion)
        still = [(i, spec.condition(end)) for i, end in held if spec.motion(end) is None]
        positions = tuple(i for i, c in still if c is not EndCondition.FREE)
        orientations = tuple(i for i, c in still if c is EndCondition.CLAMPED)
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
