"""Adapter for dismech-python: discrete elastic rods (Bergou et al.), a chain of
nodes with a twist angle per edge, integrated implicitly by Newton's method.

The rod stretches, bends and twists but does not shear. Rods touch each other and
themselves through the implicit contact model (IMC), with friction; a floor is a
level plane. Contact stiffnesses are chosen so that the rods' own weight (or their
loads) sinks them a twentieth of a radius, and the implicit step is a fixed
fraction of a second scaled by ``time_step_scale``: with an implicit integrator
the step is a matter of accuracy and cost, not stability.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import math
import os
from importlib.metadata import distribution

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")  # dismech imports matplotlib at import time
import dismech
from dismech.contact import ContactPair
from dismech.contact import imc_energy as _imc
from dismech.contact import imc_friction_energy as _imc_friction
from dismech.external_forces import compute_gravity_forces
from dismech.external_forces import ground_contact as _ground

from cosseratbench.scenario import End, EndCondition, Motion, Plane, Scenario, neighbour_elements
from cosseratbench.solver import Capability, Diverged
from cosseratbench.trajectory import Trajectory


def _floor_contact_at_every_node(original):
    """dismech's floor contact force, whose raw form lists every node.

    At the pinned commit, the raw form (per node, used for floor friction) lists only
    the nodes near the floor, while the friction built on it pairs those rows with
    every node's velocity; a rod partly on the floor raises an IndexError. The
    touching nodes are the ones the original selects, so its rows are put back in
    place and the rest left at zero. Frictionless floors do not use the raw form.
    """

    def contact(robot, q, as_raw=False):
        if not as_raw:
            return original(robot, q)
        heights = q[2 : robot.end_node_dof_index : 3]
        near = heights - robot.env.ground_h - robot.env.ground_z <= robot.env.ground_delta
        if not near.any():
            return original(robot, q, as_raw=True)
        force, stiffness = original(robot, q, as_raw=True)
        every_force = np.zeros((len(heights), 3))
        every_stiffness = np.zeros((len(heights), 3, 3))
        every_force[near], every_stiffness[near] = force, stiffness
        return every_force, every_stiffness

    contact.fixed_by_cosseratbench = True
    return contact


if not getattr(_ground.compute_ground_contact, "fixed_by_cosseratbench", False):
    _ground.compute_ground_contact = _floor_contact_at_every_node(_ground.compute_ground_contact)

# dismech derives its contact functions symbolically and compiles them every time a
# stepper is built: 3 s for two ropes, 12 s with friction, paid again inside every
# timed run although it is a one-off cost like JIT compilation. The compiled
# functions depend on nothing but the formulas, so each process keeps the first.
if not getattr(_imc.get_lambda_fns, "cache_info", None):
    _imc.get_lambda_fns = functools.cache(_imc.get_lambda_fns)
if not getattr(_imc_friction.generate_velocity_jacobian_funcs, "cache_info", None):
    _imc_friction.generate_velocity_jacobian_funcs = functools.cache(
        _imc_friction.generate_velocity_jacobian_funcs
    )

STEP = 2e-3  # s, the implicit step at time_step_scale 1
SINK = 0.05  # radii a rod sinks into what it touches under the characteristic force
NEWTON_ITERATIONS = 50


class _Clamp:
    """A clamped end, held or driven: the end node, its neighbour, and the edge between."""

    def __init__(self, robot, nodes: tuple[int, int], edge: int, motion: Motion | None) -> None:
        positions = robot.state.q[: 3 * robot.n_nodes].reshape(-1, 3)
        self.nodes, self.edge, self.motion = nodes, edge, motion
        self.start = positions[list(nodes)].copy()
        self.m1 = robot.state.m1[edge].copy()  # the material direction as it starts

    def hold(self, robot, time: float):
        """The robot with this clamp where its motion has it at ``time``: the two nodes
        moved and the edge's material direction turned, both rigidly, with the
        reference frames carried to the new tangent as the stepper carries them."""
        displacement, rotation = self.motion.pose(time)
        q = robot.state.q.copy()
        target = (
            self.start[0]
            + displacement
            + np.array([np.zeros(3), self.start[1] - self.start[0]]) @ rotation.T
        )
        held = np.arange(3)
        if self.motion.slides_along is not None:
            held = held[np.abs(self.motion.axis) < 0.5]  # the slide axis stays the rod's own
        for node, point in zip(self.nodes, target):
            q[3 * node + held] = point[held]
        a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
        turned = rotation @ self.m1
        q[robot.map_edge_to_dof(self.edge)] = math.atan2(
            turned @ a2[self.edge], turned @ a1[self.edge]
        )
        m1, m2 = robot.compute_material_directors(q, a1, a2)
        ref_twist = robot.compute_reference_twist(robot.twist_springs, q, a1, robot.state.ref_twist)
        return robot.update(q=q, a1=a1, a2=a2, m1=m1, m2=m2, ref_twist=ref_twist)


def _damped(base, velocity_gradient: float):
    """A stepper that adds damping proportional to mass, at a rate set on construction,
    for quasi-static scenarios: the same rule as the other adapters."""

    class Damped(base):
        def __init__(self, robot, rate: float) -> None:
            # Below min_force the stepper stops iterating, so it must sit under the tolerance.
            super().__init__(robot, min_force=0.1 * robot.sim_params.tol)
            self.rate = rate

        def _compute_forces_and_jacobian(self, forces, jacobian, robot, q, u, first_iter=False):
            forces, jacobian = super()._compute_forces_and_jacobian(
                forces, jacobian, robot, q, u, first_iter
            )
            # dismech at the pinned commit subtracts gravity from the residual twice, once
            # before the contact forces and once after, so its rods fall at 2 g. One of
            # them is put back here; see the findings.
            if "gravity" in robot.env.ext_force_list:
                forces += compute_gravity_forces(robot)
            if self.rate:
                mass = robot.mass_matrix
                forces += self.rate * mass * u
                diagonal = np.arange(len(q))
                jacobian[diagonal, diagonal] += (
                    self.rate * mass * velocity_gradient / robot.sim_params.dt
                )
            return forces, jacobian

    return Damped


def _installed_version() -> str:
    """dismech-python's version and, installed from git, the commit: its version number
    alone does not change between commits."""
    installed = distribution("dismech-python")
    source = json.loads(installed.read_text("direct_url.json") or "{}")
    commit = source.get("vcs_info", {}).get("commit_id")
    return f"{installed.version}+{commit[:7]}" if commit else installed.version


class DismechSolver:
    name = "dismech"
    backend_version = _installed_version()
    # Discrete elastic rods stretch, bend and twist, and IMC gives them frictional
    # contact with each other; a floor is a level plane. No shear, no cylinders.
    capabilities = frozenset(
        {
            Capability.STRETCH,
            Capability.ROD_CONTACT,
            Capability.ROD_FRICTION,
            Capability.PLANE_CONTACT,
        }
    )

    def __init__(self, time_step_scale: float = 1.0) -> None:
        self.time_step = STEP * time_step_scale

    def _numerics(self, scenario: Scenario, n_frames: int) -> dict:
        """Every numerical choice the adapter makes for a scenario, in one place."""
        frame_interval = scenario.duration / (n_frames - 1)
        steps_per_frame = math.ceil(frame_interval / self.time_step)
        first = scenario.rods[0]
        weight = sum(
            r.material.density * r.area * r.length * float(np.linalg.norm(scenario.gravity))
            for r in scenario.rods
        )
        loads = sum(float(np.linalg.norm(load.force)) for r in scenario.rods for load in r.loads)
        bending = first.material.youngs_modulus * first.second_moment_of_area / first.length**2
        force = max(weight, loads, bending)  # the characteristic force of the scenario
        return {
            "dt": frame_interval / steps_per_frame,
            "steps_per_frame": steps_per_frame,
            "force": force,
            "integrator": (
                "implicit Euler (Newton)" if scenario.quasi_static else "Newmark-beta (Newton)"
            ),
            "newton_tolerance": 1e-8 * force,
            "damping_rate": 2.0 * scenario.slowest_frequency() if scenario.quasi_static else 0.0,
            # IMC's energy is kc ((2h - d) / h)^2 for centrelines d apart, so the force on
            # a rod sunk SINK radii into another is 2 kc SINK / h.
            "contact_stiffness": force * first.radius / (2 * SINK),
            # The floor's force on a node sunk a distance s is about 2 stiffness s.
            "floor_stiffness": force / (2 * SINK * first.radius),
            "friction_velocity_tolerance": 1e-4 * first.length,
        }

    def settings(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> dict:
        """The numerical choices this adapter makes for a scenario, as it makes them."""
        numerics = self._numerics(scenario, n_frames)
        touching = scenario.self_contact or len(scenario.rods) > 1
        return {
            "integrator": numerics["integrator"],
            "time_step": numerics["dt"],
            "damping_rate": numerics["damping_rate"],
            "newton_tolerance": numerics["newton_tolerance"],
            **({"contact_stiffness": numerics["contact_stiffness"]} if touching else {}),
            **({"floor_stiffness": numerics["floor_stiffness"]} if scenario.obstacles else {}),
        }

    def run(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> Trajectory:
        rods = scenario.rods
        first = rods[0]
        if any(r.radius != first.radius or r.material != first.material for r in rods):
            raise NotImplementedError("rods of more than one radius or material")
        for obstacle in scenario.obstacles:
            if not isinstance(obstacle, Plane) or not np.allclose(obstacle.unit_normal, [0, 0, 1]):
                raise NotImplementedError("an obstacle other than a level floor")
        for rod in rods:
            for end in End:
                motion = rod.motion(end)
                sliding = motion is not None and motion.slides_along is not None
                if sliding and np.count_nonzero(np.abs(motion.axis) > 1e-12) != 1:
                    raise NotImplementedError("an end sliding along a direction off the axes")

        numerics = self._numerics(scenario, n_frames)
        dt, steps_per_frame = numerics["dt"], numerics["steps_per_frame"]

        # One chain of nodes and edges per rod, all in one system so that rods can
        # touch each other. dismech takes the shape it is built in as stress-free, so it
        # is built straight and unstretched, the scenario's rest shape, then moved to
        # where the rod starts. (Unstretched but bent along the start, as PyElastica is
        # built, would make the starting bend the rod's natural curvature.) Each rod's
        # straight line runs along its first element, from where it starts.
        starts, rests, node_offsets, edge_offsets = [], [], [], []
        for rod in rods:
            nodes = rod.nodes(n_elements)
            along = (nodes[1] - nodes[0]) / np.linalg.norm(nodes[1] - nodes[0])
            steps = np.arange(n_elements + 1)[:, None] * rod.length / n_elements
            rest = nodes[0] + steps * along
            node_offsets.append(sum(len(s) for s in starts))
            edge_offsets.append(sum(len(s) - 1 for s in starts))
            starts.append(nodes)
            rests.append(rest)
        edges = np.concatenate(
            [
                np.stack(
                    [np.arange(o, o + n_elements), np.arange(o + 1, o + n_elements + 1)], axis=1
                )
                for o in node_offsets
            ]
        )
        geometry = dismech.Geometry(np.concatenate(rests), edges, np.empty(0), plot_from_txt=False)

        material = first.material
        params = dismech.SimParams(
            static_sim=False,
            two_d_sim=False,
            use_mid_edge=False,
            use_line_search=True,
            log_data=False,
            log_step=1,
            show_floor=False,
            dt=dt,
            max_iter=NEWTON_ITERATIONS,
            total_time=scenario.duration,
            plot_step=1,
            tol=numerics["newton_tolerance"],
            ftol=1e-12,
            dtol=0.0,
        )
        environment = dismech.Environment()
        if any(scenario.gravity):
            environment.add_force("gravity", g=np.asarray(scenario.gravity, dtype=float))
        pulls: dict[int, np.ndarray] = {}
        for offset, rod in zip(node_offsets, rods):
            for load in rod.loads:
                node = offset + (0 if load.at is End.START else n_elements)
                pulls[node] = pulls.get(node, np.zeros(3)) + np.asarray(load.force, dtype=float)
        if pulls:
            environment.add_force(
                "pointForces",
                point_force_node_indices=list(pulls),
                point_force_vectors=np.array(list(pulls.values())),
            )
        radius = first.radius
        friction = max(r.friction for r in rods)
        touching = scenario.self_contact or len(rods) > 1
        if touching:
            # IMC's energy is kc ((2h - d) / h)^2 for centrelines d apart, so the force
            # on a rod sunk SINK radii into another is 2 kc SINK / h.
            environment.add_force(
                "selfContact", delta=0.2 * radius, h=radius, kc=numerics["contact_stiffness"]
            )
            if friction > 0.0:
                environment.add_force(
                    "selfFriction", mu=friction, vel_tol=numerics["friction_velocity_tolerance"]
                )
        for obstacle in scenario.obstacles:
            # The floor's force on a node sunk a distance s is about 2 stiffness s.
            environment.add_force(
                "floorContact",
                ground_z=float(obstacle.point[2]),
                stiffness=numerics["floor_stiffness"],
                delta=0.2 * radius,
                h=radius,
            )
            mu = max(obstacle.friction, friction)
            if mu > 0.0:
                environment.add_force(
                    "floorFriction", mu=mu, vel_tol=numerics["friction_velocity_tolerance"]
                )

        shear = material.shear_modulus
        robot = dismech.SoftRobot(
            dismech.GeomParams(rod_r0=radius, shell_h=0.0),
            dismech.Material(
                density=material.density,
                youngs_rod=material.youngs_modulus,
                youngs_shell=0.0,
                poisson_rod=material.youngs_modulus / (2.0 * shear) - 1.0,  # G = E / 2(1 + nu)
                poisson_shell=0.0,
            ),
            geometry,
            params,
            environment,
        )
        if touching:
            # Which elements can touch is the benchmark's rule, not dismech's (any two edges
            # three apart in the whole system): elements of different rods always, and of
            # one rod when the scenario lets it reach itself and they are further apart
            # along it than neighbours (decision 0011).
            apart = neighbour_elements(radius, first.length / n_elements)
            rod_of = np.repeat(np.arange(len(rods)), n_elements)
            i, j = np.triu_indices(len(edges), k=1)
            same = rod_of[i] == rod_of[j]
            keep = ~same | (scenario.self_contact & (j - i >= apart))
            robot._SoftRobot__contact_pairs = [
                ContactPair(np.concatenate([edges[a], edges[b]]), robot.map_node_to_dof)
                for a, b in zip(i[keep], j[keep])
            ]

        # The stepper is built now, while every rod is straight and unstretched: dismech's
        # elastic energies take their natural strains from the state it is built with, and
        # built later they would make the starting shape stress-free (see the findings).
        # Quasi-static scenarios settle under mass-proportional damping, critical for the
        # slowest mode, with implicit Euler, whose own damping helps; dynamics use
        # Newmark-beta, which conserves energy.
        if scenario.quasi_static:
            stepper_class = _damped(dismech.ImplicitEulerTimeStepper, 1.0)
        else:
            stepper_class = _damped(dismech.NewmarkBetaTimeStepper, 2.0)  # gamma / beta
        rate = numerics["damping_rate"]
        with contextlib.redirect_stdout(io.StringIO()):  # IMC prints its settings
            stepper = stepper_class(robot, rate)

        # Move the rods to where they start. dismech built its reference frames for the
        # straight rod, so they are rebuilt for the starting shape: each rod's directors
        # (its normal carried along it by parallel transport), with no twist.
        q = robot.state.q.copy()
        q[: 3 * robot.n_nodes] = np.concatenate(starts).ravel()
        a1 = np.concatenate([rod.directors(n_elements) for rod in rods])
        tangents = np.concatenate(
            [
                np.diff(nodes, axis=0)
                / np.linalg.norm(np.diff(nodes, axis=0), axis=1, keepdims=True)
                for nodes in starts
            ]
        )
        a2 = np.cross(tangents, a1)
        m1, m2 = robot.compute_material_directors(q, a1, a2)
        ref_twist = robot.compute_reference_twist(
            robot.twist_springs, q, a1, np.zeros_like(robot.state.ref_twist)
        )
        robot = robot.update(q=q, a1=a1, a2=a2, m1=m1, m2=m2, ref_twist=ref_twist)

        # Ends. A pinned end is one node held; a clamped end is two, and the edge between.
        clamps: list[_Clamp] = []
        for offset, edge_offset, rod in zip(node_offsets, edge_offsets, rods):
            for end in End:
                condition = rod.condition(end)
                if condition is EndCondition.FREE:
                    continue
                if end is End.START:
                    nodes, edge = (offset, offset + 1), edge_offset
                else:
                    nodes, edge = (
                        (offset + n_elements, offset + n_elements - 1),
                        edge_offset + n_elements - 1,
                    )
                if condition is EndCondition.PINNED:
                    robot = robot.fix_nodes([nodes[0]], fix_edges=False)
                    continue
                motion = rod.motion(end)
                if motion is not None and motion.slides_along is not None:
                    for axis in np.flatnonzero(np.abs(motion.axis) < 0.5):
                        robot = robot.fix_nodes(list(nodes), axis=int(axis))
                    robot = robot.fix_edges(np.array([edge]))
                else:
                    robot = robot.fix_nodes(list(nodes))
                if motion is not None:
                    clamps.append(_Clamp(robot, nodes, edge, motion))

        # Each rod's scenario director, in its own material frame; read back off it later.
        local = []
        for offset, edge_offset, rod in zip(node_offsets, edge_offsets, rods):
            m1 = robot.state.m1[edge_offset : edge_offset + n_elements]
            m2 = robot.state.m2[edge_offset : edge_offset + n_elements]
            d1 = rod.directors(n_elements)
            local.append(np.stack([(d1 * m1).sum(axis=1), (d1 * m2).sum(axis=1)], axis=1))

        def state(robot) -> tuple[list[np.ndarray], list[np.ndarray]]:
            positions = robot.state.q[: 3 * robot.n_nodes].reshape(-1, 3)
            rods_positions = [positions[o : o + n_elements + 1].copy() for o in node_offsets]
            directors = []
            for edge_offset, (c, s) in zip(edge_offsets, (l.T for l in local)):
                m1 = robot.state.m1[edge_offset : edge_offset + n_elements]
                m2 = robot.state.m2[edge_offset : edge_offset + n_elements]
                directors.append(c[:, None] * m1 + s[:, None] * m2)
            return rods_positions, directors

        positions, directors = state(robot)
        frames = [[p] for p in positions]
        turned = [[d] for d in directors]
        time = 0.0
        for _ in range(n_frames - 1):
            for _ in range(steps_per_frame):
                for clamp in clamps:
                    robot = clamp.hold(robot, time + dt)
                try:
                    robot, _, _ = stepper.step(robot, debug=False)
                except (ValueError, np.linalg.LinAlgError) as error:
                    raise Diverged(f"dismech's Newton solve failed: {error}", time=time) from error
                time += dt
            positions, directors = state(robot)
            if not all(np.isfinite(p).all() for p in positions):
                raise Diverged("dismech simulation diverged", time=time)
            for history, p in zip(frames, positions):
                history.append(p)
            for history, d in zip(turned, directors):
                history.append(d)
        times = np.linspace(0.0, scenario.duration, n_frames)
        return Trajectory(
            times,
            tuple(np.stack(history) for history in frames),
            tuple(np.stack(history) for history in turned),
        )
