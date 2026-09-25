"""Adapter for MuJoCo's cable plugin: an inextensible chain of rigid segments
joined by ball joints, with bending and twisting stiffness at the joints."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from cosseratbench.scenario import (
    End,
    EndCondition,
    Motion,
    Obstacle,
    Plane,
    Rod,
    Scenario,
    neighbour_elements,
)
from cosseratbench.solver import Capability, Diverged
from cosseratbench.trajectory import Trajectory

# How the composite attaches the first segment to the world. A driven start is
# attached freely instead, and welded to a body the adapter moves.
_INITIAL = {EndCondition.FREE: "free", EndCondition.PINNED: "ball", EndCondition.CLAMPED: "none"}


def _numbers(values) -> str:
    return " ".join(repr(float(v)) for v in np.ravel(values))


# Contact groups. A pair collides when one's type meets the other's affinity. Obstacles
# take bit 0 and accept everything; each rod takes a bit of its own. A rod that cannot
# reach itself refuses its own bit, which switches its self-collision off; one that can
# accepts every bit. Rods beyond the 31 bits share the everything mask, which only ever
# adds collisions, never removes them.
_ALL_GROUPS = (1 << 31) - 1
_OBSTACLE_GROUP = f'contype="1" conaffinity="{_ALL_GROUPS}"'


def _rod_groups(index: int, self_contact: bool) -> str:
    bit = 1 << (index + 1)
    if self_contact or bit > _ALL_GROUPS:
        return f'contype="{_ALL_GROUPS}" conaffinity="{_ALL_GROUPS}"'
    return f'contype="{bit}" conaffinity="{_ALL_GROUPS - bit}"'


def _obstacle_xml(obstacle: Obstacle, dt: float) -> str:
    # Where two geoms have friction of their own MuJoCo takes the larger, which is the
    # rule the scenario states for a rod against an obstacle.
    # The model uses elliptic friction cones: MuJoCo's default pyramid allows as little as
    # mu / sqrt(2) of friction for sliding askew to the contact's axes, and let the capstan's
    # rope slide off at 95% of the overhang that holds.
    # Contact is as stiff as the step allows, like the welds: at MuJoCo's default a rope
    # sank 40% of its radius into the capstan's cylinder; at this, 8%.
    contact = f'friction="{obstacle.friction!r} 0 0" condim="3" solref="{2.0 * dt!r} 1"'
    if isinstance(obstacle, Plane):
        # A plane geom is unbounded (zero half-sizes) and collides on its +z side.
        return f"""
    <geom type="plane" pos="{_numbers(obstacle.point)}" zaxis="{_numbers(obstacle.unit_normal)}"
          size="0 0 1" {contact} {_OBSTACLE_GROUP}/>"""
    half = obstacle.unit_axis * obstacle.length / 2
    ends = np.concatenate([np.asarray(obstacle.center) - half, np.asarray(obstacle.center) + half])
    return f"""
    <geom type="cylinder" fromto="{_numbers(ends)}" size="{obstacle.radius!r}"
          {contact} {_OBSTACLE_GROUP}/>"""


def _defined_first_frame(nodes: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """MuJoCo takes a cable's first frame from its bend at the first vertex. A cable
    that starts straight has none, so the frame is undefined, and where the cable
    later curves its segments come out turned half a turn from each other and it
    diverges at once. Nudging the third vertex a millionth of a segment toward the
    rod's reference normal defines the frame, as the one PyElastica uses."""
    if len(nodes) < 3:
        return nodes
    first, second = np.diff(nodes[:3], axis=0)
    length = np.linalg.norm(first)
    if np.linalg.norm(np.cross(first, second)) > 1e-9 * length * np.linalg.norm(second):
        return nodes
    tangent = first / length
    across = normal - (normal @ tangent) * tangent
    if np.linalg.norm(across) < 1e-12:  # a normal along the rod: pick any perpendicular
        across = np.cross(tangent, [1.0, 0.0, 0.0] if abs(tangent[0]) < 0.9 else [0.0, 1.0, 0.0])
    nudged = nodes.copy()
    nudged[2] += 1e-6 * length * across / np.linalg.norm(across)
    return nudged


def _body_names(index: int, n_elements: int) -> list[str]:
    """What the composite calls each segment's body, in order along the rod."""
    return [
        f"r{index}_B_first",
        *(f"r{index}_B_{k}" for k in range(1, n_elements - 1)),
        f"r{index}_B_last",
    ]


def _exclude_xml(index: int, rod: Rod, n_elements: int) -> str:
    """Pairs of a rod's own segments that are too close along it to touch. MuJoCo
    already ignores segments that share a joint; this covers the rest of the span a
    rod needs to bend back on itself, which matters only for a rod discretised
    finer than its own thickness."""
    apart = neighbour_elements(rod.radius, rod.length / n_elements)
    names = _body_names(index, n_elements)
    return "".join(
        f"""
    <exclude body1="{names[k]}" body2="{names[k + offset]}"/>"""
        for offset in range(2, apart)
        for k in range(n_elements - offset)
    )


def _rod_xml(
    index: int, rod: Rod, n_elements: int, dt: float, self_contact: bool = True
) -> tuple[str, str, list[tuple[str, np.ndarray, Motion | None]]]:
    """MJCF for one rod: its worldbody elements, its equality constraints, and the
    mocap bodies holding its clamped ends as (name, starting position, motion)."""
    m = rod.material
    nodes = _defined_first_frame(rod.nodes(n_elements), np.asarray(rod.normal, dtype=float))
    segments = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    # Capsules roll smoothly over curved obstacles and over each other, with contact as
    # stiff as the step allows, as the obstacles have: at MuJoCo's default a rope sank
    # most of a radius into the rope it was resting on. A capsule's volume is not its
    # segment's, so each carries the mass of the segment it stands for, the unstretched
    # rod's mass shared out: the chain cannot stretch, so it is built in its initial
    # shape, stretched or not.
    geom = (
        f'type="capsule" size="{rod.radius!r}" '
        f'mass="{m.density * rod.area * rod.length / n_elements!r}" '
        f'friction="{rod.friction!r} 0 0" solref="{2.0 * dt!r} 1" '
        f"{_rod_groups(index, self_contact)}"
    )

    driven_start = rod.start_motion is not None
    initial = "free" if driven_start else _INITIAL[rod.start]
    body = f"""
    <composite type="cable" prefix="r{index}_" initial="{initial}"
               vertex="{_numbers(nodes)}">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="{m.shear_modulus!r}"/>
        <config key="bend" value="{m.youngs_modulus!r}"/>
        <config key="flat" value="true"/>
      </plugin>
      <joint kind="main" damping="0"/>
      <geom {geom}/>
    </composite>"""
    # MuJoCo's equality constraints are soft, and at their default stiffness a held end
    # drifts by millimetres under the cable's weight. Make them as stiff as the step allows.
    # A known limit of welds, found on the twist experiment: a cable clamped at both ends
    # diverges once it holds about 5 rad of twist, and softer welds survive only by
    # letting the clamped end turn with the twist.
    stiff = f'solref="{2.0 * dt!r} 1" solimp="0.99 0.999 0.0001"'
    equality = ""
    mocaps = []
    if rod.end is EndCondition.PINNED:
        # Segment frames have their origin at the segment's start and x along it.
        anchor = _numbers([segments[-1], 0.0, 0.0])
        equality += f"""
    <connect body1="r{index}_B_last" anchor="{anchor}" {stiff}/>"""
    # A clamped end is welded, at its end point, to a mocap body there: held still, or
    # moved as its motion says. The weld keeps the two bodies' starting relative pose.
    for end, segment, point in ((End.START, "B_first", nodes[0]), (End.END, "B_last", nodes[-1])):
        clamped = rod.condition(end) is EndCondition.CLAMPED
        if clamped and (end is End.END or driven_start):
            name = f"r{index}_{end.value}_clamp"
            motion = rod.motion(end)
            if motion is not None and motion.slides_along is not None:
                # The end is welded to a carriage that slides along the clamp. It turns only
                # about the slide direction, so the carriage's axis stays that direction.
                # The carriage's mass, one element's, rides with the end.
                mass = m.density * rod.area * rod.length / n_elements
                held = f"{name}_carriage"
                body += f"""
    <body name="{name}" mocap="true" pos="{_numbers(point)}">
      <body name="{held}">
        <joint type="slide" axis="{_numbers(motion.axis)}"/>
        <inertial pos="0 0 0" mass="{mass!r}" diaginertia="{mass * rod.radius**2!r} {mass * rod.radius**2!r} {mass * rod.radius**2!r}"/>
      </body>
    </body>"""
            else:
                held = name
                body += f"""
    <body name="{name}" mocap="true" pos="{_numbers(point)}"/>"""
            equality += f"""
    <weld body1="r{index}_{segment}" body2="{held}" {stiff}/>"""
            mocaps.append((name, point, motion))
    return body, equality, mocaps


def _stable_time_step(rod: Rod, n_elements: int) -> float:
    """Upper bound on the step from the fastest bending and twisting modes; the
    plugin's joint stiffness is integrated explicitly."""
    m = rod.material
    dl = rod.length / n_elements
    wave_speed = math.sqrt(m.youngs_modulus / m.density)
    twist_speed = math.sqrt(m.shear_modulus / m.density)
    return min(4.0 * dl**2 / (math.pi**2 * rod.radius * wave_speed), dl / twist_speed)


class MuJoCoSolver:
    name = "mujoco"
    backend_version = mujoco.__version__
    # Segments neither stretch nor shear; they do collide, with friction, with each
    # other and with fixed geoms of any shape.
    capabilities = frozenset(
        {
            Capability.ROD_CONTACT,
            Capability.ROD_FRICTION,
            Capability.CYLINDER_CONTACT,
            Capability.PLANE_CONTACT,
        }
    )

    def __init__(self, time_step_scale: float = 1.0) -> None:
        # Half the estimated stability limit, times any scale asked for.
        self.time_step_safety = 0.5 * time_step_scale

    def run(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> Trajectory:
        frame_interval = scenario.duration / (n_frames - 1)
        limit = self.time_step_safety * min(_stable_time_step(r, n_elements) for r in scenario.rods)
        steps_per_frame = math.ceil(frame_interval / limit)
        dt = float(frame_interval / steps_per_frame)  # a NumPy float's repr is not MJCF

        parts = [
            _rod_xml(i, rod, n_elements, dt, scenario.self_contact)
            for i, rod in enumerate(scenario.rods)
        ]
        obstacles = "".join(_obstacle_xml(o, dt) for o in scenario.obstacles)
        excludes = (
            "".join(_exclude_xml(i, rod, n_elements) for i, rod in enumerate(scenario.rods))
            if scenario.self_contact
            else ""
        )
        model = mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="{dt!r}" gravity="{_numbers(scenario.gravity)}" integrator="implicitfast"
          cone="elliptic"/>
  <worldbody>{obstacles}{"".join(body for body, _, _ in parts)}
  </worldbody>
  <equality>{"".join(equality for _, equality, _ in parts)}
  </equality>
  <contact>{excludes}
  </contact>
</mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        segments = [
            [model.body(name).id for name in _body_names(i, n_elements)]
            for i in range(len(scenario.rods))
        ]
        tips = [model.site(f"r{i}_S_last").id for i in range(len(scenario.rods))]
        # Each segment's body carries its cross-section, so the scenario's director,
        # written in the body's own frame at the start, is read back off the body's
        # rotation at every frame.
        rotations = [data.xmat[bodies].reshape(-1, 3, 3) for bodies in segments]
        local = [
            np.einsum("nji,nj->ni", rotation, rod.directors(n_elements))
            for rotation, rod in zip(rotations, scenario.rods)
        ]
        # (force, body it acts on, site it acts at)
        loads = [
            (
                np.asarray(load.force, dtype=float),
                segments[i][0 if load.at is End.START else -1],
                model.site(f"r{i}_S_first" if load.at is End.START else f"r{i}_S_last").id,
            )
            for i, rod in enumerate(scenario.rods)
            for load in rod.loads
        ]
        driven = [
            (model.body(name).mocapid[0], point, motion)
            for _, _, mocaps in parts
            for name, point, motion in mocaps
            if motion is not None
        ]
        quaternion = np.zeros(4)
        # Mass-proportional damping, critical for the slowest mode: fastest route to equilibrium.
        damping = 2.0 * scenario.slowest_frequency() if scenario.quasi_static else 0.0
        momentum = np.zeros(model.nv)
        no_torque = np.zeros(3)

        def state() -> tuple[list[np.ndarray], list[np.ndarray]]:
            """Node positions and element directors of every rod, as [n, 3] each."""
            mujoco.mj_kinematics(model, data)
            positions = [
                np.vstack([data.xpos[bodies], data.site_xpos[tip]])
                for bodies, tip in zip(segments, tips)
            ]
            directors = [
                np.einsum("nij,nj->ni", data.xmat[bodies].reshape(-1, 3, 3), d)
                for bodies, d in zip(segments, local)
            ]
            return positions, directors

        positions, directors = state()
        frames = [[p] for p in positions]
        turned = [[d] for d in directors]
        for frame in range(n_frames - 1):
            for _ in range(steps_per_frame):
                # Aim each driven clamp where the motion has it at the end of this step.
                for mocap, point, motion in driven:
                    displacement, rotation = motion.pose(data.time + dt)
                    data.mocap_pos[mocap] = point + displacement
                    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
                    data.mocap_quat[mocap] = quaternion
                # Forces are computed from this step's state, between MuJoCo's two half
                # steps. The mass matrix in particular must be current: a thin cable's is
                # badly conditioned, and damping computed with last step's makes a free
                # cable spin up and diverge.
                mujoco.mj_step1(model, data)
                data.qfrc_applied[:] = 0.0
                for force, body, site in loads:
                    mujoco.mj_applyFT(
                        model, data, force, no_torque, data.site_xpos[site], body, data.qfrc_applied
                    )
                if damping:
                    mujoco.mj_mulM(model, data, momentum, data.qvel)
                    data.qfrc_applied -= damping * momentum
                mujoco.mj_step2(model, data)
            # MuJoCo resets a diverged simulation and carries on, so stop at the first sign.
            if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                # MuJoCo has already reset its clock, so report the frame's time.
                raise Diverged("MuJoCo simulation diverged", time=(frame + 1) * frame_interval)
            positions, directors = state()
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
