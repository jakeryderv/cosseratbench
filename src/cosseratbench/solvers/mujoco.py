"""Adapter for MuJoCo's cable plugin: an inextensible chain of rigid segments
joined by ball joints, with bending and twisting stiffness at the joints."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from cosseratbench.scenario import End, EndCondition, Motion, Rod, Scenario
from cosseratbench.solver import Capability
from cosseratbench.trajectory import Trajectory

# How the composite attaches the first segment to the world. A driven start is
# attached freely instead, and welded to a body the adapter moves.
_INITIAL = {EndCondition.FREE: "free", EndCondition.PINNED: "ball", EndCondition.CLAMPED: "none"}


def _numbers(values) -> str:
    return " ".join(repr(float(v)) for v in np.ravel(values))


def _rod_xml(
    index: int, rod: Rod, n_elements: int, dt: float
) -> tuple[str, str, list[tuple[str, np.ndarray, Motion | None]]]:
    """MJCF for one rod: its worldbody elements, its equality constraints, and the
    mocap bodies holding its clamped ends as (name, starting position, motion)."""
    m = rod.material
    nodes = rod.nodes(n_elements)
    segments = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    # The chain cannot stretch, so it is built in its initial shape, stretched or not.
    # Scaling the density keeps the mass that of the unstretched rod.
    density = float(m.density * rod.length / segments.sum())
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
      <geom type="cylinder" size="{rod.radius!r}" density="{density!r}" contype="0" conaffinity="0"/>
    </composite>"""
    # MuJoCo's equality constraints are soft, and at their default stiffness a held end
    # drifts by millimetres under the cable's weight. Make them as stiff as the step allows.
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
            body += f"""
    <body name="{name}" mocap="true" pos="{_numbers(point)}"/>"""
            equality += f"""
    <weld body1="r{index}_{segment}" body2="{name}" {stiff}/>"""
            mocaps.append((name, point, rod.motion(end)))
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
    capabilities: frozenset[Capability] = frozenset()  # segments neither stretch nor shear

    def __init__(self, time_step_safety: float = 0.5) -> None:
        self.time_step_safety = time_step_safety

    def run(self, scenario: Scenario, *, n_elements: int, n_frames: int) -> Trajectory:
        frame_interval = scenario.duration / (n_frames - 1)
        limit = self.time_step_safety * min(_stable_time_step(r, n_elements) for r in scenario.rods)
        steps_per_frame = math.ceil(frame_interval / limit)
        dt = frame_interval / steps_per_frame

        parts = [_rod_xml(i, rod, n_elements, dt) for i, rod in enumerate(scenario.rods)]
        model = mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="{dt!r}" gravity="{_numbers(scenario.gravity)}" integrator="implicitfast"/>
  <worldbody>{"".join(body for body, _, _ in parts)}
  </worldbody>
  <equality>{"".join(equality for _, equality, _ in parts)}
  </equality>
</mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        segments = [
            [model.body(f"r{i}_B_first").id]
            + [model.body(f"r{i}_B_{k}").id for k in range(1, n_elements - 1)]
            + [model.body(f"r{i}_B_last").id]
            for i in range(len(scenario.rods))
        ]
        tips = [model.site(f"r{i}_S_last").id for i in range(len(scenario.rods))]
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

        def nodes() -> list[np.ndarray]:
            mujoco.mj_kinematics(model, data)
            return [
                np.vstack([data.xpos[bodies], data.site_xpos[tip]])
                for bodies, tip in zip(segments, tips)
            ]

        frames = [[positions] for positions in nodes()]
        for _ in range(n_frames - 1):
            for _ in range(steps_per_frame):
                data.qfrc_applied[:] = 0.0
                for force, body, site in loads:
                    mujoco.mj_applyFT(
                        model, data, force, no_torque, data.site_xpos[site], body, data.qfrc_applied
                    )
                if damping:
                    mujoco.mj_mulM(model, data, momentum, data.qvel)
                    data.qfrc_applied -= damping * momentum
                # Aim each driven clamp where the motion has it at the end of this step.
                for mocap, point, motion in driven:
                    displacement, rotation = motion.pose(data.time + dt)
                    data.mocap_pos[mocap] = point + displacement
                    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
                    data.mocap_quat[mocap] = quaternion
                mujoco.mj_step(model, data)
            for history, positions in zip(frames, nodes()):
                history.append(positions)

        if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise FloatingPointError("MuJoCo simulation diverged")
        times = np.linspace(0.0, scenario.duration, n_frames)
        return Trajectory(times, tuple(np.stack(history) for history in frames))
