"""Each built-in solver must reproduce the analytical references to within the
error its physical model and default resolution explain."""

import dataclasses
import itertools

import numpy as np
import pytest

from cosseratbench import (
    End,
    EndCondition,
    Material,
    Motion,
    PointLoad,
    Rod,
    Scenario,
    registry,
    run,
)
from cosseratbench.experiment import max_rod_overlap
from cosseratbench.experiments.pendulum import static_strain
from cosseratbench.metrics import twist_angles

pytestmark = pytest.mark.slow

# MuJoCo's cable cannot stretch, and stretch deepens this catenary's sag by ~1.2%.
CATENARY_SAG_TOLERANCE = {"pyelastica": 2e-3, "mujoco": 2e-2, "dismech": 2e-3}


@pytest.fixture(params=["pyelastica", "mujoco", "dismech"])
def solver(request):
    pytest.importorskip({"pyelastica": "elastica"}.get(request.param, request.param))
    return registry.load_solver(request.param)


def completed(result):
    """The result, unless the solver could not run it, which is a fact about the
    solver rather than a failure of the test."""
    if not result.supported:
        pytest.skip(f"{result.solver} cannot run this: needs {', '.join(result.missing)}")
    return result


# dismech's floor friction cannot hold anything still: Newton fails on the first step
# in which it must stick. Strict, so the test says so once it is fixed; see the findings.
FLOOR_FRICTION_STICKS = "dismech's floor friction fails to converge when it must stick"


def known_failure(request, solver, names, reason):
    if solver.name in names:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))


def test_catenary(solver):
    result = run(registry.load_experiment("catenary"), solver)
    assert result.failure is None
    assert result.metrics["settling_residual"] < 1e-5
    assert result.metrics["sag_error"] < CATENARY_SAG_TOLERANCE[solver.name]


def test_cantilever_converges_to_beam_theory(solver):
    experiment = registry.load_experiment("cantilever")
    coarse, fine = (run(experiment, solver, n_elements=n) for n in (10, 20))
    assert fine.failure is None
    assert fine.metrics["settling_residual"] < 1e-5
    # Both solvers hold the clamped element rigid, which costs first-order accuracy.
    assert fine.metrics["tip_deflection_error"] < 0.08
    assert fine.metrics["tip_deflection_error"] < 0.6 * coarse.metrics["tip_deflection_error"]


def test_pendulum_swings_at_the_right_rate_without_losing_energy(solver):
    result = run(registry.load_experiment("pendulum"), solver)
    assert result.failure is None
    # Measured at the default 50 elements: pyelastica 1.4e-4, mujoco 2e-5.
    assert result.metrics["period_error"] < 3e-4
    assert abs(result.metrics["amplitude_change"]) < 1e-4


def test_a_cable_that_starts_as_it_hangs_stays_there(solver):
    # Started unstretched, this cable would bounce 1 mm along its length, and the bounce
    # would pump sideways vibration (see the pendulum experiment).
    pendulum = registry.load_experiment("pendulum").scenario
    rod = pendulum.rods[0]
    s = np.linspace(0.0, rod.length, 201)
    strain = static_strain(s, rod.length, rod.material, 9.81)
    stretched = np.concatenate(
        ([0.0], np.cumsum((2.0 + strain[1:] + strain[:-1]) / 2.0 * np.diff(s)))
    )
    hanging = dataclasses.replace(
        rod,
        centerline=tuple((0.0, 0.0, -z) for z in stretched),
        rest_arc_length=tuple(s),
    )
    scenario = dataclasses.replace(pendulum, rods=(hanging,), duration=0.5)
    tip = solver.run(scenario, n_elements=50, n_frames=101).positions[0][:, -1, 2]
    assert np.ptp(tip) < 5e-5


def eased(total, seconds=1.0, samples=41):
    t = np.linspace(0.0, seconds, samples)
    share = 0.5 - 0.5 * np.cos(np.pi * t / seconds)
    return tuple(t), tuple(tuple(total * f) for f in share)


@pytest.mark.parametrize(
    "displacement, rotation, tip",
    [
        ((0, 0, 0), (0, np.pi / 2, 0), (0, 0, -1)),  # turned a quarter about y
        ((0, 0.3, 0), (0, 0, 0), (1, 0.3, 0)),  # carried sideways
    ],
)
def test_a_driven_clamp_carries_the_rod_with_it(solver, displacement, rotation, tip):
    times, moved = eased(np.array(displacement, dtype=float))
    _, turned = eased(np.array(rotation, dtype=float))
    rod = Rod(
        centerline=((0, 0, 0), (1, 0, 0)),
        radius=0.02,
        material=Material(1e7, 1e7 / 3.0, 1000.0),
        start=EndCondition.CLAMPED,
        start_motion=Motion(times, moved, turned),
    )
    # Settling damping drags on the moving rod, so allow it time to catch up.
    scenario = Scenario(rods=(rod,), duration=6.0, quasi_static=True)
    final = solver.run(scenario, n_elements=20, n_frames=11).positions[0][-1]
    np.testing.assert_allclose(final[-1], tip, atol=1e-6)


def test_clamping_either_end_bends_a_cantilever_alike(solver):
    cantilever = registry.load_experiment("cantilever").scenario
    base = cantilever.rods[0]
    mirrored = Rod(
        centerline=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        radius=base.radius,
        material=base.material,
        end=EndCondition.CLAMPED,
        loads=(PointLoad(force=base.loads[0].force, at=End.START),),
    )
    flipped = dataclasses.replace(cantilever, rods=(mirrored,))
    usual = solver.run(cantilever, n_elements=20, n_frames=11).positions[0][-1, -1, 2]
    other = solver.run(flipped, n_elements=20, n_frames=11).positions[0][-1, 0, 2]
    # MuJoCo holds a far end with a slightly compliant weld: 0.2% more deflection.
    assert other == pytest.approx(usual, rel=5e-3)


def test_a_sliding_end_slides_under_a_pull(solver):
    material = Material(1e6, 1e6 / 3.0, 1000.0)
    radius, tension = 0.01, 1.0
    t = np.linspace(0.0, 1.0, 5)
    still = tuple((0.0, 0.0, 0.0) for _ in t)
    rod = Rod(
        centerline=((0, 0, 0), (1, 0, 0)),
        radius=radius,
        material=material,
        start=EndCondition.CLAMPED,
        end=EndCondition.CLAMPED,
        end_motion=Motion(tuple(t), still, still, slides_along=(1.0, 0.0, 0.0)),
        loads=(PointLoad(force=(tension, 0.0, 0.0)),),
    )
    # Settling damping is critical for the slowest mode only; the axial ringing a sudden
    # pull starts dies at half the damping rate, so give it time.
    scenario = Scenario(rods=(rod,), duration=15.0, quasi_static=True)
    tip = solver.run(scenario, n_elements=20, n_frames=11).positions[0][-1, -1]
    stretch = (
        tension / (material.youngs_modulus * rod.area)
        if "stretch" in {c.value for c in solver.capabilities}
        else 0.0
    )
    assert tip[0] - 1.0 == pytest.approx(stretch, rel=0.01, abs=2e-6)
    np.testing.assert_allclose(tip[1:], 0.0, atol=1e-9)


def test_directors_show_the_twist_a_turned_clamp_puts_in(solver):
    """Every solver reports the same material line: the scenario's directors at the
    start, unit and perpendicular to the rod throughout, and turned by as much as the
    clamp turned once the rod has settled (1 rad here, well below buckling)."""
    times, turned = eased(np.array([1.0, 0.0, 0.0]))
    still = tuple((0.0, 0.0, 0.0) for _ in times)
    rod = Rod(
        centerline=((0, 0, 0), (1, 0, 0)),
        radius=0.01,
        material=Material(1e6, 1e6 / 3.0, 1000.0),
        start=EndCondition.CLAMPED,
        end=EndCondition.CLAMPED,
        end_motion=Motion(times, still, turned, slides_along=(1.0, 0.0, 0.0)),
        loads=(PointLoad(force=(1.0, 0.0, 0.0)),),
    )
    scenario = Scenario(rods=(rod,), duration=6.0, quasi_static=True, self_contact=False)
    trajectory = solver.run(scenario, n_elements=20, n_frames=13)
    positions, directors = trajectory.positions[0], trajectory.directors[0]
    tangents = np.diff(positions, axis=1)
    tangents /= np.linalg.norm(tangents, axis=2, keepdims=True)
    np.testing.assert_allclose(np.linalg.norm(directors, axis=2), 1.0, atol=1e-9)
    assert np.abs((directors * tangents).sum(axis=2)).max() < 1e-5
    np.testing.assert_allclose(directors[0], rod.directors(20), atol=1e-9)
    total = twist_angles(positions, directors).sum(axis=1)
    assert total[0] == pytest.approx(0.0, abs=1e-9)
    assert total[-1] == pytest.approx(1.0, rel=1e-3)  # MuJoCo's weld gives 2e-4 of it back


def test_twist_buckles_near_greenhill(solver):
    if solver.name == "mujoco":
        pytest.skip("MuJoCo's welds diverge here; see the note in its adapter")
    if solver.name == "dismech":
        pytest.skip("hours per run: six rods in contact, a dense Newton solve every step")
    result = run(registry.load_experiment("twist"), solver)
    assert result.failure is None
    # Measured at the default 50 elements: 1.1%; the method itself is good to ~0.5%.
    assert result.metrics["critical_twist_error"] < 0.03


def test_settling_damping_lets_a_free_rod_fall_at_its_terminal_speed(solver):
    # Damping proportional to mass at rate c makes a free body fall at g / c. MuJoCo's
    # adapter once computed it from a stale mass matrix, and a free cable spun up.
    rod = Rod(((0, 0, 0), (1, 0, 0)), 0.01, Material(1e6, 1e6 / 3.0, 1000.0))
    scenario = Scenario(rods=(rod,), duration=3.0, gravity=(0.0, 0.0, -9.81), quasi_static=True)
    heights = solver.run(scenario, n_elements=20, n_frames=31).positions[0][:, :, 2].mean(axis=1)
    speed = (heights[-1] - heights[-2]) / 0.1
    assert speed == pytest.approx(-9.81 / (2.0 * scenario.slowest_frequency()), rel=0.02)


@pytest.mark.parametrize("overhang, slides", [(0.5, False), (1.5, True)])
def test_capstan_rope_holds_or_slides_as_friction_allows(solver, overhang, slides):
    result = completed(
        run(registry.load_experiment("capstan"), solver, values={"overhang": overhang})
    )
    assert result.outcome == "completed"
    if slides:
        assert result.metrics["slide"] > 0.1
    else:
        assert result.metrics["slide"] < 0.005
    assert result.observations["max_penetration"] < 0.5


def test_mujoco_frames_a_cable_that_starts_straight_then_curves():
    mujoco = pytest.importorskip("mujoco")
    from cosseratbench.experiments.capstan import capstan
    from cosseratbench.solvers.mujoco import _rod_xml

    body, _, _ = _rod_xml(0, capstan.scenario.rods[0], 100, 2e-4)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><extension><plugin plugin="mujoco.elasticity.cable"/></extension>'
        f"<worldbody>{body}</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = ["r0_B_first"] + [f"r0_B_{k}" for k in range(1, 99)] + ["r0_B_last"]
    quaternions = [data.xquat[model.body(name).id] for name in names]
    # Neighbouring segments start at most a small bend apart, never half a turn.
    turns = [2 * np.arccos(min(1.0, abs(float(a @ b)))) for a, b in itertools.pairwise(quaternions)]
    assert max(turns) < 0.2


def test_a_rod_dropped_on_a_floor_rests_on_it(solver, request):
    from cosseratbench import Plane

    known_failure(request, solver, {"dismech"}, FLOOR_FRICTION_STICKS)
    from cosseratbench.experiment import max_penetration

    rod = Rod(((-0.25, 0, 0.05), (0.25, 0, 0.05)), 0.005, Material(1e6, 1e6 / 3.0, 1000.0))
    floor = Plane(point=(0, 0, 0), normal=(0, 0, 1), friction=0.3)
    scenario = Scenario(
        rods=(rod,),
        obstacles=(floor,),
        duration=1.5,
        gravity=(0.0, 0.0, -9.81),
        quasi_static=True,
        self_contact=False,
    )
    trajectory = solver.run(scenario, n_elements=20, n_frames=16)
    heights = trajectory.positions[0][-1, :, 2]
    np.testing.assert_allclose(heights, 0.005, atol=2e-5)  # one radius up, all along
    assert max_penetration(scenario, trajectory) < 0.01


@pytest.mark.parametrize("friction, holds", [(0.5, True), (0.1, False)])
def test_a_rod_on_a_slope_holds_or_slides_as_friction_allows(solver, request, friction, holds):
    from cosseratbench import Plane

    if holds:
        known_failure(request, solver, {"dismech"}, FLOOR_FRICTION_STICKS)
    # A level floor under gravity tilted 20 degrees: a slope, as every solver can have it.
    angle = np.radians(20.0)  # tan 20 degrees is 0.36: between the two coefficients
    rod = Rod(((0, 0, 0.005), (0.5, 0, 0.005)), 0.005, Material(1e6, 1e6 / 3.0, 1000.0))
    floor = Plane(point=(0, 0, 0), normal=(0, 0, 1), friction=friction)
    scenario = Scenario(
        rods=(rod,),
        obstacles=(floor,),
        duration=1.5,
        gravity=(9.81 * np.sin(angle), 0.0, -9.81 * np.cos(angle)),
        quasi_static=True,
        self_contact=False,
    )
    trajectory = solver.run(scenario, n_elements=20, n_frames=16)
    moved = np.linalg.norm(trajectory.positions[0][-1] - trajectory.positions[0][0], axis=1).max()
    assert moved < 1e-3 if holds else moved > 0.2


def test_a_rope_fed_onto_a_floor_coils_on_it(solver, request):
    """The pile experiment at a coarse resolution: the rope lands, does not pass
    through itself, and coils in three dimensions rather than lying in a line."""
    known_failure(request, solver, {"dismech"}, FLOOR_FRICTION_STICKS)
    result = run(registry.load_experiment("pile"), solver, n_elements=50, n_frames=41)
    assert result.outcome == "completed", result.failure
    # PyElastica's plane contact acts at element centres, so the rope's end sinks half
    # an element (three diameters here) into the floor on landing; see the findings.
    assert result.observations["max_penetration"] < {"pyelastica": 8.0, "mujoco": 0.5}[solver.name]
    assert result.observations["max_rod_overlap"] < 1.0
    assert result.metrics["contacts"] >= 1
    assert result.metrics["footprint"] < 0.3
    final = result.trajectory.positions[0][-1]
    assert np.ptp(final[:, 0]) > 0.05 and np.ptp(final[:, 1]) > 0.05  # coiled, not folded flat
    assert (final[:, 2] < 0.05).mean() > 0.8  # and nearly all of it is down on the floor


def test_a_rod_dropped_across_another_lands_on_it_instead_of_through_it(solver):
    material = Material(youngs_modulus=1e6, shear_modulus=3e5, density=1000.0)
    held = Rod(
        ((-0.3, 0.0, 0.0), (0.3, 0.0, 0.0)),
        radius=0.01,
        material=material,
        start=EndCondition.CLAMPED,
        end=EndCondition.CLAMPED,
    )
    # Free at both ends, so it simply drops across the one below.
    falling = Rod(
        ((0.0, -0.3, 0.1), (0.0, 0.3, 0.1)),
        radius=0.01,
        material=material,
        normal=(1.0, 0.0, 0.0),
    )
    scenario = Scenario(
        rods=(held, falling),
        duration=1.0,
        gravity=(0.0, 0.0, -9.81),
        quasi_static=True,
        self_contact=False,
    )
    trajectory = solver.run(scenario, n_elements=31, n_frames=41)
    fell, held = trajectory.positions[1][:, 15, 2], trajectory.positions[0][:, 15, 2]
    assert fell[-1] < 0.1 - 0.05  # it dropped most of the way
    assert (fell > held).all()  # and stayed on top of the rod below throughout
    assert max_rod_overlap(scenario, trajectory) < 1.0  # without sinking a radius into it


def test_a_rod_left_to_itself_stays_out_of_its_own_way(solver):
    """Self-contact costs a solver time, so a scenario says when its rods can reach
    themselves. Turning it on must not disturb a rod that never does."""
    material = Material(youngs_modulus=1e6, shear_modulus=3e5, density=1000.0)
    rod = Rod(
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        radius=0.005,
        material=material,
        start=EndCondition.CLAMPED,
        loads=(PointLoad(force=(0.0, 0.0, -0.02)),),
    )
    scenario = Scenario(rods=(rod,), duration=1.0, quasi_static=True, self_contact=False)
    apart, together = (
        solver.run(dataclasses.replace(scenario, self_contact=looking), n_elements=20, n_frames=11)
        for looking in (False, True)
    )
    assert np.allclose(apart.positions[0][-1], together.positions[0][-1], atol=1e-9)
