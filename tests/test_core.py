import json
import math

import numpy as np
import pytest

from cosseratbench import (
    Capability,
    EndCondition,
    Experiment,
    Material,
    Motion,
    Rod,
    Scenario,
    Trajectory,
    registry,
    run,
    variations,
)
from cosseratbench.experiment import Parameter
from cosseratbench.experiments.catenary import LENGTH, catenary, reference_curve
from cosseratbench.solver import Diverged

RUBBER = Material(youngs_modulus=1e6, shear_modulus=1e6 / 3.0, density=1000.0)
STRAIGHT = Rod(centerline=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), radius=0.01, material=RUBBER)


class FakeSolver:
    """Returns a rod that never moves, optionally scaled so every segment is `stretch` times longer."""

    name = "fake"

    def __init__(self, capabilities=frozenset(), stretch=1.0):
        self.capabilities = capabilities
        self.stretch = stretch

    def run(self, scenario, *, n_elements, n_frames):
        nodes = scenario.rods[0].nodes(n_elements)
        frames = np.repeat(nodes[None], n_frames, axis=0)
        frames[-1] *= self.stretch
        directors = np.repeat(scenario.rods[0].directors(n_elements)[None], n_frames, axis=0)
        return Trajectory(np.linspace(0.0, scenario.duration, n_frames), (frames,), (directors,))


def experiment(**overrides) -> Experiment:
    fields = {
        "name": "still",
        "description": "",
        "build": lambda: Scenario(rods=(STRAIGHT,), duration=1.0),
        "metrics": {"tip_x": lambda scenario, trajectory: trajectory.positions[0][-1, -1, 0]},
    }
    return Experiment(**(fields | overrides))


def test_rod_length_and_resampling_follow_the_centerline():
    bent = Rod(centerline=((0, 0, 0), (1, 0, 0), (1, 1, 0)), radius=0.01, material=RUBBER)
    assert bent.length == pytest.approx(2.0)
    nodes = bent.nodes(4)
    np.testing.assert_allclose(nodes, [(0, 0, 0), (0.5, 0, 0), (1, 0, 0), (1, 0.5, 0), (1, 1, 0)])


def test_directors_start_at_the_normal_and_follow_the_rod_round_a_bend():
    # A rod bent through a quarter turn in the x-y plane, with its normal out of it.
    bent = Rod(centerline=((0, 0, 0), (1, 0, 0), (1, 1, 0)), radius=0.01, material=RUBBER)
    directors = bent.directors(4)
    assert directors.shape == (4, 3)
    np.testing.assert_allclose(directors, [(0, 0, 1)] * 4, atol=1e-12)  # carried unchanged

    # With the normal in the plane of the bend, it turns with the rod: no twist either way.
    in_plane = Rod(bent.centerline, 0.01, RUBBER, normal=(0.0, 1.0, 0.0))
    directors = in_plane.directors(4)
    np.testing.assert_allclose(directors[:2], [(0, 1, 0)] * 2, atol=1e-12)
    np.testing.assert_allclose(directors[2:], [(-1, 0, 0)] * 2, atol=1e-12)
    with pytest.raises(ValueError, match="normal"):
        Rod(STRAIGHT.centerline, 0.01, RUBBER, normal=(1.0, 0.0, 0.0)).directors(2)


def test_twist_is_the_turn_of_the_director_beyond_what_the_bend_carries():
    from cosseratbench.metrics import twist_angles

    # A straight rod whose director turns 0.1 rad per element: 0.1 rad of twist per node.
    nodes = np.stack([np.linspace(0, 1, 6), np.zeros(6), np.zeros(6)], axis=1)
    angles = 0.1 * np.arange(5)
    directors = np.stack([np.zeros(5), -np.sin(angles), np.cos(angles)], axis=1)
    np.testing.assert_allclose(twist_angles(nodes, directors), 0.1, atol=1e-12)
    # A bent, untwisted rod: none, whichever way its normal points.
    for normal in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.3, 0.5, 0.8)):
        bent = Rod(((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)), 0.01, RUBBER, normal=normal)
        np.testing.assert_allclose(twist_angles(bent.nodes(9), bent.directors(9)), 0.0, atol=1e-12)
    # Frames stack: a whole trajectory at once.
    trajectory = np.repeat(nodes[None], 3, axis=0), np.repeat(directors[None], 3, axis=0)
    assert twist_angles(*trajectory).shape == (3, 4)


def test_a_trajectory_needs_a_director_for_every_element():
    times = np.array([0.0, 1.0])
    positions = np.zeros((2, 5, 3))
    Trajectory(times, (positions,), (np.zeros((2, 4, 3)),))
    with pytest.raises(ValueError, match="directors"):
        Trajectory(times, (positions,), ())
    with pytest.raises(ValueError, match="directors"):
        Trajectory(times, (positions,), (np.zeros((2, 5, 3)),))


def test_a_rod_can_start_stretched():
    # Two unstretched metres drawn out to three: the middle material point sits halfway.
    rod = Rod(
        centerline=((0, 0, 0), (3, 0, 0)), radius=0.01, material=RUBBER, rest_arc_length=(0.0, 2.0)
    )
    assert rod.length == pytest.approx(2.0)
    np.testing.assert_allclose(rod.nodes(2), [(0, 0, 0), (1.5, 0, 0), (3, 0, 0)])
    with pytest.raises(ValueError):
        Rod(centerline=((0, 0, 0), (1, 0, 0)), radius=0.01, material=RUBBER, rest_arc_length=(0.0,))


def test_motion_interpolates_between_times_and_then_holds():
    motion = Motion(
        times=(0.0, 2.0),
        displacement=((0, 0, 0), (0.2, 0, 0)),
        rotation=((0, 0, 0), (0, 0, np.pi)),
    )
    displacement, rotation = motion.pose(1.0)
    np.testing.assert_allclose(displacement, (0.1, 0, 0))
    np.testing.assert_allclose(rotation @ (1, 0, 0), (0, 1, 0), atol=1e-12)  # a quarter turn
    np.testing.assert_allclose(motion.pose(5.0)[0], (0.2, 0, 0))
    assert not np.any(motion.rates(5.0)[0]) and not np.any(motion.rates(5.0)[1])


def test_motion_rates_are_the_derivatives_of_its_pose():
    motion = Motion(
        times=(0.0, 1.0, 2.5),
        displacement=((0, 0, 0), (0.1, -0.2, 0.3), (0.4, 0, 0)),
        rotation=((0, 0, 0), (0.3, 1.2, -0.5), (2.0, -0.4, 1.1)),  # not about one axis
    )
    h = 1e-6
    for t in (0.3, 0.9, 1.7, 2.4):
        (d1, r1), (d2, r2) = motion.pose(t - h), motion.pose(t + h)
        spin = (r2 - r1) / (2 * h) @ motion.pose(t)[1].T  # skew matrix of the angular velocity
        velocity, angular_velocity = motion.rates(t)
        np.testing.assert_allclose(velocity, (d2 - d1) / (2 * h), atol=1e-8)
        np.testing.assert_allclose(
            angular_velocity, (spin[2, 1], spin[0, 2], spin[1, 0]), atol=1e-7
        )


def test_motion_must_start_at_rest_where_the_end_is():
    with pytest.raises(ValueError):
        Motion(times=(0.0, 1.0), displacement=((0.1, 0, 0), (0, 0, 0)), rotation=((0, 0, 0),) * 2)
    with pytest.raises(ValueError):
        Motion(times=(1.0, 2.0), displacement=((0, 0, 0),) * 2, rotation=((0, 0, 0),) * 2)


def test_a_sliding_end_turns_only_about_its_slide_direction():
    zero = ((0, 0, 0),) * 2
    Motion(
        times=(0.0, 1.0), displacement=zero, rotation=((0, 0, 0), (2, 0, 0)), slides_along=(1, 0, 0)
    )
    with pytest.raises(ValueError, match="slide"):
        Motion((0.0, 1.0), zero, ((0, 0, 0), (0, 2, 0)), slides_along=(1, 0, 0))
    with pytest.raises(ValueError, match="slide"):
        Motion((0.0, 1.0), ((0, 0, 0), (0.1, 0, 0)), zero, slides_along=(1, 0, 0))


def test_only_a_clamped_end_can_be_driven():
    turn = Motion(times=(0.0, 1.0), displacement=((0, 0, 0),) * 2, rotation=((0, 0, 0), (0, 0, 1)))
    with pytest.raises(ValueError, match="clamped"):
        Rod(STRAIGHT.centerline, 0.01, RUBBER, start=EndCondition.PINNED, start_motion=turn)
    Rod(STRAIGHT.centerline, 0.01, RUBBER, end=EndCondition.CLAMPED, end_motion=turn)


def test_catenary_start_shape_has_the_right_length_and_span():
    rod = catenary.scenario.rods[0]
    assert rod.length == pytest.approx(LENGTH, rel=1e-5)
    assert rod.centerline[0] == pytest.approx((0.0, 0.0, 0.0))
    assert rod.centerline[-1] == pytest.approx((0.8, 0.0, 0.0), abs=1e-12)


def test_elastic_catenary_spans_the_supports_and_tends_to_the_inextensible_one():
    rod = catenary.scenario.rods[0]
    curve = reference_curve(rod, 9.81, 0.8)
    np.testing.assert_allclose(curve[[0, -1]], [(0, 0, 0), (0.8, 0, 0)], atol=1e-9)

    stiff = Rod(rod.centerline, rod.radius, Material(1e12, 1e12 / 3.0, 1000.0))
    inextensible = reference_curve(stiff, 9.81, 0.8)
    arc_length = np.linalg.norm(np.diff(inextensible, axis=0), axis=1).sum()
    assert arc_length == pytest.approx(stiff.length, rel=1e-6)
    assert curve[:, 2].min() < inextensible[:, 2].min()  # stretch deepens the sag


def test_trajectory_round_trips_through_disk(tmp_path):
    trajectory = FakeSolver().run(experiment().scenario, n_elements=4, n_frames=3)
    trajectory.save(tmp_path / "t.npz")
    loaded = Trajectory.load(tmp_path / "t.npz")
    np.testing.assert_array_equal(loaded.times, trajectory.times)
    np.testing.assert_array_equal(loaded.positions[0], trajectory.positions[0])
    np.testing.assert_array_equal(loaded.directors[0], trajectory.directors[0])


def test_run_computes_metrics_from_the_trajectory():
    result = run(experiment(), FakeSolver(), n_elements=4)
    assert result.supported and result.failure is None
    assert result.metrics == {"tip_x": 2.0}


def test_run_warms_the_solver_up_on_a_short_version_before_the_timed_run():
    calls = []

    class Recording(FakeSolver):
        def run(self, scenario, *, n_elements, n_frames):
            calls.append((scenario.duration, n_frames))
            return super().run(scenario, n_elements=n_elements, n_frames=n_frames)

    run(experiment(), Recording(), n_elements=4, n_frames=11)
    assert calls == [(pytest.approx(1e-3), 2), (1.0, 11)]


def test_a_metric_that_cannot_be_computed_is_saved_as_null(tmp_path):
    result = run(
        experiment(metrics={"broken": lambda s, t: float("nan")}), FakeSolver(), n_elements=4
    )
    result.save(tmp_path)
    assert json.loads((tmp_path / "result.json").read_text())["metrics"] == {"broken": None}


def test_run_reports_missing_capabilities_without_running():
    result = run(experiment(requires=frozenset({Capability.STRETCH})), FakeSolver())
    assert not result.supported
    assert result.missing == ("stretch",)
    assert result.trajectory is None


def test_run_flags_a_blown_up_trajectory_instead_of_scoring_it():
    result = run(experiment(), FakeSolver(stretch=50.0), n_elements=4)
    assert "stretched" in result.failure
    assert result.metrics == {}


def test_builtins_are_registered_through_entry_points():
    assert {"catenary", "cantilever"} <= set(registry.names(registry.EXPERIMENTS))
    assert {"pyelastica", "mujoco"} <= set(registry.names(registry.SOLVERS))
    assert registry.load_experiment("catenary") is catenary
    with pytest.raises(KeyError):
        registry.load_solver("no-such-solver")


def test_parameters_build_the_scenario_and_default_to_the_ordinary_case():
    def build(length):
        return Scenario(rods=(Rod(((0, 0, 0), (length, 0, 0)), 0.01, RUBBER),), duration=1.0)

    sized = experiment(build=build, parameters=(Parameter("length", 2.0, (1.0, 2.0, 3.0)),))
    assert sized.scenario.rods[0].length == pytest.approx(2.0)
    assert sized.scenario_for(length=3.0).rods[0].length == pytest.approx(3.0)
    with pytest.raises(KeyError):
        sized.scenario_for(width=1.0)
    with pytest.raises(ValueError):
        Parameter("length", 2.5, (1.0, 2.0))  # the default must be swept


def test_outcomes_name_what_happened():
    assert run(experiment(), FakeSolver(), n_elements=4).outcome == "completed"
    assert (
        run(experiment(requires=frozenset({Capability.STRETCH})), FakeSolver()).outcome
        == "unsupported"
    )
    diverged = run(experiment(), FakeSolver(stretch=50.0), n_elements=4, n_frames=5)
    assert diverged.outcome == "diverged"
    assert diverged.diverged_at == pytest.approx(1.0)  # the last frame, where it broke


def test_a_solver_can_say_when_it_diverged():
    class Breaks(FakeSolver):
        def run(self, scenario, *, n_elements, n_frames):
            if n_frames > 2:  # not the warm-up
                raise Diverged("broke", time=0.25)
            return super().run(scenario, n_elements=n_elements, n_frames=n_frames)

    result = run(experiment(), Breaks(), n_elements=4)
    assert (result.outcome, result.failure, result.diverged_at) == ("diverged", "broke", 0.25)


def test_every_completed_run_observes_the_largest_strain():
    result = run(experiment(), FakeSolver(stretch=1.5), n_elements=4, n_frames=3)
    assert result.observations["max_strain"] == pytest.approx(0.5)


def test_variants_change_one_thing_at_a_time():
    sized = experiment(parameters=(Parameter("length", 2.0, (1.0, 2.0, 3.0)),), n_elements=10)
    keys = [v.key for v in variations.variants(sized, ["length", "resolution", "time_step_scale"])]
    assert keys == [
        "default",
        "length=1",
        "length=3",
        "resolution=5",
        "resolution=20",
        "time_step_scale=2",
        "time_step_scale=4",
    ]
    assert len(variations.variants(sized, ["all"])) == len(keys)
    with pytest.raises(KeyError):
        variations.variants(sized, ["width"])


def test_solvers_are_built_with_options():
    solver = registry.load_solver("pyelastica", time_step_scale=2.0)
    assert solver.time_step_safety == pytest.approx(1.0)
    with pytest.raises(ValueError):
        registry.load_solver("pyelastica", bogus=1.0)


def test_penetration_is_how_deep_nodes_sink_into_obstacles():
    from cosseratbench import Cylinder
    from cosseratbench.experiment import max_penetration

    cylinder = Cylinder(center=(0, 0, 0), axis=(0, 1, 0), radius=0.1, length=1.0)
    rod = Rod(((-1, 0, 0.12), (1, 0, 0.12)), 0.01, RUBBER)  # resting 1 cm above the surface
    positions = np.zeros((2, 3, 3))
    positions[:, :, 0] = (-1.0, 0.0, 1.0)
    positions[:, :, 2] = 0.11  # just touching
    positions[1, 1, 2] = 0.105  # its middle node sinks half its radius
    touching = Scenario(rods=(rod,), obstacles=(cylinder,), duration=1.0)
    trajectory = Trajectory(np.array([0.0, 1.0]), (positions,), (np.zeros((2, 2, 3)),))
    assert max_penetration(touching, trajectory) == pytest.approx(0.5)
    assert math.isnan(max_penetration(Scenario(rods=(rod,), duration=1.0), trajectory))


def test_penetration_is_how_deep_nodes_sink_into_a_plane():
    from cosseratbench import Plane
    from cosseratbench.experiment import max_penetration

    floor = Plane(point=(0, 0, 0.5), normal=(0, 0, 2.0))  # a floor at z = 0.5, facing up
    rod = Rod(((-1, 0, 1), (1, 0, 1)), 0.01, RUBBER)
    positions = np.zeros((2, 3, 3))
    positions[:, :, 0] = (-1.0, 0.0, 1.0)
    positions[:, :, 2] = 0.51  # just touching
    positions[1, 1, 2] = 0.5075  # its middle node sinks a quarter of its radius
    scenario = Scenario(rods=(rod,), obstacles=(floor,), duration=1.0)
    trajectory = Trajectory(np.array([0.0, 1.0]), (positions,), (np.zeros((2, 2, 3)),))
    assert max_penetration(scenario, trajectory) == pytest.approx(0.25)
    assert floor.kind == "plane"  # so a saved scenario says what shape each obstacle is


def test_segment_distance_finds_the_closest_approach_of_two_segments():
    from cosseratbench.metrics import segment_distance

    rng = np.random.default_rng(0)

    def sampled(a0, a1, b0, b1, n=600):
        t = np.linspace(0.0, 1.0, n)[:, None]
        first, second = a0 + t * (a1 - a0), b0 + t * (b1 - b0)
        return np.linalg.norm(first[:, None] - second[None, :], axis=2).min()

    for trial in range(60):
        a0, a1, b0, b1 = rng.normal(size=(4, 3))
        if trial % 3 == 0:  # parallel, where the closest points are not unique
            b1 = b0 + (a1 - a0) * rng.normal()
        if trial % 7 == 0:  # a segment of no length
            a1 = a0.copy()
        assert segment_distance(a0, a1, b0, b1) == pytest.approx(sampled(a0, a1, b0, b1), abs=1e-4)

    # Crossing at a right angle, one above the other: the gap is the height between them.
    assert segment_distance(
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.3]),
        np.array([0.0, 1.0, 0.3]),
    ) == pytest.approx(0.3)


def test_overlap_is_how_far_rods_pass_into_each_other():
    from cosseratbench.experiment import max_rod_overlap

    def crossed(height):
        along = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        across = np.array([[0.0, -1.0, height], [0.0, 0.0, height], [0.0, 1.0, height]])
        return along, across

    rod = Rod(((-1, 0, 0), (1, 0, 0)), 0.1, RUBBER)
    other = Rod(((0, -1, 0.5), (0, 1, 0.5)), 0.1, RUBBER)
    scenario = Scenario(rods=(rod, other), duration=1.0)
    clear, touching, sunk = (crossed(h) for h in (0.5, 0.2, 0.15))
    frames = Trajectory(
        np.arange(3.0),
        (np.stack([clear[0], touching[0], sunk[0]]), np.stack([clear[1], touching[1], sunk[1]])),
        (np.zeros((3, 2, 3)),) * 2,
    )
    # They touch at 0.2 apart and the deepest frame is 0.05 in, half a radius.
    assert max_rod_overlap(scenario, frames) == pytest.approx(0.5)


def test_overlap_ignores_the_neighbours_a_rod_cannot_bend_back_onto():
    from cosseratbench.experiment import max_rod_overlap

    # A straight rod, too thick to double back within its own length: nothing to report.
    rod = Rod(((0, 0, 0), (1, 0, 0)), 0.3, RUBBER)
    positions = rod.nodes(6)[None, :, :]
    scenario = Scenario(rods=(rod,), duration=1.0)
    still = Trajectory(np.array([0.0]), (positions,), (np.zeros((1, 6, 3)),))
    assert math.isnan(max_rod_overlap(scenario, still))

    # Folded in half, the two halves lie on top of each other and it is reported.
    thin = Rod(((0, 0, 0), (1, 0, 0)), 0.02, RUBBER)
    folded = np.array(
        [[x, 0.0, 0.0] for x in np.linspace(0, 0.5, 11)]
        + [[x, 0.0, 0.01] for x in np.linspace(0.45, 0.0, 10)]
    )
    scenario = Scenario(rods=(thin,), duration=1.0)
    folded_frames = Trajectory(np.array([0.0]), (folded[None],), (np.zeros((1, 20, 3)),))
    overlap = max_rod_overlap(scenario, folded_frames)
    assert overlap == pytest.approx((0.04 - 0.01) / 0.02, rel=1e-6)


def test_a_scenario_says_whether_its_rods_can_reach_themselves():
    from cosseratbench.scenario import neighbour_elements

    assert Scenario(rods=(), duration=1.0).self_contact is True
    # A rod in coarse pieces can only touch itself two elements or more apart; one cut
    # finer than its own thickness needs a longer stretch to bend back on.
    assert neighbour_elements(radius=0.01, segment_length=1.0) == 2
    assert neighbour_elements(radius=0.01, segment_length=0.005) == 7
