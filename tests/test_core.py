import json

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
)
from cosseratbench.experiments.catenary import LENGTH, SPAN, catenary, reference_curve

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
        return Trajectory(np.linspace(0.0, scenario.duration, n_frames), (frames,))


def experiment(**overrides) -> Experiment:
    fields = {
        "name": "still",
        "description": "",
        "scenario": Scenario(rods=(STRAIGHT,), duration=1.0),
        "metrics": {"tip_x": lambda scenario, trajectory: trajectory.positions[0][-1, -1, 0]},
    }
    return Experiment(**(fields | overrides))


def test_rod_length_and_resampling_follow_the_centerline():
    bent = Rod(centerline=((0, 0, 0), (1, 0, 0), (1, 1, 0)), radius=0.01, material=RUBBER)
    assert bent.length == pytest.approx(2.0)
    nodes = bent.nodes(4)
    np.testing.assert_allclose(nodes, [(0, 0, 0), (0.5, 0, 0), (1, 0, 0), (1, 0.5, 0), (1, 1, 0)])


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


def test_only_a_clamped_end_can_be_driven():
    turn = Motion(times=(0.0, 1.0), displacement=((0, 0, 0),) * 2, rotation=((0, 0, 0), (0, 0, 1)))
    with pytest.raises(ValueError, match="clamped"):
        Rod(STRAIGHT.centerline, 0.01, RUBBER, start=EndCondition.PINNED, start_motion=turn)
    Rod(STRAIGHT.centerline, 0.01, RUBBER, end=EndCondition.CLAMPED, end_motion=turn)


def test_catenary_start_shape_has_the_right_length_and_span():
    rod = catenary.scenario.rods[0]
    assert rod.length == pytest.approx(LENGTH, rel=1e-5)
    assert rod.centerline[0] == pytest.approx((0.0, 0.0, 0.0))
    assert rod.centerline[-1] == pytest.approx((SPAN, 0.0, 0.0), abs=1e-12)


def test_elastic_catenary_spans_the_supports_and_tends_to_the_inextensible_one():
    rod = catenary.scenario.rods[0]
    curve = reference_curve(rod, 9.81)
    np.testing.assert_allclose(curve[[0, -1]], [(0, 0, 0), (SPAN, 0, 0)], atol=1e-9)

    stiff = Rod(rod.centerline, rod.radius, Material(1e12, 1e12 / 3.0, 1000.0))
    inextensible = reference_curve(stiff, 9.81)
    arc_length = np.linalg.norm(np.diff(inextensible, axis=0), axis=1).sum()
    assert arc_length == pytest.approx(stiff.length, rel=1e-6)
    assert curve[:, 2].min() < inextensible[:, 2].min()  # stretch deepens the sag


def test_trajectory_round_trips_through_disk(tmp_path):
    trajectory = FakeSolver().run(experiment().scenario, n_elements=4, n_frames=3)
    trajectory.save(tmp_path / "t.npz")
    loaded = Trajectory.load(tmp_path / "t.npz")
    np.testing.assert_array_equal(loaded.times, trajectory.times)
    np.testing.assert_array_equal(loaded.positions[0], trajectory.positions[0])


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
