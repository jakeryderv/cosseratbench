import dataclasses
import json

import pytest
from test_core import STRAIGHT, FakeSolver, experiment

from cosseratbench import Scenario, registry, runs
from cosseratbench.provenance import scenario_digest


class Counting(FakeSolver):
    """A fake solver that counts its timed runs (a warm-up asks for two frames)."""

    timed = 0

    def run(self, scenario, *, n_elements, n_frames):
        if n_frames > 2:
            type(self).timed += 1
        return super().run(scenario, n_elements=n_elements, n_frames=n_frames)


class Broken(FakeSolver):
    def run(self, scenario, *, n_elements, n_frames):
        raise RuntimeError("an adapter bug")


@pytest.fixture
def fakes(monkeypatch):
    """A registry holding one fake experiment and fake solvers, as the runner loads them."""
    Counting.timed = 0
    experiments = {"still": experiment()}
    solvers = {"fake": Counting, "broken": Broken}
    monkeypatch.setattr(registry, "load_experiment", lambda name: experiments[name])
    monkeypatch.setattr(registry, "load_solver", lambda name, **options: solvers[name](**options))
    return experiments


def go(out, *, solvers=("fake",), n_elements=4, force=False):
    lines = []
    tasks = runs.plan(
        [registry.load_experiment("still")],
        solvers,
        [],
        n_elements,
        out,
        force=force,
        report=lines.append,
    )
    runs.execute(tasks, report=lines.append)
    return lines


def test_a_saved_run_is_reused_until_something_it_depended_on_changes(tmp_path, fakes):
    go(tmp_path)
    assert Counting.timed == 1
    lines = go(tmp_path)
    assert Counting.timed == 1 and lines[0].endswith("[saved]")

    go(tmp_path, force=True)  # asked to
    go(tmp_path, n_elements=6)  # a different resolution
    assert Counting.timed == 3

    Counting.backend_version = "2.0"  # a different version of the solver library
    try:
        go(tmp_path, n_elements=6)
    finally:
        del Counting.backend_version
    assert Counting.timed == 4

    (tmp_path / "still" / "default" / "fake" / "trajectory.npz").unlink()  # half missing
    go(tmp_path, n_elements=6)
    assert Counting.timed == 5


def test_a_changed_scenario_is_run_again(tmp_path, fakes):
    go(tmp_path)
    longer = dataclasses.replace(STRAIGHT, centerline=((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    fakes["still"] = experiment(build=lambda: Scenario(rods=(longer,), duration=1.0))
    go(tmp_path)
    assert Counting.timed == 2


def test_each_result_records_what_its_trajectory_depended_on(tmp_path, fakes):
    go(tmp_path)
    saved = json.loads((tmp_path / "still" / "default" / "fake" / "result.json").read_text())
    assert saved["provenance"]["scenario"] == scenario_digest(fakes["still"].scenario)
    assert len(saved["provenance"]["code"]) == 16
    assert saved["provenance"]["solver_version"] is None  # the fake does not say
    assert (saved["n_elements"], saved["n_frames"], saved["jobs"]) == (4, 101, 1)


def test_a_crashing_adapter_is_reported_and_the_other_runs_are_kept(tmp_path, fakes):
    with pytest.raises(RuntimeError, match="1 run"):
        go(tmp_path, solvers=("broken", "fake"))
    assert (tmp_path / "still" / "default" / "fake" / "result.json").exists()
    assert not (tmp_path / "still" / "default" / "broken" / "result.json").exists()


def test_rescore_recomputes_metrics_from_saved_trajectories(tmp_path, fakes):
    go(tmp_path)
    path = tmp_path / "still" / "default" / "fake" / "result.json"
    before = json.loads(path.read_text())
    fakes["still"] = experiment(
        metrics={"tip_y": lambda scenario, trajectory: trajectory.positions[0][-1, -1, 1]}
    )
    lines = []
    runs.rescore(tmp_path, report=lines.append)
    after = json.loads(path.read_text())
    assert Counting.timed == 1  # nothing simulated
    assert after["metrics"] == {"tip_y": 0.0}
    assert after["wall_time"] == before["wall_time"]
    assert "tip_y" in lines[0]


def test_rescore_leaves_a_run_whose_scenario_has_changed(tmp_path, fakes):
    go(tmp_path)
    path = tmp_path / "still" / "default" / "fake" / "result.json"
    before = path.read_text()
    longer = dataclasses.replace(STRAIGHT, centerline=((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    fakes["still"] = experiment(build=lambda: Scenario(rods=(longer,), duration=1.0))
    lines = []
    runs.rescore(tmp_path, report=lines.append)
    assert path.read_text() == before
    assert "rerun it" in lines[0]


@pytest.mark.slow
def test_runs_in_parallel_match_runs_one_at_a_time(tmp_path):
    pytest.importorskip("elastica")
    cantilever = registry.load_experiment("cantilever")
    results = {}
    for jobs in (1, 2):
        out = tmp_path / f"jobs{jobs}"
        tasks = runs.plan([cantilever], ["pyelastica"], ["resolution"], None, out, report=print)
        runs.execute(tasks, jobs=jobs, report=print)
        results[jobs] = {
            path.parent.parent.name: json.loads(path.read_text())
            for path in out.glob("cantilever/*/pyelastica/result.json")
        }
    assert results[1].keys() == results[2].keys() and len(results[1]) == 3
    for key, serial in results[1].items():
        parallel = results[2][key]
        assert parallel["metrics"] == serial["metrics"]
        assert (serial["jobs"], parallel["jobs"]) == (1, 2)


def test_a_run_records_the_numerical_settings_its_adapter_reports(tmp_path, fakes):
    class Reporting(Counting):
        def settings(self, scenario, *, n_elements, n_frames):
            return {"time_step": scenario.duration / (10 * (n_frames - 1)), "integrator": "fake"}

    from cosseratbench import run

    result = run(fakes["still"], Reporting(), n_elements=4, n_frames=11)
    assert result.settings == {"time_step": 0.01, "integrator": "fake"}
    assert run(fakes["still"], Counting(), n_elements=4).settings == {}  # it need not say


@pytest.mark.parametrize(
    "name, module", [("pyelastica", "elastica"), ("mujoco", "mujoco"), ("dismech", "dismech")]
)
def test_every_built_in_adapter_reports_its_settings(name, module):
    pytest.importorskip(module)
    capstan, pile = registry.load_experiment("capstan"), registry.load_experiment("pile")
    for scenario in (capstan.scenario, pile.scenario):
        usual = registry.load_solver(name).settings(scenario, n_elements=50, n_frames=101)
        coarse = registry.load_solver(name, time_step_scale=2.0).settings(
            scenario, n_elements=50, n_frames=101
        )
        json.dumps(usual)  # saved as it is
        assert usual["time_step"] > 0 and isinstance(usual["integrator"], str)
        assert coarse["time_step"] == pytest.approx(2 * usual["time_step"], rel=0.1)
