import json

import numpy as np
import pytest
from test_core import FakeSolver, experiment

from cosseratbench import Capability, EndCondition, Trajectory, run, site, variations
from cosseratbench.experiments.catenary import catenary


def write_results(results, *runs):
    for exp, solver in runs:
        exp.save(results / exp.name / "default")
        run(exp, solver, n_elements=4, n_frames=3).save(
            results / exp.name / "default" / solver.name
        )


def test_experiment_save_is_plain_json_a_reader_can_interpret(tmp_path):
    catenary.save(tmp_path)
    saved = json.loads((tmp_path / "experiment.json").read_text())
    rod = saved["scenario"]["rods"][0]
    assert rod["start"] == EndCondition.PINNED.value
    assert saved["rod_lengths"] == [pytest.approx(1.0, rel=1e-5)]
    assert np.asarray(saved["reference"]).shape[1:] == (2001, 3)  # one curve per rod it covers


def test_build_writes_the_viewer_a_manifest_and_browser_readable_trajectories(tmp_path):
    results, out = tmp_path / "results", tmp_path / "site"
    write_results(results, (experiment(), FakeSolver()))
    site.build(results, out)

    assert {"index.html", "viewer.js", "style.css"} <= {p.name for p in out.iterdir()}
    manifest = json.loads((out / "data" / "manifest.json").read_text())
    assert manifest["solvers"] == ["fake"]
    (listed,) = manifest["experiments"]
    (variant,) = listed["variants"]
    assert variant["key"] == "default"
    (listed_run,) = variant["runs"]
    assert listed_run["metrics"] == {"tip_x": 2.0}
    assert listed_run["trajectory"]["n_nodes"] == [5]
    assert listed_run["trajectory"]["times"] == [0.0, 0.5, 1.0]

    # The binary the browser reads holds the saved trajectory: positions, then directors.
    served = np.fromfile(out / listed_run["trajectory"]["file"], dtype="<f4")
    saved = Trajectory.load(results / "still" / "default" / "fake" / "trajectory.npz")
    positions, directors = (
        served[: 3 * 5 * 3].reshape(3, 5, 3),
        served[3 * 5 * 3 :].reshape(3, 4, 3),
    )
    np.testing.assert_allclose(positions, saved.positions[0], rtol=1e-6)
    np.testing.assert_allclose(directors, saved.directors[0], rtol=1e-6)


def test_build_lists_runs_without_a_trajectory_and_explains_them(tmp_path):
    results, out = tmp_path / "results", tmp_path / "site"
    needs_stretch = experiment(requires=frozenset({Capability.STRETCH}))
    write_results(results, (needs_stretch, FakeSolver()))
    site.build(results, out)

    manifest = json.loads((out / "data" / "manifest.json").read_text())
    (listed_run,) = manifest["experiments"][0]["variants"][0]["runs"]
    assert listed_run["missing"] == ["stretch"]
    assert "trajectory" not in listed_run


def test_build_shortens_a_long_reference_curve(tmp_path):
    results, out = tmp_path / "results", tmp_path / "site"
    write_results(results, (catenary, FakeSolver()))
    site.build(results, out)

    manifest = json.loads((out / "data" / "manifest.json").read_text())
    curves = manifest["experiments"][0]["variants"][0]["reference"]
    reference = np.asarray(curves[0])
    assert len(curves) == 1 and len(reference) == 201
    np.testing.assert_allclose(reference[[0, -1], 0], [0.0, 0.8], atol=1e-9)  # ends survive


def test_build_drops_runs_left_over_from_an_earlier_build(tmp_path):
    results, out = tmp_path / "results", tmp_path / "site"
    write_results(results, (experiment(), FakeSolver()))
    site.build(results, out)
    stale = out / "data" / "gone" / "old.f32"
    stale.parent.mkdir()
    stale.write_bytes(b"")
    site.build(results, out)
    assert not stale.exists()


def test_build_says_what_to_do_when_there_are_no_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="cosseratbench run"):
        site.build(tmp_path, tmp_path / "site")


def test_build_groups_variants_under_their_experiment(tmp_path):
    results, out = tmp_path / "results", tmp_path / "site"
    for variant in variations.variants(catenary, ["span"]):
        directory = results / "catenary" / variant.key
        catenary.save(directory, variant.parameters, variant.varied, variations.defaults(catenary))
        run(catenary, FakeSolver(), values=variant.parameters, n_elements=4, n_frames=3).save(
            directory / "fake"
        )
    site.build(results, out)
    (listed,) = json.loads((out / "data" / "manifest.json").read_text())["experiments"]
    assert [v["key"] for v in listed["variants"]] == ["default", "span=0.6", "span=0.95"]
    assert listed["variants"][1]["varied"] == {"span": 0.6}
    assert listed["defaults"]["span"] == 0.8
    assert {p["name"] for p in listed["parameters"]} == {"youngs_modulus", "span"}
    files = {v["runs"][0]["trajectory"]["file"] for v in listed["variants"]}
    assert len(files) == 3  # each variant's trajectory kept apart
