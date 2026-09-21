"""Turn a results directory into a static website that plays the trajectories back.

The site is plain files, so it can be served locally or hosted anywhere static.
Browsers cannot read ``.npz``, so each trajectory is rewritten as raw
little-endian float32, rod after rod, each ``[n_frames, n_nodes, 3]``; the
manifest carries the shapes.
"""

from __future__ import annotations

import functools
import json
import shutil
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np

from cosseratbench.trajectory import Trajectory

_REFERENCE_POINTS = 201  # plenty for a smooth line, small in the manifest


def _run(result_dir: Path, experiment: dict, data_dir: Path) -> dict:
    run = json.loads((result_dir / "result.json").read_text())
    path = result_dir / "trajectory.npz"
    if path.exists():
        trajectory = Trajectory.load(path)
        file = data_dir / experiment["name"] / f"{run['solver']}.f32"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"".join(rod.astype("<f4").tobytes() for rod in trajectory.positions))
        run["trajectory"] = {
            "file": file.relative_to(data_dir.parent).as_posix(),
            "times": trajectory.times.tolist(),
            "n_nodes": [rod.shape[1] for rod in trajectory.positions],
        }
    return run


def _experiment(experiment_dir: Path, data_dir: Path) -> dict:
    experiment = json.loads((experiment_dir / "experiment.json").read_text())
    if experiment["reference"] is not None:
        curve = np.asarray(experiment["reference"])
        keep = np.linspace(0, len(curve) - 1, min(len(curve), _REFERENCE_POINTS)).round()
        experiment["reference"] = curve[keep.astype(int)].tolist()
    experiment["runs"] = [
        _run(result.parent, experiment, data_dir)
        for result in sorted(experiment_dir.glob("*/result.json"))
    ]
    return experiment


def build(results: Path, out: Path) -> None:
    """Write the viewer and the data it needs for every experiment under ``results`` to ``out``."""
    experiments = sorted(results.glob("*/experiment.json"))
    if not experiments:
        raise FileNotFoundError(f"no results under {results}/; run `cosseratbench run` first")

    with as_file(files("cosseratbench") / "viewer") as viewer:
        shutil.copytree(viewer, out, dirs_exist_ok=True)
    data_dir = out / "data"
    shutil.rmtree(data_dir, ignore_errors=True)  # no stale runs from an earlier build
    data_dir.mkdir()
    manifest = {"experiments": [_experiment(path.parent, data_dir) for path in experiments]}
    # One list for the whole site, so a solver keeps its colour from one experiment to the next.
    manifest["solvers"] = sorted(
        {run["solver"] for experiment in manifest["experiments"] for run in experiment["runs"]}
    )
    (data_dir / "manifest.json").write_text(json.dumps(manifest))


class _FreshHandler(SimpleHTTPRequestHandler):
    """Results change between runs under the same URLs, so the browser must not reuse old ones."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass  # one line per request would bury the URL the user needs


def serve(directory: Path, port: int, open_browser: bool) -> None:
    handler = functools.partial(_FreshHandler, directory=str(directory))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"viewer at {url} (Ctrl+C to stop)", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
