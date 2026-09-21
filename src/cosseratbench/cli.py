"""Run rod simulation benchmarks and view the results."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from cosseratbench import registry, site
from cosseratbench.experiment import Result, run


def _describe(result: Result) -> str:
    if result.missing:
        return "unsupported (needs " + ", ".join(result.missing) + ")"
    if result.failure:
        return f"failed: {result.failure}"
    metrics = "  ".join(f"{name}={value:.3e}" for name, value in result.metrics.items())
    return f"{result.wall_time:6.1f}s  {metrics}"


def _list(_: argparse.Namespace) -> None:
    print("experiments:")
    for name in registry.names(registry.EXPERIMENTS):
        print(f"  {name:12s} {registry.load_experiment(name).description}")
    print("solvers:")
    for name in registry.names(registry.SOLVERS):
        print(f"  {name}")


def _run(args: argparse.Namespace) -> None:
    experiments = args.experiment or registry.names(registry.EXPERIMENTS)
    solvers = args.solver or registry.names(registry.SOLVERS)
    for experiment_name in experiments:
        experiment = registry.load_experiment(experiment_name)
        experiment.save(args.out / experiment_name)
        for solver_name in solvers:
            try:
                solver = registry.load_solver(solver_name)
            except ImportError as error:
                print(f"{experiment_name:12s} {solver_name:12s} not installed ({error.name})")
                continue
            result = run(experiment, solver, n_elements=args.n_elements)
            result.save(args.out / experiment_name / solver_name)
            print(
                f"{experiment_name:12s} {solver_name:12s} n={result.n_elements:<4d} {_describe(result)}"
            )


def _view(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as directory:
        site.build(args.results, Path(directory))
        site.serve(Path(directory), args.port, open_browser=not args.no_open)


def _site(args: argparse.Namespace) -> None:
    site.build(args.results, args.out)
    print(f"wrote {args.out}/")


def main() -> None:
    parser = argparse.ArgumentParser(prog="cosseratbench", description=__doc__)
    commands = parser.add_subparsers(required=True)

    commands.add_parser("list", help="show registered experiments and solvers").set_defaults(
        func=_list
    )

    run_parser = commands.add_parser("run", help="run experiments and save results")
    run_parser.add_argument("experiment", nargs="*", help="experiments to run (default: all)")
    run_parser.add_argument(
        "-s", "--solver", action="append", help="solver to use; repeatable (default: all)"
    )
    run_parser.add_argument(
        "-n", "--n-elements", type=int, help="elements per rod (default: per experiment)"
    )
    run_parser.add_argument(
        "-o", "--out", type=Path, default=Path("results"), help="output directory"
    )
    run_parser.set_defaults(func=_run)

    view_parser = commands.add_parser("view", help="open the results in a browser")
    view_parser.add_argument("-r", "--results", type=Path, default=Path("results"))
    view_parser.add_argument("-p", "--port", type=int, default=8000, help="0 picks a free port")
    view_parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    view_parser.set_defaults(func=_view)

    site_parser = commands.add_parser("site", help="write the viewer as a static website")
    site_parser.add_argument("out", type=Path, help="directory to write")
    site_parser.add_argument("-r", "--results", type=Path, default=Path("results"))
    site_parser.set_defaults(func=_site)

    args = parser.parse_args()
    try:
        args.func(args)
    except FileNotFoundError as error:
        parser.exit(1, f"cosseratbench: {error}\n")
