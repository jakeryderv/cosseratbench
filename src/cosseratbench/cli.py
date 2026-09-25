"""Run rod simulation benchmarks and view the results."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from cosseratbench import registry, runs, site, variations


def _list(_: argparse.Namespace) -> None:
    print("experiments:")
    for name in registry.names(registry.EXPERIMENTS):
        experiment = registry.load_experiment(name)
        print(f"  {name:12s} {experiment.description}")
        print(f"  {'':12s} can vary: {', '.join(variations.sweepable(experiment))}")
    print("solvers:")
    for name in registry.names(registry.SOLVERS):
        print(f"  {name}")


def _run(args: argparse.Namespace) -> None:
    experiments = [
        registry.load_experiment(name)
        for name in args.experiment or registry.names(registry.EXPERIMENTS)
    ]
    tasks = runs.plan(
        experiments,
        args.solver or registry.names(registry.SOLVERS),
        args.vary or [],
        args.n_elements,
        args.out,
        force=args.force,
    )
    runs.execute(tasks, jobs=args.jobs)


def _rescore(args: argparse.Namespace) -> None:
    runs.rescore(args.results, args.experiment)


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
    run_parser.add_argument(
        "-v",
        "--vary",
        action="append",
        metavar="NAME",
        help="also run each other value of this parameter or solver option, one at a time; "
        "repeatable, or 'all' (see `cosseratbench list`)",
    )
    run_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="runs at once (default 1). Above 1 they compete for the machine and their wall "
        "times read high; each result records how many ran together. Keep it below the "
        "number of physical cores.",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even runs whose saved result still stands (same scenario, resolution, "
        "options, solver version and adapter code)",
    )
    run_parser.set_defaults(func=_run)

    rescore_parser = commands.add_parser(
        "rescore",
        help="recompute metrics and observations from saved trajectories, without simulating",
    )
    rescore_parser.add_argument(
        "experiment", nargs="*", help="experiments to rescore (default: all)"
    )
    rescore_parser.add_argument("-r", "--results", type=Path, default=Path("results"))
    rescore_parser.set_defaults(func=_rescore)

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
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"cosseratbench: {error}\n")
