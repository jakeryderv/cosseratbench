"""Run rod simulation benchmarks and view the results."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from cosseratbench import registry, site, variations
from cosseratbench.experiment import DIVERGED, UNSUPPORTED, Result, run


def _describe(result: Result) -> str:
    if result.outcome == UNSUPPORTED:
        return "unsupported (needs " + ", ".join(result.missing) + ")"
    if result.outcome == DIVERGED:
        when = f" at t = {result.diverged_at:.3g} s" if result.diverged_at is not None else ""
        return f"diverged{when}: {result.failure}"
    metrics = "  ".join(f"{name}={value:.3e}" for name, value in result.metrics.items())
    return f"{result.wall_time:6.1f}s  {metrics}"


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
    solvers = args.solver or registry.names(registry.SOLVERS)
    vary = args.vary or []
    # Each name applies where it means something; one that means nothing anywhere is a typo.
    sweepable = {name: variations.sweepable(e) for e in experiments for name in [e.name]}
    for name in vary:
        if name != "all" and not any(name in names for names in sweepable.values()):
            raise KeyError(f"nothing to vary called {name!r}; see `cosseratbench list`")
    for experiment in experiments:
        experiment_name = experiment.name
        defaults = variations.defaults(experiment)
        applicable = [n for n in vary if n == "all" or n in sweepable[experiment_name]]
        for variant in variations.variants(experiment, applicable):
            directory = args.out / experiment_name / variant.key
            experiment.save(directory, variant.parameters, variant.varied, defaults)
            for solver_name in solvers:
                label = f"{experiment_name:12s} {variant.key:24s} {solver_name:12s}"
                try:
                    solver = registry.load_solver(solver_name, **variant.options)
                except ImportError as error:
                    print(f"{label} not installed ({error.name})")
                    continue
                n_elements = variant.n_elements or args.n_elements
                result = run(
                    experiment,
                    solver,
                    values=variant.parameters,
                    options=variant.options,
                    n_elements=n_elements,
                )
                result.save(directory / solver_name)
                print(f"{label} n={result.n_elements:<4d} {_describe(result)}")


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
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(1, f"cosseratbench: {error}\n")
