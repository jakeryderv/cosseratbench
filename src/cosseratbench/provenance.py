"""What a run's trajectory depended on, recorded with its result.

A saved run can be reused only while everything that shaped its trajectory is
unchanged: the scenario, the resolution and frames asked for, the solver's
options, the solver library, and the code that turned the scenario into calls to
it. The first four are recorded as they are; the code is recorded as a digest of
the adapter's source and of the scenario module whose geometry every adapter
builds from. Metrics are not part of it: they can be recomputed from a saved
trajectory at any time (decision 0002).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from cosseratbench import scenario as _scenario_module
from cosseratbench.scenario import Scenario


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def scenario_digest(scenario: Scenario) -> str:
    """A short digest of everything the scenario says."""
    text = json.dumps(
        asdict(scenario),
        sort_keys=True,
        default=lambda value: value.value if isinstance(value, Enum) else repr(value),
    )
    return _digest(text.encode())


def code_digest(solver: object) -> str:
    """A short digest of the source of the solver's adapter and of the scenario module."""
    files = (Path(inspect.getfile(type(solver))), Path(inspect.getfile(_scenario_module)))
    return _digest(b"\0".join(path.read_bytes() for path in files))


def provenance(scenario: Scenario, solver: object) -> dict[str, str | None]:
    """What a run of ``solver`` on ``scenario`` depends on, besides resolution and options.

    ``solver_version`` is the adapter's optional ``backend_version`` attribute: the
    version of the library it drives, or None if it does not say.
    """
    return {
        "scenario": scenario_digest(scenario),
        "code": code_digest(solver),
        "solver_version": getattr(solver, "backend_version", None),
    }
