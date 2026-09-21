"""Each built-in solver must reproduce the analytical references to within the
error its physical model and default resolution explain."""

import dataclasses

import numpy as np
import pytest

from cosseratbench import registry, run
from cosseratbench.experiments.pendulum import static_strain

pytestmark = pytest.mark.slow

# MuJoCo's cable cannot stretch, and stretch deepens this catenary's sag by ~1.2%.
CATENARY_SAG_TOLERANCE = {"pyelastica": 2e-3, "mujoco": 2e-2}


@pytest.fixture(params=["pyelastica", "mujoco"])
def solver(request):
    pytest.importorskip({"pyelastica": "elastica"}.get(request.param, request.param))
    return registry.load_solver(request.param)


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
