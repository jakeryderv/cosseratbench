"""Each built-in solver must reproduce the analytical references to within the
error its physical model and default resolution explain."""

import pytest

from cosseratbench import registry, run

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
