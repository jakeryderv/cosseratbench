"""Each built-in solver must reproduce the analytical references to within the
error its physical model and default resolution explain."""

import dataclasses

import numpy as np
import pytest

from cosseratbench import (
    End,
    EndCondition,
    Material,
    Motion,
    PointLoad,
    Rod,
    Scenario,
    registry,
    run,
)
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


def eased(total, seconds=1.0, samples=41):
    t = np.linspace(0.0, seconds, samples)
    share = 0.5 - 0.5 * np.cos(np.pi * t / seconds)
    return tuple(t), tuple(tuple(total * f) for f in share)


@pytest.mark.parametrize(
    "displacement, rotation, tip",
    [
        ((0, 0, 0), (0, np.pi / 2, 0), (0, 0, -1)),  # turned a quarter about y
        ((0, 0.3, 0), (0, 0, 0), (1, 0.3, 0)),  # carried sideways
    ],
)
def test_a_driven_clamp_carries_the_rod_with_it(solver, displacement, rotation, tip):
    times, moved = eased(np.array(displacement, dtype=float))
    _, turned = eased(np.array(rotation, dtype=float))
    rod = Rod(
        centerline=((0, 0, 0), (1, 0, 0)),
        radius=0.02,
        material=Material(1e7, 1e7 / 3.0, 1000.0),
        start=EndCondition.CLAMPED,
        start_motion=Motion(times, moved, turned),
    )
    # Settling damping drags on the moving rod, so allow it time to catch up.
    scenario = Scenario(rods=(rod,), duration=6.0, quasi_static=True)
    final = solver.run(scenario, n_elements=20, n_frames=11).positions[0][-1]
    np.testing.assert_allclose(final[-1], tip, atol=1e-6)


def test_clamping_either_end_bends_a_cantilever_alike(solver):
    cantilever = registry.load_experiment("cantilever").scenario
    base = cantilever.rods[0]
    mirrored = Rod(
        centerline=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        radius=base.radius,
        material=base.material,
        end=EndCondition.CLAMPED,
        loads=(PointLoad(force=base.loads[0].force, at=End.START),),
    )
    flipped = dataclasses.replace(cantilever, rods=(mirrored,))
    usual = solver.run(cantilever, n_elements=20, n_frames=11).positions[0][-1, -1, 2]
    other = solver.run(flipped, n_elements=20, n_frames=11).positions[0][-1, 0, 2]
    # MuJoCo holds a far end with a slightly compliant weld: 0.2% more deflection.
    assert other == pytest.approx(usual, rel=5e-3)
