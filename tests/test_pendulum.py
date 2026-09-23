import math

import numpy as np
import pytest

from cosseratbench import Material, Trajectory
from cosseratbench.experiments.pendulum import (
    AMPLITUDE,
    GRAVITY,
    LENGTH,
    amplitude_change,
    first_mode,
    measured_period,
    pendulum,
    period_error,
    reference_period,
    static_strain,
)

MATERIAL = pendulum.scenario.rods[0].material
BERNOULLI = 2.404825557695773 / 2.0 * math.sqrt(GRAVITY / LENGTH)  # first zero of J0, halved


def test_reference_is_bernoullis_chain_without_bending_or_stretch():
    # E r^2 sets bending against weight and 1/E sets stretch, so both vanish together.
    omega, displacement, _ = first_mode(LENGTH, 1e-10, Material(1e15, 1e15 / 3.0, 1000.0), GRAVITY)
    assert omega == pytest.approx(BERNOULLI, rel=1e-8)
    heights = np.linspace(0.0, LENGTH, len(displacement))
    series = sum(  # J0(j01 sqrt(z / L)), the chain's mode shape
        (-1) ** k
        * (2.404825557695773 * np.sqrt(heights / LENGTH) / 2) ** (2 * k)
        / math.factorial(k) ** 2
        for k in range(40)
    )
    np.testing.assert_allclose(displacement, series, atol=1e-5)


def test_bending_speeds_the_swing_and_stretch_slows_it():
    thin = 1e-10
    stiff = Material(1e15, 1e15 / 3.0, 1000.0)
    stretchy = Material(1e7, 1e7 / 3.0, 1000.0)
    bending_only = first_mode(LENGTH, 0.002 * math.sqrt(1e7 / 1e15), stiff, GRAVITY)[0]
    stretch_only = first_mode(LENGTH, thin, stretchy, GRAVITY)[0]
    assert bending_only > BERNOULLI
    # Stretch lengthens the cable by its mean strain, which slows it by about half that.
    mean_strain = 1000.0 * GRAVITY * LENGTH / (2.0 * 1e7)
    assert (BERNOULLI - stretch_only) / BERNOULLI == pytest.approx(mean_strain / 2.0, rel=0.2)


def test_reference_has_converged():
    coarse = first_mode(LENGTH, 0.002, MATERIAL, GRAVITY, n_elements=100)[0]
    fine = first_mode(LENGTH, 0.002, MATERIAL, GRAVITY, n_elements=400)[0]
    assert fine == pytest.approx(coarse, rel=1e-8)


def test_cable_starts_in_its_first_mode_stretched_as_it_hangs():
    rod = pendulum.scenario.rods[0]
    assert rod.length == pytest.approx(LENGTH)
    nodes = rod.nodes(50)
    assert nodes[-1, 0] == pytest.approx(AMPLITUDE)
    segments = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    rest = LENGTH / 50
    middles = (np.arange(50) + 0.5) * rest
    expected = static_strain(middles, LENGTH, MATERIAL, GRAVITY)
    np.testing.assert_allclose(segments / rest - 1.0, expected, atol=2e-6)


def swing(period, amplitude=AMPLITUDE, decay=0.0, wobble=0.0, frames=101):
    """A trajectory whose free end swings sideways as given; wobble adds fast vibration."""
    t = np.linspace(0.0, pendulum.scenario.duration, frames)
    x = amplitude * np.exp(-decay * t) * np.cos(2 * np.pi * t / period)
    x += wobble * np.sin(2 * np.pi * 12.3 * t)
    positions = np.zeros((frames, 2, 3))
    positions[:, 1, 0] = x
    return Trajectory(t, (positions,), (np.zeros((frames, 1, 3)),))


def test_metrics_read_period_and_energy_loss_from_the_swing():
    scenario = pendulum.scenario
    period = reference_period(scenario)
    assert period_error(scenario, swing(period)) < 1e-6
    assert period_error(scenario, swing(1.01 * period)) == pytest.approx(0.01, rel=1e-3)
    assert amplitude_change(scenario, swing(period)) == pytest.approx(0.0, abs=1e-6)
    decay = -math.log(0.9) / (scenario.duration - period)  # 10% smaller one period from the end
    assert amplitude_change(scenario, swing(period, decay=decay)) == pytest.approx(-0.1, abs=0.02)


def test_period_is_robust_to_fast_vibration_riding_on_the_swing():
    # Crossing times broke on this: 1 mm of 12 Hz on a 1 cm swing gives extra crossings.
    scenario = pendulum.scenario
    period = reference_period(scenario)
    shaky = swing(period, amplitude=0.01, wobble=0.001)
    assert measured_period(scenario, shaky) == pytest.approx(period, rel=1e-3)
