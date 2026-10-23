"""Public simulation checks and an independent isotropic-slab reference."""

import numpy as np
import pytest

from offaxsim import (
    SingleLayerSimulation, MultiLayerSimulation, build_single_layer,
    build_hwp_stack, build_klopfenstein_arc, wrap_with_symmetric_arc,
)

FREQ = np.array([60.0, 90.0, 120.0]) * 1e9
RHO = np.array([0.0, 45.0, 90.0])
PARAMS = dict(no=3.059 - 0.000137655j, ne=3.397 - 0.00023779j,
              thickness_m=5e-3, psi_deg=30.0)


def test_single_and_multilayer_agree():
    J1, M1 = SingleLayerSimulation(10.0).run(
        FREQ, RHO, **PARAMS, show_progress=False)
    J2, M2 = MultiLayerSimulation(10.0).run(
        FREQ, RHO, build_single_layer(**PARAMS), show_progress=False)
    assert J1.shape == J2.shape == (3, 3, 2, 2)
    assert M1.shape == M2.shape == (3, 3, 4, 4)
    np.testing.assert_allclose(J1, J2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(M1, M2, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 10.0, 35.0])
def test_zero_thickness_is_identity(theta):
    J, M = SingleLayerSimulation(theta).run(
        FREQ, RHO, **dict(PARAMS, thickness_m=0.0), show_progress=False)
    np.testing.assert_allclose(J, np.broadcast_to(np.eye(2), J.shape), atol=1e-12)
    np.testing.assert_allclose(M, np.broadcast_to(np.eye(4), M.shape), atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 20.0, 45.0])
def test_isotropic_slab_matches_fresnel_series(theta):
    """Sum internal round trips independently for a lossless air/slab/air system.

    For each polarization, use optical admittance Y and interface reflection
    r = (Y_air - Y_slab)/(Y_air + Y_slab). With P = exp(-i*delta),
    t = (1-r**2)*P / (1-r**2*P**2). This reference does not call TMM helpers.
    """
    n, d = 1.7, 1.3e-3
    angle = np.deg2rad(theta)
    cos_air = np.cos(angle)
    cos_slab = np.sqrt(1.0 - (np.sin(angle) / n)**2)
    phase = np.exp(-1j * 2.0 * np.pi * FREQ / 299792458.0 * n * d * cos_slab)
    expected = np.zeros((len(FREQ), 2, 2), dtype=complex)
    # p: Y=n/cos(theta); s: Y=n*cos(theta).
    for i, (y_air, y_slab) in enumerate([
        (1.0 / cos_air, n / cos_slab), (cos_air, n * cos_slab),
    ]):
        r = (y_air - y_slab) / (y_air + y_slab)
        expected[:, i, i] = (1.0 - r**2) * phase / (1.0 - r**2 * phase**2)
    J, _ = SingleLayerSimulation(theta).run(
        FREQ, RHO, no=n, ne=n, thickness_m=d, psi_deg=0.0,
        show_progress=False)
    np.testing.assert_allclose(
        J, np.broadcast_to(expected, J.shape), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("multilayer", [False, True])
def test_mueller_can_be_disabled_and_singleton_axes_are_preserved(multilayer):
    args = dict(return_mueller=False, show_progress=False)
    if multilayer:
        J, M = MultiLayerSimulation(10).run(
            [90e9], [45.0], build_single_layer(**PARAMS), **args)
    else:
        J, M = SingleLayerSimulation(10).run([90e9], [45.0], **PARAMS, **args)
    assert J.shape == (1, 1, 2, 2)
    assert np.isfinite(J).all()
    assert M is None


def test_arc_stack_produces_finite_transmission():
    core = build_hwp_stack(PARAMS['no'], PARAMS['ne'], np.full(5, 5e-3),
                           10.0, [88.65, 61.68, 0.0, 61.68, 88.65])
    arc = build_klopfenstein_arc(2e-3, 11, 3.397, 1.0, 0.03878242227997966)
    stack = wrap_with_symmetric_arc(core, arc)
    assert stack.Nlayer == 25
    np.testing.assert_allclose(np.sum(stack.d_arr), 29e-3, atol=1e-15)
    J, M = MultiLayerSimulation(10).run(FREQ, RHO, stack, show_progress=False)
    assert np.isfinite(J).all() and np.isfinite(M).all()
    np.testing.assert_allclose(M[..., 0, 0],
                               0.5 * np.sum(np.abs(J)**2, axis=(-2, -1)),
                               rtol=1e-10, atol=1e-12)
