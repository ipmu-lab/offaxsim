"""Check Mueller conversion against electric-field Stokes definitions."""

import numpy as np
import pytest
from offaxsim import jones_to_mueller, jones_to_mueller_from_elements


def stokes(E):
    p, s = E[..., 0], E[..., 1]
    cross = p * np.conjugate(s)
    return np.stack((np.abs(p)**2 + np.abs(s)**2,
                     np.abs(p)**2 - np.abs(s)**2,
                     2 * cross.real, -2 * cross.imag), axis=-1)


def test_identity_jones_gives_identity_mueller():
    np.testing.assert_allclose(jones_to_mueller(np.eye(2)), np.eye(4), atol=1e-12)


def test_quarter_wave_retarder_uses_documented_v_sign():
    # A +45-degree linear input becomes (1, i)/sqrt(2), which has V=+1.
    J = np.diag([1.0, 1.0j])
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 0, -1], [0, 0, 1, 0]])
    np.testing.assert_allclose(jones_to_mueller(J), expected, atol=1e-12)
    np.testing.assert_allclose(stokes(np.array([1, 1j]) / np.sqrt(2)),
                               [1, 0, 0, 1], atol=1e-12)


@pytest.mark.parametrize("E", [[1, 0], [0, 1], [1, 1], [1, 1j], [1, -1j]])
def test_batched_mueller_action_matches_jones_fields(E):
    E = np.asarray(E, dtype=complex)
    E /= np.linalg.norm(E)
    rng = np.random.default_rng(42)
    J = rng.normal(size=(2, 3, 2, 2)) + 1j * rng.normal(size=(2, 3, 2, 2))
    M = jones_to_mueller(J)
    assert M.shape == (2, 3, 4, 4)
    np.testing.assert_allclose(M @ stokes(E), stokes(J @ E),
                               rtol=1e-10, atol=1e-12)
    from_elements = jones_to_mueller_from_elements(
        J[..., 0, 0], J[..., 0, 1], J[..., 1, 0], J[..., 1, 1])
    np.testing.assert_allclose(from_elements, M, rtol=1e-10, atol=1e-12)
