"""
Jones -> Mueller conversion utilities.

This module provides:
- jones_to_mueller(J): convert Jones matrix/matrices to Mueller matrix/matrices
- jones_to_mueller_from_elements(tpp, tps, tsp, tss): helper to build J then convert

Stokes convention used here (coherency matrix rho_ab = E_a * conjugate(E_b)):
  S0 = rho_xx + rho_yy
  S1 = rho_xx - rho_yy
  S2 = rho_xy + rho_yx
  S3 = i (rho_xy - rho_yx)

The labels x, y denote the two Jones components; simulation outputs use (p, s).

Notes
-----
- The returned Mueller matrix should be real for physical Jones matrices.
  This implementation drops the imaginary part at the end.
"""

import numpy as np


def jones_to_mueller(J):
    """
    Convert Jones matrix to Mueller matrix using:
        M_ij = 1/2 * Tr( sigma_i * J * sigma_j * J^† )

    Parameters
    ----------
    J : ndarray, complex, shape (..., 2, 2)
        Jones matrices for many frequencies/angles at once.

    Returns
    -------
    M : ndarray, float, shape (..., 4, 4)
        Mueller matrices. Imag part is dropped (should be ~0).
    """
    J = np.asarray(J, dtype=np.complex128)
    Jdag = np.conjugate(np.swapaxes(J, -1, -2))

    # Pauli-like basis for Stokes defined in the module docstring
    sigma = np.array([
        [[1, 0], [0, 1]],              # S0
        [[1, 0], [0,-1]],              # S1
        [[0, 1], [1, 0]],              # S2
        [[0,-1j],[1j,0]],              # S3
    ], dtype=np.complex128)            # shape (4,2,2)

    # T_j = J * sigma_j * J^†
    tmp = np.einsum('...ab,jbc->...jac', J, sigma)        # (...,4,2,2)
    T   = np.einsum('...jac,...cd->...jad', tmp, Jdag)    # (...,4,2,2)

    # M_ij = 1/2 Tr( sigma_i * T_j )
    M = 0.5 * np.einsum('iab,...jba->...ij', sigma, T)    # (...,4,4)

    return np.real(M)


def jones_to_mueller_from_elements(tpp, tps, tsp, tss):
    """
    Build Jones J from elements and convert to Mueller.
    Arrays broadcast naturally (e.g. (Nf,) or (Nrho, Nf)).

    Parameters
    ----------
    tpp, tps, tsp, tss : ndarray, complex, shape (...)
        Jones matrix elements.

    Returns
    -------
    M : ndarray, float, shape (..., 4, 4)
        Mueller matrices.
    """
    tpp = np.asarray(tpp, dtype=np.complex128)
    tps = np.asarray(tps, dtype=np.complex128)
    tsp = np.asarray(tsp, dtype=np.complex128)
    tss = np.asarray(tss, dtype=np.complex128)

    J = np.empty(tpp.shape + (2, 2), dtype=np.complex128)
    J[..., 0, 0] = tpp
    J[..., 0, 1] = tps
    J[..., 1, 0] = tsp
    J[..., 1, 1] = tss

    return jones_to_mueller(J)
