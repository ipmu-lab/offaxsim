"""
AR coating index profile utilities.

This module provides:
- klop_index_profile(h, num, ns, n0, Gamma_m): discretized Klopfenstein index profile

Notes
-----
- This function returns a piecewise-constant refractive index profile for (num-1) layers.
- Units:
  - h: total thickness [m]
  - z_arr, z_space: [m]
  - n_ave: refractive index [-]
"""

import scipy.special as special
import scipy.integrate as integrate
import numpy as np


def klop_index_profile(h, num, ns, n0, Gamma_m):
    """
    Compute discretized Klopfenstein index profile.

    Parameters
    ----------
    h : float
        Total thickness of the AR coating [m].
    num : int
        Number of sampling points along z (returns num-1 layers).
    ns : float
        Substrate refractive index (end index).
    n0 : float
        Incident medium refractive index (start index), typically 1.0.
    Gamma_m : float
        Maximum in-band reflection coefficient (design parameter).

    Returns
    -------
    n_ave : ndarray, shape (num-1,)
        Layer refractive indices (averaged between adjacent sampling points).
    z_arr : ndarray, shape (num-1,)
        Layer center positions along z [m].
    z_space : ndarray, shape (num-1,)
        Layer thicknesses [m] (uniform in the current implementation).
    """
    z_arr = np.linspace(0, h, num)
    z_space = np.ones(len(z_arr)) * (h / (num - 1))

    n_arr = np.zeros(num)

    rho_0 = 0.5 * np.log(ns / n0)
    A = np.arccosh(rho_0 / Gamma_m)

    for i in range(num):
        x_i = 2.0 * z_arr[i] / h - 1.0
        phi_int = integrate.quad(
            lambda y: special.i1(A * np.sqrt(1.0 - y**2)) / (A * np.sqrt(1.0 - y**2)),
            0,
            x_i,
        )
        n_arr[i] = np.sqrt(n0 * ns) * np.exp(Gamma_m * A**2 * phi_int[0])

    n_ave = np.zeros(num - 1)
    for i in range(num - 1):
        n_ave[i] = 0.5 * (n_arr[i + 1] + n_arr[i])

    z_space = z_space[:-1]
    z_arr = z_arr[:-1] + z_space[0] / 2.0

    return n_ave, z_arr, z_space
