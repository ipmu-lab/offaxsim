"""
4x4 transfer-matrix (TMM) utilities for the current formulation.

This module provides:
- TMM(theta_deg): 4x4 transfer-matrix pipeline
  - Air_layer_0(), Air_layer_f(): boundary matrices for air
  - output_theta(nj)
  - output_kj(thetaj, geom)
  - output_q(nj, thetaj)
  - output_D(kj, xi)
  - output_H(kj, D)
  - output_A(ep_rot_inv, Dj, Hj, nj)
  - output_phase(freq, thickness_m, qj)
  - output_T(A, phase, A_inv)
  - output_jones(T, A0_inv, Af): extract (tpp, tps, tsp, tss)

Notes
-----
- This class implements your current 4x4 formulation and naming conventions.
- theta_deg is stored in the instance and used by output_theta() and air boundary blocks.
- Frequency is in Hz. Thickness is in meters.
- Constants pi and c are expected to be imported from const.py.
"""

import numpy as np
from .const import pi, c


class TMM:
    """
    4x4 transfer-matrix machinery for the current formulation.

    Parameters
    ----------
    theta_deg : float
        Incidence angle in degrees. Stored and reused internally.
    """

    def __init__(self, theta_deg):
        self.theta_deg = float(theta_deg)

    # ---- air layers (boundary matrices) ----
    def air_blocks(self):
        """
        Internal helper: build gm and H blocks for the air boundary.

        Returns
        -------
        gm : ndarray, float, shape (4, 3)
        H  : ndarray, float, shape (4, 3)
        """
        th = np.deg2rad(self.theta_deg)
        cos = np.cos(th)
        sin = np.sin(th)

        gm = np.array(
            [
                [0.0, cos, -sin],
                [1.0, 0.0, 0.0],
                [0.0, cos,  sin],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        H = np.array(
            [
                [-1.0, 0.0, 0.0],
                [ 0.0, cos, -sin],
                [ 1.0, 0.0, 0.0],
                [ 0.0,-cos, -sin],
            ],
            dtype=float,
        )

        return gm, H

    def build_A_from_blocks(self, gm, H):
        """
        Internal helper: build A from gm and H blocks.

        Parameters
        ----------
        gm, H : ndarray, shape (4, 3)

        Returns
        -------
        A : ndarray, float, shape (4, 4)
        """
        return np.vstack([gm[:, 0], gm[:, 1], H[:, 0], H[:, 1]])

    def Air_layer_0(self):
        """
        Boundary matrix at the incident (air) side.

        Returns
        -------
        A0 : ndarray, float, shape (4, 4)
        A0_inv : ndarray, float, shape (4, 4)
        """
        gm, H = self.air_blocks()
        A_0 = self.build_A_from_blocks(gm, H)
        return A_0, np.linalg.inv(A_0)

    def Air_layer_f(self):
        """
        Boundary matrix at the exit (air) side.

        Returns
        -------
        Af : ndarray, float, shape (4, 4)
        Af_inv : ndarray, float, shape (4, 4)
        """
        gm, H = self.air_blocks()
        A_f = self.build_A_from_blocks(gm, H)
        return A_f, np.linalg.inv(A_f)

    # ---- per-layer pipeline pieces ----
    def output_theta(self, nj):
        """
        Compute propagation angles inside the layer for each mode.

        Parameters
        ----------
        nj : ndarray, complex, shape (4,)
            Effective refractive indices for the four modes.

        Returns
        -------
        thetaj : ndarray, complex, shape (4,)
            Propagation angles inside the layer (complex allowed).
        """
        return np.arcsin(np.sin(np.deg2rad(self.theta_deg)) / nj)

    def output_kj(self, thetaj, geom):
        """
        Compute k direction unit vectors for the four modes.

        Parameters
        ----------
        thetaj : ndarray, complex, shape (4,)
        geom : Geometry
            Geometry instance providing k_dir().

        Returns
        -------
        kj : ndarray, complex, shape (4, 3)
        """
        k1 = geom.k_dir(thetaj[0], forward=True)
        k2 = geom.k_dir(thetaj[1], forward=True)
        k3 = geom.k_dir(thetaj[2], forward=False)
        k4 = geom.k_dir(thetaj[3], forward=False)
        return np.array([k1, k2, k3, k4])

    def output_q(self, nj, thetaj):
        """
        Compute qj = sign * n_j * cos(theta_j).

        Parameters
        ----------
        nj : ndarray, complex, shape (4,)
        thetaj : ndarray, complex, shape (4,)

        Returns
        -------
        qj : ndarray, complex, shape (4,)
        """
        sign = np.array([1, 1, -1, -1], dtype=float)
        return sign * nj * np.cos(thetaj)

    def output_D(self, kj, xi):
        """
        Compute D vectors for the four modes.

        Parameters
        ----------
        kj : ndarray, complex, shape (4, 3)
        xi : ndarray, float, shape (3,)
            Optic axis vector (real).

        Returns
        -------
        D : ndarray, complex, shape (4, 3)
        """
        D2 = np.cross(kj[1], xi)
        D4 = np.cross(kj[3], xi)
        D1 = np.cross(kj[0], D2)
        D3 = np.cross(kj[2], D4)
        return np.array([D1, D2, D3, D4])

    def output_H(self, kj, D):
        """
        Compute H vectors for the four modes.

        Parameters
        ----------
        kj : ndarray, complex, shape (4, 3)
        D : ndarray, complex, shape (4, 3)

        Returns
        -------
        H : ndarray, complex, shape (4, 3)
        """
        H1 = np.cross(kj[0], D[0])
        H2 = np.cross(kj[1], D[1])
        H3 = np.cross(kj[2], D[2])
        H4 = np.cross(kj[3], D[3])
        return np.array([H1, H2, H3, H4])

    def output_A(self, ep_rot_inv, Dj, Hj, nj):
        """
        Build the 4x4 A matrix for the layer.

        Parameters
        ----------
        ep_rot_inv : ndarray, complex, shape (3, 3)
            Inverse rotated dielectric tensor.
        Dj : ndarray, complex, shape (4, 3)
        Hj : ndarray, complex, shape (4, 3)
        nj : ndarray, complex, shape (4,)

        Returns
        -------
        A : ndarray, complex, shape (4, 4)
        A_inv : ndarray, complex, shape (4, 4)
        """
        gamma1 = ep_rot_inv @ Dj[0]
        gamma2 = ep_rot_inv @ Dj[1]
        gamma3 = ep_rot_inv @ Dj[2]
        gamma4 = ep_rot_inv @ Dj[3]

        zeta1 = Hj[0] / nj[0]
        zeta2 = Hj[1] / nj[1]
        zeta3 = Hj[2] / nj[2]
        zeta4 = Hj[3] / nj[3]

        A = np.array(
            [
                [gamma1[0], gamma2[0], gamma3[0], gamma4[0]],
                [gamma1[1], gamma2[1], gamma3[1], gamma4[1]],
                [zeta1[0],  zeta2[0],  zeta3[0],  zeta4[0]],
                [zeta1[1],  zeta2[1],  zeta3[1],  zeta4[1]],
            ]
        )
        return A, np.linalg.inv(A)

    def output_phase(self, freq, thickness_m, qj):
        """
        Compute phase factors exp(-i k0 d qj) for each mode.

        Parameters
        ----------
        freq : ndarray, float, shape (Nf,)
            Frequency [Hz].
        thickness_m : float
            Layer thickness [m].
        qj : ndarray, complex, shape (4,)

        Returns
        -------
        phase : ndarray, complex, shape (4, Nf)
        """
        k0 = 2.0 * pi * freq / c
        return np.exp(-1j * (k0 * thickness_m)[None, :] * qj[:, None])

    def output_T(self, A, phase, A_inv):
        """
        Build the 4x4 transfer matrix for a single layer.

        Parameters
        ----------
        A : ndarray, complex, shape (4, 4)
        phase : ndarray, complex, shape (4, Nf)
        A_inv : ndarray, complex, shape (4, 4)

        Returns
        -------
        T : ndarray, complex, shape (4, 4, Nf)
        """
        AP = A[:, :, None] * phase[None, :, :]
        return np.einsum("ijf,jk->ikf", AP, A_inv)

    def output_jones(self, T, A0_inv, Af):
        """
        Extract Jones transmission coefficients from the total 4x4 matrix.

        Parameters
        ----------
        T : ndarray, complex, shape (4, 4, Nf)
            Total transfer matrix.
        A0_inv : ndarray, complex, shape (4, 4)
            Inverse boundary matrix at incidence side.
        Af : ndarray, complex, shape (4, 4)
            Boundary matrix at exit side.

        Returns
        -------
        tpp, tps, tsp, tss : ndarray, complex, shape (Nf,)
            Jones transmission elements.
        """
        RM = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        RM_inv = np.linalg.pinv(RM)

        tmp = np.einsum("ij,jkF->ikF", A0_inv, T)
        Mtot = np.einsum("ijF,jk->ikF", tmp, Af)

        tmp2 = np.einsum("ij,jkF->ikF", RM_inv, Mtot)
        GM = np.einsum("ijF,jk->ikF", tmp2, RM)

        den = (GM[1, 1, :] * GM[3, 3, :] - GM[1, 3, :] * GM[3, 1, :])

        tpp = GM[3, 3, :] / den
        tss = GM[1, 1, :] / den
        tps = GM[3, 1, :] / den
        tsp = GM[1, 3, :] / den
        return tpp, tps, tsp, tss
