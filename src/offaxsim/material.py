"""
Uniaxial material model utilities.

This module provides:
- UniaxialMaterial(theta_deg): uniaxial permittivity rotation and extraordinary index solver
  - make_epsilon_matrix(no, ne)
  - rot_rho(rho_deg), rot_psi(psi_deg)
  - make_epsilon_matrix_inc_rot(no, ne, rho_deg, psi_deg)
  - solve_extra_index(no, ne, xi, forward=True, t_axis="y")

Notes
-----
- The code keeps the current project conventions for rotations and the extraordinary-index solver.
- no, ne are refractive indices (complex allowed). Permittivity is eps = n^2.
- xi (optic axis unit vector) must be real; this solver enforces Im(xi) ~ 0.
"""

import numpy as np


class UniaxialMaterial:
    """
    Uniaxial material helper.

    Parameters
    ----------
    theta_deg : float
        Incidence angle in degrees, used internally by solve_extra_index.
    """

    def __init__(self, theta_deg):
        self.theta_deg = float(theta_deg)

    def make_epsilon_matrix(self, no, ne):
        """
        Construct the unrotated dielectric tensor for a uniaxial material.

        Parameters
        ----------
        no, ne : complex
            Ordinary and extraordinary refractive indices.

        Returns
        -------
        eps : ndarray, complex, shape (3, 3)
            Unrotated dielectric tensor:
              diag([no^2, ne^2, no^2])
        """
        return np.array(
            [
                [no**2, 0,     0],
                [0,     ne**2, 0],
                [0,     0,     no**2],
            ],
            dtype=complex,
        )

    def rot_rho(self, rho_deg):
        """
        Rotation matrix about the z-axis by rho_deg.

        Parameters
        ----------
        rho_deg : float
            Rotation angle in degrees.

        Returns
        -------
        R : ndarray, float, shape (3, 3)
            Rotation matrix.
        """
        rho = np.deg2rad(rho_deg)
        cos = np.cos(rho)
        sin = np.sin(rho)
        return np.array(
            [
                [cos, -sin, 0],
                [sin,  cos, 0],
                [0,    0,   1],
            ],
            dtype=float,
        )

    def rot_psi(self, psi_deg):
        """
        Rotation matrix about the x-axis by psi_deg.

        Parameters
        ----------
        psi_deg : float
            Rotation angle in degrees.

        Returns
        -------
        R : ndarray, float, shape (3, 3)
            Rotation matrix.
        """
        psi = np.deg2rad(psi_deg)
        cos = np.cos(psi)
        sin = np.sin(psi)
        return np.array(
            [
                [1,   0,    0],
                [0, cos, -sin],
                [0, sin,  cos],
            ],
            dtype=float,
        )

    def make_epsilon_matrix_inc_rot(self, no, ne, rho_deg, psi_deg):
        """
        Construct rotated dielectric tensor and its inverse.

        Parameters
        ----------
        no, ne : complex
            Ordinary and extraordinary refractive indices.
        rho_deg, psi_deg : float
            Rotation angles in degrees.

        Returns
        -------
        eps_rot : ndarray, complex, shape (3, 3)
            Rotated dielectric tensor.
        eps_rot_inv : ndarray, complex, shape (3, 3)
            Inverse of the rotated dielectric tensor.
        """
        eps = self.make_epsilon_matrix(no, ne)

        R = self.rot_rho(rho_deg) @ self.rot_psi(psi_deg)
        Rt = R.T

        eps_rot = R @ eps @ Rt

        eps_inv = np.linalg.inv(eps)
        eps_rot_inv = R @ eps_inv @ Rt

        return eps_rot, eps_rot_inv

    def solve_extra_index(self, no, ne, xi, forward=True, t_axis="y"):
        """
        Solve the effective extraordinary refractive index for the current conventions.

        Parameters
        ----------
        no, ne : complex
            Ordinary and extraordinary refractive indices.
        xi : array_like, shape (3,)
            Optic-axis unit vector in the lab frame. Must be real.
        forward : bool, default True
            Selects the sign branch for the internal quadratic solution.
        t_axis : {"x", "y"}, default "y"
            Tangential axis used by the solver (matches current project convention).

        Returns
        -------
        n_eff : complex
            Effective extraordinary refractive index.
        """
        xi = np.asarray(xi)
        if np.iscomplexobj(xi):
            imag_max = np.max(np.abs(np.imag(xi)))
            if imag_max > 1e-12:
                raise ValueError(f"xi must be real (max|Im|={imag_max}).")
            xi = np.real(xi)

        xi = xi.astype(np.float64)
        xi /= np.linalg.norm(xi)

        # rho is computed here only for debugging / historical reasons (kept as-is)
        rho = np.arctan2(-xi[0], xi[1])
        rho_deg = (np.rad2deg(rho) % 360.0)
        _ = rho_deg

        s = 1 if forward else -1

        eps_par = no**2
        eps_perp = ne**2
        alpha = eps_perp / eps_par - 1.0

        theta_i = np.deg2rad(self.theta_deg)
        p = np.sin(theta_i)

        a_t = xi[1] if t_axis == "y" else xi[0]
        a_z = xi[2]

        A = 1.0 + alpha * (a_z**2)
        B = 2.0 * alpha * p * a_t * a_z
        C = p**2 + alpha * (p**2) * (a_t**2) - eps_perp

        D = B * B - 4.0 * A * C
        w = (-B + s * np.sqrt(D)) / (2.0 * A)

        n_sq = p * p + w * w
        return np.sqrt(n_sq)
