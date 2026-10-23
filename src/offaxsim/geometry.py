"""
Geometry utilities for the TMM / anisotropic propagation code.

This module provides:
- Geometry(theta_deg): store incidence angle and generate direction vectors
  - xi_vec(psi_deg, rho_deg): optic-axis unit vector in the lab frame
  - k_dir(theta_rad=None, forward=True): wavevector unit direction in the lab frame

Notes
-----
- Angle conventions (as used in the current codebase):
  - theta: incidence angle used in k_dir; k-hat = [0, sin(theta), +/-cos(theta)]
  - psi, rho: angles defining the optic axis xi (see xi_vec)
- xi_vec returns a real-valued unit vector.
- k_dir returns a complex-valued vector to allow complex theta_rad (evanescent/lossy cases).
"""

import numpy as np


class Geometry:
    """
    Geometry helper for incidence angle and direction vectors.

    Parameters
    ----------
    theta_deg : float
        Incidence angle in degrees. Stored internally in both degrees and radians.
    """

    def __init__(self, theta_deg):
        self.theta_deg = float(theta_deg)
        self.theta_rad = np.deg2rad(self.theta_deg)
        self.sin_theta = np.sin(self.theta_rad)
        self.cos_theta = np.cos(self.theta_rad)

    def xi_vec(self, psi_deg, rho_deg):
        """
        Optic-axis unit vector xi in the lab frame.

        Parameters
        ----------
        psi_deg : float
            Tilt angle (degrees) used in the current convention.
        rho_deg : float
            Rotation angle (degrees) around the z-axis in the current convention.

        Returns
        -------
        xi : ndarray, shape (3,)
            Real unit vector:
              xi = [-cos(psi)*sin(rho), cos(psi)*cos(rho), sin(psi)]
        """
        psi = np.deg2rad(psi_deg)
        rho = np.deg2rad(rho_deg)
        return np.array(
            [
                -np.cos(psi) * np.sin(rho),
                 np.cos(psi) * np.cos(rho),
                 np.sin(psi),
            ],
            dtype=float,
        )

    def k_dir(self, theta_rad=None, forward=True):
        """
        Unit direction vector of k in the current convention.

        Parameters
        ----------
        theta_rad : float or complex, optional
            If given, overrides the stored incidence angle. Complex values are allowed
            for evanescent / lossy cases.
        forward : bool, default True
            If True, kz = +cos(theta). If False, kz = -cos(theta).

        Returns
        -------
        k_hat : ndarray, complex, shape (3,)
            Unit direction vector:
              k_hat = [0, sin(theta), +/-cos(theta)]
        """
        if theta_rad is None:
            sin = self.sin_theta
            cos = self.cos_theta
        else:
            sin = np.sin(theta_rad)
            cos = np.cos(theta_rad)

        kz = cos if forward else -cos
        return np.array([0.0 + 0.0j, sin, kz], dtype=np.complex128)
