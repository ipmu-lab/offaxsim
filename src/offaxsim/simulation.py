"""
Simulation utilities.

This module provides:
- SingleLayerSimulation(theta_deg)
  - run(...): returns (J, M)
- MultiLayerSimulation(theta_deg)
  - run(...): returns (J, M)

Notes
-----
- J shape: (Nrho, Nf, 2, 2)
- M shape: (Nrho, Nf, 4, 4)
- Frequency is in Hz. Thickness is in meters.
- Mueller conversion uses jones_mueller.py (kept separate from TMM).
"""

import numpy as np
from tqdm import tqdm

from .geometry import Geometry
from .material import UniaxialMaterial
from .tmm import TMM
from .jones_mueller import jones_to_mueller_from_elements


class SingleLayerSimulation:
    """
    Single-layer simulator.

    Parameters
    ----------
    theta_deg : float
        Incidence angle [deg].
    """

    def __init__(self, theta_deg):
        self.theta_deg = float(theta_deg)
        self.geom = Geometry(theta_deg)
        self.mat = UniaxialMaterial(theta_deg)
        self.tmm = TMM(theta_deg)

        self.A0, self.A0_inv = self.tmm.Air_layer_0()
        self.Af, self.Af_inv = self.tmm.Air_layer_f()

    def run(self, freq, rho_arr, no, ne, thickness_m, psi_deg, t_axis="y", return_mueller=True, show_progress=True):
        """
        Run a single-layer sweep over rho.

        Parameters
        ----------
        freq : ndarray, shape (Nf,)
            Frequency [Hz].
        rho_arr : ndarray, shape (Nrho,)
            rho angles [deg].
        no, ne : complex
            Refractive indices.
        thickness_m : float
            Layer thickness [m].
        psi_deg : float
            psi angle [deg].
        t_axis : {"x","y"}, default "y"
        return_mueller : bool, default True
        show_progress : bool, default True

        Returns
        -------
        J : ndarray, complex, shape (Nrho, Nf, 2, 2)
        M : ndarray, float, shape (Nrho, Nf, 4, 4) or None
        """
        freq = np.asarray(freq, dtype=float)
        rho_arr = np.asarray(rho_arr, dtype=float)

        Nrho = len(rho_arr)
        Nf = len(freq)

        J = np.zeros((Nrho, Nf, 2, 2), dtype=np.complex128)
        M = np.zeros((Nrho, Nf, 4, 4), dtype=float) if return_mueller else None

        it = enumerate(rho_arr)
        if show_progress:
            it = tqdm(list(it))

        for i, rho_deg in it:
            xi = self.geom.xi_vec(psi_deg, rho_deg)

            ep_rot, ep_rot_inv = self.mat.make_epsilon_matrix_inc_rot(no, ne, rho_deg, psi_deg)

            n1 = self.mat.solve_extra_index(no, ne, xi, forward=True,  t_axis=t_axis)
            n3 = self.mat.solve_extra_index(no, ne, xi, forward=False, t_axis=t_axis)
            nj = np.array([n1, no, n3, no], dtype=np.complex128)

            thetaj = self.tmm.output_theta(nj)
            kj = self.tmm.output_kj(thetaj, self.geom)
            qj = self.tmm.output_q(nj, thetaj)

            Dj = self.tmm.output_D(kj, xi)
            Hj = self.tmm.output_H(kj, Dj)

            A, A_inv = self.tmm.output_A(ep_rot_inv, Dj, Hj, nj)
            phase = self.tmm.output_phase(freq, thickness_m, qj)
            T = self.tmm.output_T(A, phase, A_inv)

            tpp, tps, tsp, tss = self.tmm.output_jones(T, self.A0_inv, self.Af)

            J[i, :, 0, 0] = tpp
            J[i, :, 0, 1] = tps
            J[i, :, 1, 0] = tsp
            J[i, :, 1, 1] = tss

            if return_mueller:
                M[i, :, :, :] = jones_to_mueller_from_elements(tpp, tps, tsp, tss)

        return J, M


class MultiLayerSimulation:
    """
    Multi-layer simulator.

    Parameters
    ----------
    theta_deg : float
        Incidence angle [deg].
    """

    def __init__(self, theta_deg):
        self.theta_deg = float(theta_deg)
        self.geom = Geometry(theta_deg)
        self.mat = UniaxialMaterial(theta_deg)
        self.tmm = TMM(theta_deg)

        self.A0, self.A0_inv = self.tmm.Air_layer_0()
        self.Af, self.Af_inv = self.tmm.Air_layer_f()

    def run(self, freq, rho_arr, stack, t_axis="y", return_mueller=True, show_progress=True):
        """
        Run a multilayer sweep over rho for a given LayerStack.

        Parameters
        ----------
        freq : ndarray, shape (Nf,)
            Frequency [Hz].
        rho_arr : ndarray, shape (Nrho,)
            rho angles [deg].
        stack : LayerStack
            Multilayer definition.
        t_axis : {"x","y"}, default "y"
        return_mueller : bool, default True
        show_progress : bool, default True

        Returns
        -------
        J : ndarray, complex, shape (Nrho, Nf, 2, 2)
        M : ndarray, float, shape (Nrho, Nf, 4, 4) or None
        """
        freq = np.asarray(freq, dtype=float)
        rho_arr = np.asarray(rho_arr, dtype=float)

        Nrho = len(rho_arr)
        Nf = len(freq)

        J = np.zeros((Nrho, Nf, 2, 2), dtype=np.complex128)
        M = np.zeros((Nrho, Nf, 4, 4), dtype=float) if return_mueller else None

        it = enumerate(rho_arr)
        if show_progress:
            it = tqdm(list(it))

        for i, rho_deg in it:
            # Ttot shape (4,4,Nf)
            Ttot = np.eye(4, dtype=np.complex128)[:, :, None] * np.ones((1, 1, Nf), dtype=np.complex128)

            for layeri in range(stack.Nlayer):
                rho_degi = rho_deg + stack.opt_axis_arr[layeri]
                psi_degi = stack.psi_arr[layeri]

                xi = self.geom.xi_vec(psi_degi, rho_degi)

                ep_rot, ep_rot_inv = self.mat.make_epsilon_matrix_inc_rot(
                    stack.no_arr[layeri],
                    stack.ne_arr[layeri],
                    rho_degi,
                    psi_degi,
                )

                n1 = self.mat.solve_extra_index(stack.no_arr[layeri], stack.ne_arr[layeri], xi, forward=True,  t_axis=t_axis)
                n3 = self.mat.solve_extra_index(stack.no_arr[layeri], stack.ne_arr[layeri], xi, forward=False, t_axis=t_axis)
                nj = np.array([n1, stack.no_arr[layeri], n3, stack.no_arr[layeri]], dtype=np.complex128)

                thetaj = self.tmm.output_theta(nj)
                kj = self.tmm.output_kj(thetaj, self.geom)
                qj = self.tmm.output_q(nj, thetaj)

                Dj = self.tmm.output_D(kj, xi)
                Hj = self.tmm.output_H(kj, Dj)

                A, A_inv = self.tmm.output_A(ep_rot_inv, Dj, Hj, nj)
                phase = self.tmm.output_phase(freq, stack.d_arr[layeri], qj)
                Ti = self.tmm.output_T(A, phase, A_inv)

                # Ttot = Ti @ Ttot (for each freq)
                Ttot = np.einsum("ijf,jkf->ikf", Ti, Ttot)

            tpp, tps, tsp, tss = self.tmm.output_jones(Ttot, self.A0_inv, self.Af)

            J[i, :, 0, 0] = tpp
            J[i, :, 0, 1] = tps
            J[i, :, 1, 0] = tsp
            J[i, :, 1, 1] = tss

            if return_mueller:
                M[i, :, :, :] = jones_to_mueller_from_elements(tpp, tps, tsp, tss)

        return J, M
