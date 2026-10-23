"""
Layer stack utilities.

This module provides:
- LayerStack: container for multilayer parameters
- build_single_layer(...)
- build_hwp_stack(...)
- build_klopfenstein_arc(...)
- wrap_with_symmetric_arc(core_stack, arc_stack)

Notes
-----
- A stack is represented by 1D arrays of the same length:
  no_arr, ne_arr, d_arr, psi_arr, opt_axis_arr
- Units:
  - thickness (d_arr) is in meters
  - angles are in degrees
"""

import numpy as np
from .arc_models import klop_index_profile


class LayerStack:
    """
    Simple container for multilayer parameters.

    Attributes
    ----------
    no_arr, ne_arr : ndarray, complex, shape (Nlayer,)
    d_arr : ndarray, float, shape (Nlayer,)
    psi_arr : ndarray, float, shape (Nlayer,)
    opt_axis_arr : ndarray, float, shape (Nlayer,)
    """

    def __init__(self, no_arr, ne_arr, d_arr, psi_arr, opt_axis_arr):
        self.no_arr = np.asarray(no_arr, dtype=np.complex128)
        self.ne_arr = np.asarray(ne_arr, dtype=np.complex128)
        self.d_arr = np.asarray(d_arr, dtype=float)
        self.psi_arr = np.asarray(psi_arr, dtype=float)
        self.opt_axis_arr = np.asarray(opt_axis_arr, dtype=float)

        n = len(self.no_arr)
        if not (len(self.ne_arr) == n and len(self.d_arr) == n and len(self.psi_arr) == n and len(self.opt_axis_arr) == n):
            raise ValueError("All layer arrays must have the same length.")

    @property
    def Nlayer(self):
        return len(self.no_arr)

    def reversed(self):
        """
        Return a reversed copy of the stack.

        Notes
        -----
        This matches your current ARC symmetry usage:
        - no/ne reversed
        - thickness reversed
        - psi reversed
        - opt_axis reversed
        """
        return LayerStack(
            self.no_arr[::-1],
            self.ne_arr[::-1],
            self.d_arr[::-1],
            self.psi_arr[::-1],
            self.opt_axis_arr[::-1],
        )

    @staticmethod
    def concat(*stacks):
        """
        Concatenate multiple stacks in order.

        Parameters
        ----------
        stacks : LayerStack

        Returns
        -------
        LayerStack
        """
        no = np.concatenate([s.no_arr for s in stacks])
        ne = np.concatenate([s.ne_arr for s in stacks])
        d = np.concatenate([s.d_arr for s in stacks])
        psi = np.concatenate([s.psi_arr for s in stacks])
        opt = np.concatenate([s.opt_axis_arr for s in stacks])
        return LayerStack(no, ne, d, psi, opt)


def build_single_layer(no, ne, thickness_m, psi_deg, opt_axis_deg=0.0):
    """
    Build a single-layer stack.

    Returns
    -------
    LayerStack
    """
    no_arr = np.array([no], dtype=np.complex128)
    ne_arr = np.array([ne], dtype=np.complex128)
    d_arr = np.array([thickness_m], dtype=float)
    psi_arr = np.array([psi_deg], dtype=float)
    opt_axis_arr = np.array([opt_axis_deg], dtype=float)
    return LayerStack(no_arr, ne_arr, d_arr, psi_arr, opt_axis_arr)


def build_hwp_stack(no, ne, thickness_arr_m, psi_deg, opt_axis_arr_deg):
    """
    Build a multi-layer HWP stack.

    Parameters
    ----------
    no, ne : complex
        HWP indices (same for all HWP layers in this helper).
    thickness_arr_m : ndarray, shape (N,)
        Thickness of each HWP layer [m].
    psi_deg : float
        psi for all HWP layers [deg] (matches your current usage).
    opt_axis_arr_deg : ndarray, shape (N,)
        rho offset for each layer [deg] (your 'opt_axis_HWP').

    Returns
    -------
    LayerStack
    """
    thickness_arr_m = np.asarray(thickness_arr_m, dtype=float)
    opt_axis_arr_deg = np.asarray(opt_axis_arr_deg, dtype=float)

    n = len(thickness_arr_m)
    if len(opt_axis_arr_deg) != n:
        raise ValueError("thickness_arr_m and opt_axis_arr_deg must have the same length.")

    no_arr = np.ones(n, dtype=np.complex128) * no
    ne_arr = np.ones(n, dtype=np.complex128) * ne
    d_arr = thickness_arr_m
    psi_arr = np.ones(n, dtype=float) * psi_deg
    opt_axis_arr = opt_axis_arr_deg
    return LayerStack(no_arr, ne_arr, d_arr, psi_arr, opt_axis_arr)


def build_klopfenstein_arc(h_m, num, ns, n0, Gamma_m):
    """
    Build an isotropic ARC stack from a Klopfenstein index profile.

    Parameters
    ----------
    h_m : float
        Total ARC thickness [m].
    num : int
        Number of sampling points (returns num-1 layers).
    ns : float
        Substrate index.
    n0 : float
        Incident medium index.
    Gamma_m : float
        Design reflection coefficient.

    Returns
    -------
    LayerStack
        Isotropic layers: no=ne=n_ave, psi=0, opt_axis=0
    """
    n_ave, z_arr, z_space = klop_index_profile(h_m, num, ns, n0, Gamma_m)
    no_arr = np.asarray(n_ave, dtype=np.complex128)
    ne_arr = np.asarray(n_ave, dtype=np.complex128)
    d_arr = np.asarray(z_space, dtype=float)
    psi_arr = np.zeros(len(d_arr), dtype=float)
    opt_axis_arr = np.zeros(len(d_arr), dtype=float)
    return LayerStack(no_arr, ne_arr, d_arr, psi_arr, opt_axis_arr)


def wrap_with_symmetric_arc(core_stack, arc_stack):
    """
    Return ARC + core + reversed(ARC).

    Parameters
    ----------
    core_stack : LayerStack
    arc_stack : LayerStack

    Returns
    -------
    LayerStack
    """
    return LayerStack.concat(arc_stack, core_stack, arc_stack.reversed())
