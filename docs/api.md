# Python API reference

This reference describes `offaxsim` 0.1.0. All 12 names below are importable from `offaxsim`. For installation and worked examples, see the [README](../README.md) and [example notebook](../src/notebooks/example_usage.ipynb).

## API overview

| Purpose | Public names |
| --- | --- |
| Run simulations | `SingleLayerSimulation`, `MultiLayerSimulation` |
| Define layers | `LayerStack`, `build_single_layer`, `build_hwp_stack` |
| Add coatings | `build_klopfenstein_arc`, `wrap_with_symmetric_arc` |
| Convert polarization matrices | `jones_to_mueller`, `jones_to_mueller_from_elements` |
| Work with the low-level formulation | `Geometry`, `UniaxialMaterial`, `TMM` |

## Shared conventions

- Frequency is in Hz; thickness is in meters. Public angle inputs are in degrees except `Geometry.k_dir(theta_rad=...)` and the propagation-angle arrays used by `TMM`.
- The external media are air with index 1. The surface normal is z; the incidence plane is yz. The forward incident direction is `[0, sin(theta), cos(theta)]`.
- The optical axis is `[-cos(psi)*sin(rho), cos(psi)*cos(rho), sin(psi)]`. Here psi tilts out of the xy plane and rho rotates about z. Trigonometric formulas use radians.
- The forward air basis is `p = [0, cos(theta), -sin(theta)]`, `s = [1, 0, 0]`.
- A Jones matrix has rows for output and columns for input, ordered `(p, s)`: `[[tpp, tps], [tsp, tss]]`.
- With `C_ab = E_a * conjugate(E_b)`, Stokes components are `I=Cpp+Css`, `Q=Cpp-Css`, `U=Cps+Csp`, `V=i*(Cps-Csp)`. Thus `(1, i)/sqrt(2)` has V=+1.
- `Nf`, `Nrho`, and `Nlayer` denote the number of frequencies, rotations, and layers. Leading batch dimensions are written `...`.

## Simulations

### SingleLayerSimulation

```text
SingleLayerSimulation(theta_deg)
run(freq, rho_arr, no, ne, thickness_m, psi_deg,
    t_axis='y', return_mueller=True, show_progress=True)
```

Create the simulator with the external incidence angle `theta_deg`, then call its `run` method.

| Argument | Type / shape | Meaning |
| --- | --- | --- |
| `theta_deg` | Real scalar | External incidence angle in degrees. |
| `freq` | Array-like, `(Nf,)` | Frequencies in Hz. Use positive values. |
| `rho_arr` | Array-like, `(Nrho,)` | Optical-axis rotation angles in degrees. |
| `no`, `ne` | Real or complex scalars | Ordinary and principal extraordinary indices. Constant over the sweep. |
| `thickness_m` | Real scalar | Layer thickness in meters. |
| `psi_deg` | Real scalar | Optical-axis tilt in degrees. |
| `t_axis` | `'y'` or `'x'` | Tangential component used by the extraordinary-index solver; use `'y'` for the implemented geometry. |
| `return_mueller` | Boolean | If false, skip Mueller conversion and return `None` as the second result. |
| `show_progress` | Boolean | Display a tqdm progress bar over rotations. |

**Returns:** `(J, M)`, where `J` is complex128 with shape `(Nrho, Nf, 2, 2)` and `M` is a real array with shape `(Nrho, Nf, 4, 4)`, or `None`. Singleton sweep axes are retained. Reflection and absorption are not returned.

**Usage:**

```python
import numpy as np
from offaxsim import SingleLayerSimulation

freq = np.array([60.0, 90.0, 120.0]) * 1e9
rho = np.array([0.0, 45.0, 90.0])
J, M = SingleLayerSimulation(10.0).run(
    freq, rho, no=3.059 - 0.000137655j, ne=3.397 - 0.00023779j,
    thickness_m=5e-3, psi_deg=30.0, show_progress=False,
)
assert J.shape == (3, 3, 2, 2)
assert M.shape == (3, 3, 4, 4)
```

**Constraints:** `freq` and `rho_arr` must be one-dimensional, including for single-point runs. There is no comprehensive physical-input validation. Changing `t_axis` does not rotate the incidence plane. Do not modify `theta_deg` after construction: geometry, material, and boundary objects are initialized together; create a new simulator for a new incidence angle.

### MultiLayerSimulation

```text
MultiLayerSimulation(theta_deg)
run(freq, rho_arr, stack,
    t_axis='y', return_mueller=True, show_progress=True)
```

Uses the same constructor, common arguments, output shapes, and constraints as `SingleLayerSimulation`. `stack` must be a `LayerStack`. Each layer uses its own indices, thickness, and tilt, with effective rotation `rho_arr[i] + stack.opt_axis_arr[layer]`.

The implemented total matrix is `T_last @ ... @ T_first`. Verify physical orientation independently when interpreting asymmetric stacks.

```python
from offaxsim import MultiLayerSimulation, build_single_layer

stack = build_single_layer(3.059, 3.397, 5e-3, 30.0, opt_axis_deg=15.0)
J, M = MultiLayerSimulation(10.0).run(
    [90e9], [0.0, 45.0], stack, return_mueller=False, show_progress=False,
)
assert J.shape == (2, 1, 2, 2)
assert M is None
```

## Layers and coatings

### LayerStack

```text
LayerStack(no_arr, ne_arr, d_arr, psi_arr, opt_axis_arr)
```

Stores one-dimensional, equally sized arrays:

| Attribute / constructor argument | Stored dtype | Meaning |
| --- | --- | --- |
| `no_arr`, `ne_arr` | complex128 | Per-layer ordinary and extraordinary indices. |
| `d_arr` | float | Per-layer thickness in meters. |
| `psi_arr` | float | Per-layer optical-axis tilt in degrees. |
| `opt_axis_arr` | float | Per-layer rotation offset in degrees. |

`Nlayer` is a read-only property returning the number of layers. The arrays themselves remain mutable.

| Method | Return value / behavior |
| --- | --- |
| `reversed()` | A new `LayerStack` with reversed array order. Angles are not negated. Its arrays may share memory with the original; this is not a guaranteed deep copy. |
| `LayerStack.concat(*stacks)` | A new `LayerStack` containing the supplied stacks in order. Supply at least one stack. |

Mismatched array lengths raise `ValueError`. The constructor uses `np.asarray`, so compatible input arrays may be shared with the stack. Array dimensionality, index validity, and positive thickness are not comprehensively checked. Use `no_arr == ne_arr` for isotropic layers.

### build_single_layer

```text
build_single_layer(no, ne, thickness_m, psi_deg, opt_axis_deg=0.0)
```

Returns a one-layer `LayerStack`. `no` and `ne` are scalar indices, `thickness_m` is in meters, and `psi_deg` and `opt_axis_deg` are in degrees. The rotation offset is added to the sweep rotation by `MultiLayerSimulation`.

### build_hwp_stack

```text
build_hwp_stack(no, ne, thickness_arr_m, psi_deg, opt_axis_arr_deg)
```

Returns a `LayerStack` using the same scalar `no`, `ne`, and `psi_deg` for every layer. `thickness_arr_m` and `opt_axis_arr_deg` must be one-dimensional arrays of equal length: thicknesses in meters and rotation offsets in degrees. Unequal lengths raise `ValueError`.

This helper assembles the supplied layers; it does not calculate half-wave thicknesses or optimize a design.

### build_klopfenstein_arc

```text
build_klopfenstein_arc(h_m, num, ns, n0, Gamma_m)
```

Returns an isotropic `LayerStack` with `num - 1` layers, zero tilt, and zero rotation offsets.

| Argument | Meaning |
| --- | --- |
| `h_m` | Total thickness of one coating, in meters. |
| `num` | Integer number of profile sampling points, at least 2. |
| `ns` | End refractive index of the profile. |
| `n0` | Start refractive index of the profile. |
| `Gamma_m` | Profile design parameter; not a guaranteed simulated reflectance. |

Each layer has thickness `h_m / (num - 1)` and an index averaged between adjacent profile samples. `ns` and `n0` do not change the simulation's air boundaries.

For the increasing real-index profile used in the examples, choose `h_m > 0`, `ns > n0 > 0`, and `0 < Gamma_m < 0.5 * log(ns/n0)`. The implementation does not validate this domain or explicitly handle the equality limit; invalid parameters may produce warnings, NaNs, or exceptions.

### wrap_with_symmetric_arc

```text
wrap_with_symmetric_arc(core_stack, arc_stack)
```

Both arguments are `LayerStack` objects. Returns `LayerStack.concat(arc_stack, core_stack, arc_stack.reversed())`, with `core_stack.Nlayer + 2*arc_stack.Nlayer` layers. The reversed coating preserves angle values.

```python
import numpy as np
from offaxsim import build_hwp_stack, build_klopfenstein_arc, wrap_with_symmetric_arc

core = build_hwp_stack(
    3.059, 3.397, np.full(5, 5e-3), 10.0,
    [88.65, 61.68, 0.0, 61.68, 88.65],
)
arc = build_klopfenstein_arc(2e-3, 11, 3.397, 1.0, 0.03878242227997966)
coated = wrap_with_symmetric_arc(core, arc)
assert coated.Nlayer == 25
```

## Jones-to-Mueller conversion

### jones_to_mueller

```text
jones_to_mueller(J)
```

`J` is array-like with shape `(..., 2, 2)`, converted to complex128. Returns a real array with shape `(..., 4, 4)` using

```text
M[i,j] = 0.5 * trace(sigma[i] @ J @ sigma[j] @ J.conj().T)
```

The Stokes order and V sign are defined under shared conventions. The result applies as `S_out = M @ S_in`. Any imaginary part is discarded; the function does not independently validate the physical interpretation of the input matrix.

```python
import numpy as np
from offaxsim import jones_to_mueller

np.testing.assert_allclose(jones_to_mueller(np.eye(2)), np.eye(4))
quarter_wave = jones_to_mueller(np.diag([1.0, 1.0j]))
np.testing.assert_allclose(quarter_wave @ [1, 0, 1, 0], [1, 0, 0, 1])
```

### jones_to_mueller_from_elements

```text
jones_to_mueller_from_elements(tpp, tps, tsp, tss)
```

Builds `[[tpp, tps], [tsp, tss]]` and converts it. Returns a real array with shape `tpp.shape + (4, 4)` after converting the inputs with `np.asarray`.

Supply four same-shaped arrays, or four scalars. The implementation allocates using `tpp.shape`; the other three inputs must be assignable to that shape. It does not determine a joint broadcast shape from all four inputs.

## Low-level helpers

Use the simulation classes for routine sweeps. These helpers expose the existing formulation and require consistent geometry, mode ordering, and shapes.

### Geometry

```text
Geometry(theta_deg)
xi_vec(psi_deg, rho_deg)
k_dir(theta_rad=None, forward=True)
```

The constructor takes an external incidence angle in degrees and stores `theta_deg`, `theta_rad`, `sin_theta`, and `cos_theta`.

- `xi_vec` takes real tilt and rotation angles in degrees and returns a real `(3,)` optical-axis unit vector using the shared convention.
- `k_dir` returns a complex128 `(3,)` vector `[0, sin(theta), +/-cos(theta)]`. With `theta_rad=None`, it uses the stored angle. Otherwise the argument is in **radians** and may be complex. `forward=False` reverses the z component only. For a complex angle, do not interpret this as a real Euclidean unit vector.

### UniaxialMaterial

```text
UniaxialMaterial(theta_deg)
```

Stores the external incidence angle in degrees for the extraordinary-index solver. Material indices are supplied to each method as scalars.

| Method | Inputs and return value |
| --- | --- |
| `make_epsilon_matrix(no, ne)` | Returns the complex `(3,3)` tensor `diag(no**2, ne**2, no**2)`. |
| `rot_rho(rho_deg)` | Real angle in degrees; returns a real `(3,3)` rotation about z. |
| `rot_psi(psi_deg)` | Real angle in degrees; returns a real `(3,3)` rotation about x. |
| `make_epsilon_matrix_inc_rot(no, ne, rho_deg, psi_deg)` | Returns `(eps_rot, eps_rot_inv)`, both complex `(3,3)` arrays. Uses `R = rot_rho(rho_deg) @ rot_psi(psi_deg)` and `eps_rot = R @ eps @ R.T`. |
| `solve_extra_index(no, ne, xi, forward=True, t_axis='y')` | `xi` is a nonzero real `(3,)` optical-axis vector; returns the scalar effective extraordinary index for the selected branch. The result may be complex. |

`solve_extra_index` normalizes `xi` internally and rejects imaginary components larger than `1e-12` with `ValueError`. A zero vector is not explicitly rejected and leads to invalid normalization. Supply complex-typed indices when complex roots are required: NumPy real square roots can otherwise yield NaNs. Use only `'x'` or `'y'` for `t_axis`; the implementation selects x for any value other than `'y'` rather than validating the string. Keep `'y'` when using the current yz geometry.

### TMM

```text
TMM(theta_deg)
```

Stores the external incidence angle in degrees. The internal mode order is forward extraordinary, forward ordinary, backward extraordinary, backward ordinary. Unlike simulation outputs, low-level frequency-dependent matrices place frequency on the **last axis**.

| Method | Inputs | Return value |
| --- | --- | --- |
| `air_blocks()` | None | `(gm, H)`, real arrays `(4,3)` describing air modes. |
| `build_A_from_blocks(gm, H)` | Two `(4,3)` mode arrays | Boundary matrix `(4,4)` with rows `Ex, Ey, Hx, Hy`. |
| `Air_layer_0()` | None | `(A0, A0_inv)`, air boundary matrices `(4,4)` at incidence. |
| `Air_layer_f()` | None | `(Af, Af_inv)`, air boundary matrices `(4,4)` at exit. |
| `output_theta(nj)` | Complex mode indices `(4,)` | Internal angles `(4,)` in radians, using `arcsin(sin(theta)/nj)`. |
| `output_kj(thetaj, geom)` | Angles `(4,)` in radians; a matching `Geometry` | Complex direction array `(4,3)`. |
| `output_q(nj, thetaj)` | Indices and angles `(4,)` | Signed normal wavevector factors `(4,)`, `sign * nj * cos(thetaj)` with signs `+,+,-,-`. |
| `output_D(kj, xi)` | Directions `(4,3)`; real optical axis `(3,)` | Displacement mode vectors `(4,3)`. |
| `output_H(kj, D)` | Direction and displacement arrays `(4,3)` | Magnetic mode vectors `(4,3)` from cross products. |
| `output_A(ep_rot_inv, Dj, Hj, nj)` | Inverse dielectric tensor `(3,3)`, displacement/magnetic modes `(4,3)`, indices `(4,)` | `(A, A_inv)`, complex matrices `(4,4)`. |
| `output_phase(freq, thickness_m, qj)` | NumPy frequencies `(Nf,)` in Hz, scalar thickness in m, factors `(4,)` | Complex phases `(4,Nf)`, `exp(-1j*(2*pi*freq/c)*thickness_m*qj)`. |
| `output_T(A, phase, A_inv)` | Matrices `(4,4)` and phases `(4,Nf)` | Transfer matrix `(4,4,Nf)`. |
| `output_jones(T, A0_inv, Af)` | Transfer matrix `(4,4,Nf)` and air matrices `(4,4)` | Tuple `(tpp, tps, tsp, tss)` of complex arrays `(Nf,)`. |

Matrix inversion can raise `numpy.linalg.LinAlgError` for a singular mode basis. Optical-axis/propagation alignment, grazing incidence, and strongly attenuating or thick stacks require special care; there is no dedicated degeneracy or conditioning treatment.

## Additional module function

`klop_index_profile` is available from `offaxsim.arc_models`, but is not re-exported at the package root:

```text
klop_index_profile(h, num, ns, n0, Gamma_m)
```

The parameters match `build_klopfenstein_arc`, with `h` replacing `h_m`. Returns `(n_ave, z_arr, z_space)`, each with shape `(num-1,)`: averaged refractive indices, layer center positions in meters, and layer thicknesses in meters. It has the same profile-domain constraints described above.

## Validation scope

The test suite checks simulation consistency, the zero-thickness limit, lossless isotropic-slab Fresnel transmission at selected angles, Mueller conversion, and execution of a coated stack. These checks do not establish general accuracy for tilted anisotropic axes, asymmetric stacks, or lossy/evanescent cases. See the README for test commands and the notebook for worked examples.
