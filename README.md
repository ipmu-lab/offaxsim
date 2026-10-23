# offaxsim

`offaxsim` is a Python library for 4×4 transfer-matrix simulations of uniaxial multilayer stacks. It computes complex Jones transmission matrices and the corresponding Mueller matrices over frequency and stack rotation angle.

The package includes single-layer and multilayer simulations, layer-stack builders, and discretized Klopfenstein antireflection coatings (ARCs). Both exterior media are air (refractive index 1). Reflection coefficients are not returned by the current simulation API.

## Requirements

- Python 3.9 or later, as declared in `pyproject.toml`.
- NumPy >= 1.23, SciPy >= 1.9, and tqdm >= 4.0.
- Optional plotting support: Matplotlib >= 3.6.

The examples below were checked with Python 3.10.9, NumPy 1.26.4, SciPy 1.14.1, and tqdm 4.67.1. Other supported versions have not been checked here.

## Installation

Obtain a copy of the repository and open a terminal in the directory containing this project's `pyproject.toml`. If this package is placed inside a larger repository, change into the package subdirectory first.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

The editable installation uses the source files in this checkout, so code edits take effect without reinstalling. For a regular installation, use `python -m pip install .` instead.

To install the optional plotting dependency:

```bash
python -m pip install -e ".[plot]"
```

Check the import:

```bash
python -c "import offaxsim; print(offaxsim.__version__)"
```

The current package version is `0.1.0`. Run the Python examples below in the activated environment, either from a script or an interactive Python session. There is no command-line simulation entry point.

## API reference

See the [Python API reference](docs/api.md) for all public classes and functions, argument units, array shapes, return values, and low-level helpers.

## Example notebook

Open [the worked example notebook](src/notebooks/example_usage.ipynb) for an English walkthrough of single-layer and multilayer simulations, transmission plots, Stokes conventions, and basic consistency checks. The notebook includes example outputs and can be rerun from top to bottom.

From the project directory, install the notebook tools in the same environment as the package:

```bash
python -m pip install -e ".[plot]" jupyterlab ipykernel
python -m ipykernel install --user --name offaxsim --display-name "Python (offaxsim)"
python -m jupyterlab src/notebooks/example_usage.ipynb
```

Select the **Python (offaxsim)** kernel. Saving numerical results is optional and disabled by default in the notebook.

## Quick start: a single layer

```python
import numpy as np
from offaxsim import SingleLayerSimulation

freq = np.array([60.0, 90.0, 120.0]) * 1e9  # Hz
rho_arr = np.array([0.0, 45.0, 90.0])       # degrees

sim = SingleLayerSimulation(theta_deg=10.0)
J, M = sim.run(
    freq=freq,
    rho_arr=rho_arr,
    no=3.059 - 0.000137655j,
    ne=3.397 - 0.00023779j,
    thickness_m=5e-3,
    psi_deg=30.0,
    show_progress=False,
)

print(J.shape)  # (3, 3, 2, 2)
print(M.shape)  # (3, 3, 4, 4)
assert np.isfinite(J).all()
assert np.isfinite(M).all()

# First rotation angle and second frequency (rho=0 deg, f=90 GHz).
print(J[0, 1])
print(M[0, 1])

# Save arrays with their sweep coordinates.
np.savez("single_layer.npz", freq_Hz=freq, rho_deg=rho_arr, J=J, M=M)
```

These material parameters follow the bundled example notebook; they are illustrative inputs, not a general material calibration. Refractive indices are constant across the frequency sweep in this API.

## Multilayer stack with optional AR coatings

```python
import numpy as np
from offaxsim import (
    MultiLayerSimulation,
    build_hwp_stack,
    build_klopfenstein_arc,
    wrap_with_symmetric_arc,
)

freq = np.array([60.0, 90.0, 120.0]) * 1e9
rho_arr = np.array([0.0, 45.0, 90.0])

core = build_hwp_stack(
    no=3.059 - 0.000137655j,
    ne=3.397 - 0.00023779j,
    thickness_arr_m=np.full(5, 5e-3),
    psi_deg=10.0,
    opt_axis_arr_deg=[88.65, 61.68, 0.0, 61.68, 88.65],
)
arc = build_klopfenstein_arc(
    h_m=2e-3,
    num=11,
    ns=3.397,
    n0=1.0,
    Gamma_m=0.03878242227997966,
)
stack = wrap_with_symmetric_arc(core, arc)
# To omit the coatings, use: stack = core

J, M = MultiLayerSimulation(theta_deg=10.0).run(
    freq, rho_arr, stack, show_progress=False,
)
print(stack.Nlayer)  # 25: 10 ARC layers + 5 core layers + 10 ARC layers
print(J.shape)      # (3, 3, 2, 2)
print(M.shape)      # (3, 3, 4, 4)
assert np.isfinite(J).all()
assert np.isfinite(M).all()
```

`build_hwp_stack` assigns the same material indices and tilt to every core layer. It does not calculate thicknesses or optimize a half-wave plate design.

For arbitrary per-layer parameters, construct `LayerStack(no_arr, ne_arr, d_arr, psi_arr, opt_axis_arr)` directly. All five inputs must be one-dimensional arrays with the same length. Isotropic layers use `no == ne`.

`build_single_layer(no, ne, thickness_m, psi_deg, opt_axis_deg=0.0)` produces a one-layer `LayerStack` for use with `MultiLayerSimulation`.

`LayerStack.concat(a, b, ...)` concatenates arrays in the supplied order. `stack.reversed()` reverses their order; it does not negate or transform the optical-axis angles. `wrap_with_symmetric_arc(core, arc)` returns `arc + core + arc.reversed()`.

### ARC parameters

| Parameter | Meaning |
| --- | --- |
| `h_m` | Total thickness of one coating, in meters. |
| `num` | Number of profile sampling points; produces `num - 1` uniform-thickness layers. Use an integer >= 2. |
| `ns`, `n0` | End and start indices of the coating profile. These do not change the simulator's air boundaries. |
| `Gamma_m` | Profile design parameter, not a guaranteed simulated stack reflectance. |

For the increasing, real-index profile in this example, use `h_m > 0`, `ns > n0 > 0`, and `0 < Gamma_m < 0.5 * log(ns / n0)`. The implementation does not validate these conditions; the limiting case where the inequality becomes equality is not handled explicitly.

## Inputs and geometry

| Input | Meaning and units |
| --- | --- |
| `theta_deg` | External incidence angle from the +z surface normal, in degrees. |
| `freq` | One-dimensional frequency array in Hz; use positive frequencies. Even a single frequency must be an array, e.g. `[90e9]`. |
| `rho_arr` | One-dimensional array of stack rotation angles, in degrees. |
| `no`, `ne` | Ordinary and principal extraordinary refractive indices; complex scalars are accepted. |
| `thickness_m`, `d_arr` | Layer thicknesses in meters. |
| `psi_deg`, `psi_arr` | Optical-axis tilt out of the xy plane, in degrees. |
| `opt_axis_arr` | Per-layer rotation offsets in degrees; effective rotation is `rho_arr[i] + opt_axis_arr[layer]`. |
| `t_axis` | Extraordinary-index solver option; keep the default `"y"` for the implemented yz incidence plane. Selecting `"x"` does not rotate the rest of the geometry. |
| `return_mueller` | Defaults to `True`. If `False`, the return value is `(J, None)`. |
| `show_progress` | Defaults to `True`; displays progress over rotation angles. |

The layer normal is z and the plane of incidence is yz. The incident propagation direction is `[0, sin(theta), cos(theta)]`. The optical-axis unit vector is

```text
xi = [-cos(psi) sin(rho), cos(psi) cos(rho), sin(psi)]
```

Here the trigonometric arguments are the corresponding angles in radians. Thus `psi=0, rho=0` places the optical axis along +y; `psi=0, rho=90 deg` places it along -x. The unrotated dielectric tensor is `diag(no**2, ne**2, no**2)`.

The forward air polarization basis is `p = [0, cos(theta), -sin(theta)]` and `s = [1, 0, 0]`. At normal incidence these become +y and +x, respectively.

Internally, the phase factors are `exp(-1j * (2*pi*freq/c) * thickness * q)`. The bundled examples use negative imaginary refractive indices for loss. Complex-index and evanescent-mode behavior should be checked against a reference for the intended application.

## Outputs and polarization conventions

Both simulation classes return `(J, M)`:

| Output | Shape | Type |
| --- | --- | --- |
| `J` | `(Nrho, Nf, 2, 2)` | Complex Jones transmission coefficients. |
| `M` | `(Nrho, Nf, 4, 4)` | Real Mueller matrix, or `None` if disabled. |

The frequency and angle axes remain present even for one-element sweeps. The Jones elements are stored as

```text
J = [[tpp, tps],
     [tsp, tss]]
```

For the returned Jones matrix, rows label output components and columns label input components in the `(p, s)` basis. Thus the package uses `E_out = J @ E_in`. For the current air-to-air boundaries, squared magnitudes give individual polarization-channel power ratios; sum both output components to include cross-polarized transmission:

```python
T_p_input = np.abs(J[..., 0, 0])**2 + np.abs(J[..., 1, 0])**2
T_s_input = np.abs(J[..., 0, 1])**2 + np.abs(J[..., 1, 1])**2
```

The converter computes `M[i,j] = 0.5 * trace(sigma[i] @ J @ sigma[j] @ J.conj().T)` and returns its real part. Stokes vectors are ordered `(I, Q, U, V)`, with `S_out = M @ S_in` for each angle/frequency pair.

For the implemented Pauli matrices and coherency matrix `C = E E†` in the `(p, s)` basis:

```text
I = Cpp + Css
Q = Cpp - Css
U = Cps + Csp
V = i * (Cps - Csp)
```

This convention follows the Pauli matrices used by the implementation. Check the sign convention for V before comparing circular polarization with other software.

Use `jones_to_mueller(J)` to convert an existing array of shape `(..., 2, 2)`. The helper `jones_to_mueller_from_elements(tpp, tps, tsp, tss)` allocates its output from the shape of `tpp`; supplying four same-shaped arrays is the safe usage.

## Automated tests

From the directory containing `pyproject.toml`, install the test dependencies and run:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The suite contains 17 parameterized test cases covering:

- Single-layer versus multilayer agreement for the same layer.
- Zero-thickness identity transmission at three incidence angles.
- Complex Jones transmission of a lossless isotropic slab against an independent Fresnel multiple-reflection series at 0, 20, and 45 degrees, including p/s amplitudes, phases, and zero cross-polarization.
- Singleton sweep dimensions and disabled Mueller output in both simulation classes.
- A 25-layer ARC/core/ARC stack with finite outputs and consistent unpolarized transmission.
- Identity and quarter-wave-retarder Mueller matrices, the documented V sign, and batched Mueller action against directly computed electric-field Stokes vectors.

The Fresnel reference does not use the package's transfer-matrix helpers. These tests do not establish general accuracy for tilted anisotropic axes, asymmetric stacks, or lossy/evanescent propagation.

## Basic checks

The following check compares the two simulation paths for an identical single layer and verifies the zero-thickness and identity-Jones limits:

```python
import numpy as np
from offaxsim import (
    SingleLayerSimulation, MultiLayerSimulation,
    build_single_layer, jones_to_mueller,
)

freq = np.array([60.0, 90.0, 120.0]) * 1e9
rho = np.array([0.0, 45.0, 90.0])
params = dict(
    no=3.059 - 0.000137655j,
    ne=3.397 - 0.00023779j,
    thickness_m=5e-3,
    psi_deg=30.0,
)
single = SingleLayerSimulation(10.0)
J1, M1 = single.run(freq, rho, **params, show_progress=False)
J2, M2 = MultiLayerSimulation(10.0).run(
    freq, rho, build_single_layer(**params), show_progress=False,
)
np.testing.assert_allclose(J1, J2, rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(M1, M2, rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(jones_to_mueller(np.eye(2)), np.eye(4))

zero_params = dict(params, thickness_m=0.0)
J0, _ = single.run(freq, rho, **zero_params, show_progress=False)
np.testing.assert_allclose(
    J0, np.broadcast_to(np.eye(2), J0.shape), atol=1e-12,
)
print("Basic checks passed")
```

These are consistency checks, not a full independent validation of the electromagnetic formulation.

The existing `src/notebooks/consistency_check.ipynb` compares selected transmission intensities with stored reference arrays in `src/notebooks/original/`. To use its relative data paths, run the notebook with `src/notebooks` as the working directory. A Jupyter environment must be installed separately. The notebook's final cell uninstalls `offaxsim`; skip that cell when retaining the installation. Its plots are visual comparisons, not automated assertions.

## Current limitations

- Exterior media are fixed to air. General incident/substrate boundary indices are not exposed.
- Materials are uniaxial or isotropic, with one constant index pair per layer per run. To model dispersion, run individual frequencies with the corresponding indices.
- The current high-level API returns transmission only; it does not return reflection or absorption.
- Layer products are implemented as `Ttot = Ti @ Ttot`, yielding `T_last @ ... @ T_first`. Confirm physical layer orientation against an independent reference before interpreting an asymmetric stack.
- Optical-axis/propagation alignment can make the cross-product mode basis singular. Grazing incidence and strongly attenuating or thick stacks may also be numerically problematic. There is no dedicated degeneracy or conditioning treatment.
- Input validation is limited. Supply correctly shaped arrays, consistent units, and physically appropriate parameters.
- The examples and checks above do not establish general accuracy for tilted axes, cross-polarization signs, arbitrary asymmetric stacks, or lossy/evanescent cases.

## Project layout

```text
pyproject.toml                  Package metadata and dependencies
src/offaxsim/                   Python package
    geometry.py                Coordinate and optical-axis helpers
    material.py                Uniaxial dielectric tensor and index solver
    tmm.py                     Transfer matrices and transmission extraction
    layers.py                  Layer-stack container and builders
    arc_models.py              Klopfenstein coating profile
    simulation.py              Frequency/rotation sweeps
    jones_mueller.py           Jones-to-Mueller conversion
src/notebooks/                 Example and comparison notebooks
```

## Authors and contributions

- **Original implementation:** Yurika Hoshino — Saitama University.
- **Further development and packaging:** Ryota Takaku — Okayama University; Kavli IPMU.

offaxsim builds on the original implementation by Yurika Hoshino. Ryota Takaku continued its development and organized it into a reusable Python package with documentation, worked examples, and tests. Affiliations identify the contributors; they do not by themselves identify the copyright holders.

## License

This package is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Yurika Hoshino and Ryota Takaku.
