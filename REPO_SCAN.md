# Repository Scan

Generated: 2026-06-14. Partially refreshed 2026-08-16; sections marked below
were updated, the rest still reflect the original scan.

## What This Repository Does

`centrex-state-prep` is a Python package for simulating coherent molecular state preparation and time evolution for the CeNTREX TlF experiment. The package models how selected TlF quantum states evolve along a molecular beam trajectory through electric fields, magnetic fields, and optional microwave fields.

The main package is `state_prep`, with a separate `state_prep_gpu` package that implements a flatter, array-based CuPy path for batched GPU scans.

## Main Capabilities

- Build straight-line molecular trajectories from initial position, velocity, and final z position.
- Represent static electric and magnetic fields as callables over position.
- Load measured or finite-element electric-field profiles from CSV and pickle files.
- Build slow molecular Hamiltonians using `centrex_tlf` for TlF hyperfine, Stark, and Zeeman terms.
- Build microwave coupling Hamiltonians for selected rotational transitions and polarization/intensity profiles.
- Propagate state vectors through a time-dependent Hamiltonian by repeated eigendecomposition and phase evolution.
- Track adiabatic eigenstates by matching eigenvectors between timesteps.
- Store full trajectories, final probabilities, monitor-state probabilities, and energies.
- Run CPU microwave parameter scans that reuse shared slow-Hamiltonian diagonalization.
- Run GPU batched simulations with CuPy for larger scan workloads.
- Plot state probabilities, energies, microwave couplings, and field profiles.

## Package Layout

### `src/state_prep`

This is the original CPU-oriented simulation package.

- `simulator.py`: Main simulation engine. Defines `Simulator`, `SimulationResult`, and `MicrowaveScanResult`. Handles non-microwave evolution, microwave evolution, final-only storage, downsampled storage, monitor states, and CPU batched microwave scans.
- `hamiltonians.py`: Defines `SlowHamiltonian`, which builds the TlF slow Hamiltonian from `centrex_tlf` field-independent terms and field callables.
- `electric_fields.py`: Electric-field wrappers and helpers for analytic fields, CSV interpolation, and SPB pickle-backed finite-element field profiles.
- `magnetic_fields.py`: Magnetic-field wrapper with plotting support.
- `microwaves.py`: Microwave polarization, intensity coupling, rotating-frame diagonal shifts, coupling matrix generation, Rabi-rate and power helpers.
- `intensity_profiles.py`: Gaussian, Bessel-Gaussian, measured, and uniform background microwave intensity profiles.
- `trajectory.py`: Straight-line trajectory object.
- `utils.py`: Eigenvector reordering, state-vector conversion, basis generation, legacy Hamiltonian loading, and transition-frequency calculation.
- `plotters.py`: `CouplingPlotter` for extracting and plotting microwave matrix elements between tracked state pairs.
- `approximate_states.py`: Convenient approximate uncoupled TlF states for J = 0, 1, and 2 singlet/triplet-like states.
- `core.py`: Shared abstract static-field base class.

### `src/state_prep_gpu`

This is a newer GPU-oriented package. It deliberately avoids the original nested object graph and expects precomputed arrays.

- `slow_hamiltonian.py`: Converts `centrex_tlf` Hamiltonian terms to CuPy arrays and evaluates batched slow Hamiltonians.
- `simulator.py`: CuPy batched evolution functions, including dense microwave input, scalar-envelope microwave scans, compact microwave construction, and shared-slow-Hamiltonian scans.
- `_reorder.py`: Batched eigenvector reordering on GPU.
- `_cupy.py`: Lazy CuPy import and Windows CUDA DLL directory setup.
- `example_smoke.py`: Minimal GPU smoke test.
- `example_compare_cpu.py`: CPU/GPU numerical comparison for slow-Hamiltonian evolution.
- `benchmark_compare.py`: Synthetic CPU/GPU benchmark.
- `benchmark_spa2_real.py`: SPA2-like benchmark using the real simulation setup.
- `README.md`: GPU installation notes and API overview.

## Example And Data Layout

- `examples/SPA`: Jupyter notebooks for state preparation by adiabatic passage, including experimental-verification and parameter-scan workflows.
- `examples/SPB` and `examples/SPB_high_field`: Jupyter notebooks for state preparation by branching / adiabatic passage variants.
- `examples/SPA/Experimental verification/results`: CSV outputs from parameter scans and paper-comparison workflows.
- `src/state_prep/electric_fields`: CSV and pickle electric-field profiles used by the example notebooks and simulations.
- `examples/SPA/Experimental verification/kdes`: Pickled KDEs for beam-distribution inputs.

The repository currently contains 29 notebooks, 29 CSV files, and many packaged electric-field pickle assets.

## Core Simulation Flow

1. Define a `Trajectory` with `Rini`, `Vini`, and `zfin`.
2. Define `ElectricField` and `MagneticField` objects whose callables evaluate fields at a position.
3. Build a `SlowHamiltonian` for selected rotational manifolds `Js`.
4. Optionally define one or more `MicrowaveField` objects with intensity and polarization models.
5. Instantiate `Simulator` with approximate initial states.
6. Call `Simulator.run(...)` for a single evolution, or `Simulator.run_microwave_scan(...)` for batched CPU microwave scans.
7. Inspect `SimulationResult` or `MicrowaveScanResult` probabilities, final states, energies, and helper plotting methods.

The time evolution uses small-step propagation. At each timestep it builds the Hamiltonian, diagonalizes it, applies phases in the instantaneous eigenbasis, and reorders eigenvectors to preserve adiabatic state labels.

## Dependencies And Packaging

The project uses a `src/` layout and declares modern packaging metadata in `pyproject.toml`. It requires Python 3.11 or newer.

Key dependencies:

- `numpy`
- `scipy`
- `centrex_tlf`
- `matplotlib`
- `pandas`
- `polars`
- `joblib`
- `dill`
- `tqdm`
- `astropy`
- `ipykernel`

GPU support is optional and depends on a compatible CuPy package such as `cupy-cuda12x`.

`pyproject.toml` is the single source of packaging metadata, under the
distribution name `centrex-state-prep`. The legacy `setup.py` and the conda
`state_prep.yml` were removed on 2026-08-16: the former declared a different
distribution name with a stale dependency list, and the latter pinned py39
packages against a project requiring 3.11+.

## Added Since The Original Scan (2026-08-16)

- `src/state_prep/__init__.py` now exports the public API; import
  `state_prep` directly rather than reaching into submodules.
- `src/state_prep/scans.py`: `scan_grid` and `SCAN_STORAGE_DEFAULTS` for
  constructing scans.
- `src/state_prep/microwaves.py`: `build_rotating_frame_shift`, the single
  implementation of the rotating-frame diagonal shift, previously duplicated
  across the simulator, the benchmarks and the GPU benchmark.
- `src/state_prep/simulator.py`: `limit_blas_threads` and the `blas_threads`
  argument; multitone same-manifold scans; process-parallel scans over loky;
  accessors on `MicrowaveScanResult`.
- `tests/`: 20 tests plus a GPU suite skipped without CuPy.
- `benchmarks/`: `paired.py` (before/after runner with repeats and verdicts)
  plus `bench_convergence.py`, `bench_eig_backends.py`,
  `bench_parallel_backends.py`, `bench_serial_fraction.py`,
  `bench_trotter_conditioning.py`.
- `scripts/`: SPA2 background-feature analysis and plotting.
- `AGENTS.md`, `CLAUDE.md`, `IMPROVEMENTS.md`.

## Current Repository State Notes

**Superseded.** Everything listed below was committed on 2026-08-16. At the
time of the original scan the worktree had uncommitted changes in:

- `examples/SPA/Experimental verification/SPA2 - only SPA background - start in J1.ipynb`
- `examples/SPA/Experimental verification/SPA2 for paper.ipynb`
- `src/state_prep/utils.py`
- `src/state_prep_gpu/README.md`
- `src/state_prep_gpu/_cupy.py`
- `src/state_prep_gpu/simulator.py`

It also had untracked GPU benchmark/example files and CSV result files. This scan did not modify those files.

