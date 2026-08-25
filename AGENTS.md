# AGENTS.md — centrex-state-prep

Guidance for AI coding agents working in this repository. Human-readable too.

## What this repo is

`centrex-state-prep` simulates coherent molecular **state preparation** and time
evolution for the CeNTREX TlF experiment. It propagates selected TlF quantum
states along a straight-line molecular-beam trajectory through electric,
magnetic, and (optionally) microwave fields.

Physics is done by repeated eigendecomposition: at each timestep build
`H(t)`, diagonalize, apply phases in the instantaneous eigenbasis, then
reorder eigenvectors by overlap so adiabatic state labels stay consistent.

Two packages, deliberately different in style:

| Package | Style | Purpose |
| --- | --- | --- |
| `src/state_prep` | OOP, nested objects, callables on instances | Original CPU engine. The reference implementation. |
| `src/state_prep_gpu` | Flat, functional, array-in/array-out | CuPy batched path for large scans. Optional dependency. |

## Environment & commands

Python **3.11**, `uv`-managed, `src/` layout, venv at `.venv/`. Platform is
Windows; shell examples below are PowerShell.

```powershell
uv sync                                   # install deps from uv.lock
uv pip install -e .                       # editable install of the package
.\.venv\Scripts\python.exe <script.py>    # run anything in the venv
```

GPU support is optional. Install CuPy separately (see
`src/state_prep_gpu/README.md` for the Windows no-CUDA-Toolkit recipe using
NVIDIA pip wheels).

**There is a test suite; there is still no linter/formatter config.** Do not
invent `ruff`/`black` invocations.

```bash
.\.venv\Scripts\python.exe -m pytest        # 31 tests + 1 skipped, ~70 s
```

The suite covers scan agreement (shared-slow versus repeated `run()`), the
rotating-frame construction, and the public API. `tests/test_gpu.py` skips
without CuPy. Tolerances live in `tests/conftest.py` and are named rather than
inlined, because they cannot be a single number — see the comment there.

Further verification, for anything the suite does not cover:

- running the example scripts (`examples/batched_mu_scan_cpu.py`,
  `src/state_prep_gpu/example_smoke.py`,
  `src/state_prep_gpu/example_compare_cpu.py`),
- running benchmarks in `benchmarks/`,
- comparing against the notebooks in `examples/`.

Note what the suite deliberately does *not* prove: it checks internal
consistency, so a change to the underlying `centrex_tlf` Hamiltonian would move
every result together and leave all tests passing. For dependency bumps, capture
reference outputs before and compare after.

### Benchmarks

Every benchmark is parameterised and writes JSON with `benchmark`,
`generated_at`, `environment`, `config`, `results`; `--csv` additionally writes
flattened rows.

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_storage_modes.py --n-steps 500 --output results\storage.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_microwave_scan.py --n-steps 500 --batch 5 --compare-naive --output results\microwave.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_gpu_spa2.py --n-steps 1000 --batch-sweep 5 25 --cpu --gpu --output results\gpu_spa2.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_gpu_kernels.py --batch 32 --n 32 --output results\gpu_kernels.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_convergence.py --multitone --n-steps 10000 20000 40000
.\.venv\Scripts\python.exe benchmarks\bench_parallel_backends.py --n-steps 4000 --batch 25
.\.venv\Scripts\python.exe benchmarks\bench_serial_fraction.py --n-steps 1500 --batch 25
.\.venv\Scripts\python.exe benchmarks\bench_eig_backends.py
.\.venv\Scripts\python.exe benchmarks\paired.py --self-test
```

- `bench_convergence.py` — timestep convergence over a detuning × coupling grid,
  reporting the worst cell. Convergence varies non-monotonically in both axes,
  so a single parameter point certifies nothing.
- `bench_parallel_backends.py` — serial, loky and threaded variants end to end.
- `bench_serial_fraction.py` — separates shared per-timestep work from
  per-scan-point work and reports the Amdahl ceiling that implies.
- `bench_eig_backends.py` — eigensolver × BLAS threads × parallelism grid.
- `bench_trotter_conditioning.py` — exact versus split-step propagator.
- `paired.py` — the before/after runner. Use it rather than timing pairs.

Every benchmark writes JSON with `benchmark`, `generated_at`, `environment`,
`config`, `results`; `--csv` additionally writes flattened rows. Shared setup
(SPA2 fields, Hamiltonian, microwaves, monitor states) lives in
`benchmarks/common.py` — reuse it rather than rebuilding a setup.

## Module map (`src/state_prep`)

- `trajectory.py` — `Trajectory(Rini, Vini, zfin)` with `R_t(t)` and `get_T()`.
- `core.py` — `StaticField` ABC shared by the field wrappers.
- `electric_fields.py` — `ElectricField` wrapping a callable `E_R(R)`; supports
  `+ - * neg`, `get_E_t_func(R_t)`, `plot()`. Also analytic fields
  (`linear_E_field`, `E_field_lens`, `E_field_ring`), CSV loaders
  (`Ez_from_csv`), and SPB pickle loaders (`E_SPB_from_pickle`).
- `magnetic_fields.py` — `MagneticField`, same shape as above with `B_R`.
- `hamiltonians.py` — `SlowHamiltonian(Js, trajectory, electric_field,
  magnetic_field)` builds the TlF hyperfine + Stark + Zeeman Hamiltonian via
  `centrex_tlf`; exposes `QN`, `H_R`, `get_H_t_func()`. `SlowHamiltonianOld` is
  the legacy pickle-file path.
- `microwaves.py` — `Polarization`, `MicrowaveField(Jg, Je, intensity,
  polarization, muW_freq, QN, background_field=False)`; coupling-matrix
  generation (`H_list`), rotating-frame shift matrix `D`, `calculate_rabi_rate`,
  `calculate_microwave_power`, and setters `set_frequency/set_position/set_power`.
- `intensity_profiles.py` — `GaussianBeam`, `BesselGaussianBeam`,
  `MeasuredBeam`, `BackgroundField`, all implementing `I_R(R)` / `E_R(R)`.
- `simulator.py` — the engine (~1.7k lines). `Simulator`, `SimulationResult`,
  `MicrowaveScanResult`, and `limit_blas_threads`. See below.
- `scans.py` — `scan_grid` builds the `(B, M)` detuning and prefactor arrays,
  including the case where a physically fixed source stays at zero detuning
  while the scanned fields move. `SCAN_STORAGE_DEFAULTS` carries the final-only
  storage settings for scans built on repeated `run()` calls.
- `__init__.py` — the public API. Import from `state_prep` directly
  (`import state_prep as sp`) rather than reaching into submodules.
- `utils.py` — `reorder_evecs`, `find_max_overlap_idx`, `vector_to_state`,
  `matrix_to_states`, `make_QN`, `calculate_transition_frequency`.
- `plotters.py` — `CouplingPlotter` for microwave matrix elements between
  tracked state pairs.
- `approximate_states.py` — convenient approximate uncoupled TlF states for
  J = 0, 1, 2.
- `electric_fields/` — packaged CSV + pickle field profiles (package data).

### Typical flow

1. `Trajectory(Rini, Vini, zfin)`
2. `ElectricField(E_R=...)`, `MagneticField(B_R=...)`
3. `SlowHamiltonian(Js=[0,1,2,3], trajectory=..., electric_field=..., magnetic_field=...)`
4. optional `MicrowaveField(...)` list
5. `Simulator(trajectory, electric_field, magnetic_field, initial_states_approx, hamiltonian, microwave_fields)`
6. `sim.run(...)` → `SimulationResult`, or `sim.run_microwave_scan(...)` → `MicrowaveScanResult`

`examples/batched_mu_scan_cpu.py` is the shortest end-to-end example; the
notebooks under `examples/SPA`, `examples/SPB`, `examples/SPB_high_field` are the
real workflows.

### `Simulator.run(...)` storage flags

`N_steps`, `store_every`, `store_psis`, `store_energies`, `store_probabilities`,
`store_final_probabilities`, `monitor_states`, `store_monitor_probabilities`,
`store_final_monitor_probabilities`, `eig_backend` (`"zheevd"` default, or
`"numpy"`), `blas_threads` (default `1`).

`blas_threads` pins BLAS for the duration of the evolution loop. The
Hamiltonians are small enough that OpenBLAS's internal threading is overhead
rather than speedup, and the limit is scoped and restored on exit so it does not
affect the rest of a session. It does not change results at all. Pass `None` to
leave BLAS configuration alone.

Storage mode dominates memory and a large share of runtime. For scans, prefer
final-only or monitor-only storage; benchmark numbers are in
`PERFORMANCE_BENCHMARK_RESULTS.md`.

`monitor_states` are mapped to the nearest eigenstate at t=0 (max overlap with
`V_ini`) and then tracked adiabatically — they are *not* fixed basis states.

### `Simulator.run_microwave_scan(...)`

Keyword-only. Reuses one shared slow-Hamiltonian diagonalization across all scan
points, varying only microwave parameters. This is the fast path — use it
instead of looping `run()` whenever `H_slow(t)` is identical across points.

- `detunings_hz` and `intensity_prefactors` must have identical shapes:
  `(B, M)` for batch `B` and `M` microwave fields; 1D/scalar forms are broadcast.
- Couplings scale as `sqrt(intensity_prefactors)` — the prefactor is an
  *intensity/power* ratio, not an amplitude ratio.
- `workers > 1` (or `-1`) chunks the batch across loky processes; each chunk
  repeats the shared slow diagonalization, but concurrently, so that duplication
  costs CPU rather than wall time. This is a substantial win at production scan
  sizes and produces bitwise identical output. It loses badly on short scans,
  where process startup dominates, and there is no automatic size-based
  fallback, so keep `workers=1` there. `workers=1` never enters the loky path at
  all.
- Multiple tones shifting the same excited-J manifold are rejected unless
  `allow_multitone_same_manifold=True`, which keeps the first field as the
  rotating-frame reference and applies a beat phase to the rest.

## Module map (`src/state_prep_gpu`)

Flat and array-based on purpose — it does *not* consume `state_prep` objects.

- `slow_hamiltonian.py` — `terms_from_centrex_tlf`, `BatchedSlowHamiltonianTerms`,
  `slow_hamiltonian_batch(terms, E, B)`.
- `simulator.py` — `simulate_batched(...)` plus dense-microwave,
  scalar-envelope, compact-microwave, and shared-slow scan variants.
- `_reorder.py` — batched eigenvector reordering on GPU.
- `_cupy.py` — lazy CuPy import + Windows CUDA DLL directory setup. CuPy is
  never imported at package import time; GPU entry points raise an actionable
  error if it is missing.
- `example_smoke.py`, `example_compare_cpu.py`, `benchmark_compare.py`,
  `benchmark_spa2_real.py` — runnable checks.

Inputs you must precompute: slow-Hamiltonian terms (`Hff`, `HSx/y/z`,
`HZx/y/z`), field time series `E_t`, `B_t` with shape `(T, B, 3)`, and optionally
`H_mu_t` with shape `(T-1, B, n, n)` plus a `D_mu` shift.

Keep `state_prep_gpu` importable without CuPy. Any change that imports `cupy` at
module scope is a bug.

## Units & numerical conventions

These are domain-specific and easy to break. Follow them exactly.

- Time `t_array` in **seconds**; plots usually show µs (`t_array / 1e-6`).
- Positions in **meters**; plots sometimes show cm (`z_array / 1e-2`).
- `SimulationResult.energies` are **angular** frequencies — displayed as
  `energies / (2*np.pi)` in Hz.
- Detunings passed to scans are in **Hz** (converted internally).
- Fields expose both XYZ and an internal `r` basis via the conversion matrix
  `R_to_r`; keep the `get_*_R` / `get_*_r` distinction straight.
- Eigenvector ordering: after every diagonalization along a sweep, track labels
  with `utils.reorder_evecs`. Shapes and ordering must stay consistent.
- `SimulationResult.save_to_pickle` uses `dill` — keep it picklable (lambdas and
  closures are stored on these objects, which is why `dill` is required).

## Conventions to follow

- Small composable callables stored on instances (`E_R`, `B_R`, `H_R`, `*_t`).
- `dataclasses` for containers, with `__post_init__` for derived fields.
- NumPy vectorization; `np.einsum` is already used for overlaps.
- Type hints are present but not enforced; match surrounding style.
- Import order: stdlib, third-party, then local `from .module import ...`.
- New field → callable `*_R(R) -> np.ndarray` wrapped in
  `ElectricField`/`MagneticField`, plus `get_*_t_func(R_t)` if used along a
  trajectory.
- New Hamiltonian → provide `get_H_t_func() -> Callable[[float], np.ndarray]`
  and store `QN` on the object.
- New outputs → store arrays on `SimulationResult` / `MicrowaveScanResult` and
  add small helper methods; do not add side effects inside `Simulator`.

## Do not

- Don't "fix" physics, units, or constants unless explicitly asked. Many
  scalings are experiment-specific and look wrong out of context.
- Don't rename public attributes crossing module boundaries: `QN`, `H_list`,
  `D`, `t_array`, `psis`, `V_ini`, `V_fin`, `probabilities_final`.
- Don't add dependencies. Stack is `numpy`, `scipy`, `centrex_tlf`,
  `matplotlib`, `pandas`, `polars`, `joblib`, `dill`, `tqdm`, `astropy`.
- Don't reformat whole files; keep diffs minimal.
- Don't edit files under `.venv/`, `build/`, or `src/*.egg-info/`.
- Don't rewrite notebooks wholesale — they carry executed outputs used for
  paper figures. Edit targeted cells.
- Don't commit generated CSVs into `results/` or the notebook `results/`
  directories unless asked.

## Self-check before proposing changes

- Touched diagonalization or time evolution? Verify eigenvector ordering and
  array shapes are preserved end to end.
- Touched coordinates? Confirm XYZ vs `r` basis (`R_to_r`) and units.
- Touched serialization? `dill` compatibility must hold.
- Touched a scan path? Confirm CPU and GPU results still agree
  (`src/state_prep_gpu/example_compare_cpu.py`).
- Claimed a speedup? Back it with a `benchmarks/` run, not reasoning — and run
  it **with repeats**. Baseline and candidate must be measured on the same
  machine in the same session, interleaved, with `--warmup 1 --repeat 5` or
  more. Report mean ± std, not a single number, and call the change unproven if
  the difference is within the combined spread. See the Benchmarking Protocol
  section of `IMPROVEMENTS.md`.

## Reference documents in-repo

- `README.md` — user-facing setup, minimal example, and scan usage.
- `REPO_SCAN.md` — full repository inventory and capability list.
- `PERFORMANCE_IMPROVEMENTS.md` — prioritized optimization backlog with findings.
- `PERFORMANCE_BENCHMARK_RESULTS.md` — measured numbers and environment.
- `src/state_prep_gpu/README.md` — GPU install and API notes.
- `copilot-instructions.md` — a pointer back to this file.
- `IMPROVEMENTS.md` — measured performance findings, what was rejected and why,
  and the remaining backlog.
