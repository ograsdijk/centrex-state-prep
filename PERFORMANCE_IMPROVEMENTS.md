# Performance Improvements

Generated: 2026-06-14

This file combines the source-level performance scan with the benchmark results recorded in `PERFORMANCE_BENCHMARK_RESULTS.md`. The recommendations below are ordered by current evidence and expected impact.

## Benchmark-Informed Summary

Measured results:

| Area | Benchmark Result | Recommendation |
| --- | ---: | --- |
| CPU microwave scans using shared slow Hamiltonian | `1.90x` faster than repeated single runs | High priority; use and improve this path |
| Output storage controls | about `1.17x` faster in a small case, much lower memory | High priority for scans and notebooks |
| Synthetic batched slow-H GPU path | `3.46x` faster than CPU | Keep and benchmark at larger scan sizes |
| Real SPA2-like GPU path | only `1.06x` faster than CPU | Improve only after profiling; do not assume automatic win |
| GPU `slow_hamiltonian_batch(out=...)` reuse | inconclusive; microbenchmark timed out | Do not prioritize until a better benchmark exists |

The most defensible near-term work is therefore:

1. Make shared-slow microwave scans the standard route for detuning/power/background scans.
2. Make final-only and monitor-only storage the default pattern for scans.
3. Add proper benchmark/profiling scripts so real workloads can be compared without editing source or using inline scripts.
4. Profile why the real SPA2 GPU path only slightly beats CPU despite a strong synthetic GPU result.
5. Only then optimize GPU buffer reuse, CPU field precomputation, or eigenvector reordering.

## Existing Performance Work Already Present

- `Simulator.run(...)` supports `store_every`, final-only probabilities, monitor-only probabilities, and a LAPACK `zheevd` backend.
- `_time_evolve(...)` and `_time_evolve_mu(...)` apply propagators without explicitly forming the full timestep unitary `U_dt`.
- `run_microwave_scan(...)` reuses the slow-Hamiltonian diagonalization for microwave scans where the slow fields are shared.
- `state_prep_gpu` provides a flat CuPy API for batched slow-Hamiltonian and microwave-evolution workloads.
- `state_prep_gpu.simulator` includes dense, compact, scalar-envelope, and shared-slow microwave paths.
- Benchmark scripts exist for synthetic CPU/GPU and real SPA2-like CPU/GPU comparisons.

## Priority 1: Standardize Shared-Slow Microwave Scans

Benchmark evidence:

- SPA2-like scan, `N_steps=1000`, batch `5`, `n=64`.
- `Simulator.run_microwave_scan(...)`: `3.352 s`.
- Repeated final-only `Simulator.run(...)`: `6.353 s`.
- Speedup: `1.90x`.
- Max probability difference: `7.314e-10`.

Recommendation:

- Route notebook detuning, power, and background-intensity scans through `Simulator.run_microwave_scan(...)` whenever the trajectory, electric field, magnetic field, and slow Hamiltonian are shared across scan points.
- Add examples in the notebooks or README showing the shared-slow API as the default scan pattern.
- Treat notebook-level loops over `Simulator.run(...)` as the fallback path only when the slow Hamiltonian itself changes between scan points.

Next implementation targets:

- Add a small helper/wrapper for common scan shapes so notebooks do not need to construct `detunings_hz` and `intensity_prefactors` manually each time.
- Add a regression test comparing a small shared-slow scan against repeated single runs.
- Add timing output or an optional `return_timings=True` path for scan development.

## Priority 2: Use Final-Only Or Monitor-Only Storage In Scans

Benchmark evidence:

- Non-microwave real setup, `N_steps=2500`, `n=36`, four initial states.
- Full default output: `0.341 s`, `8.93 MiB`.
- Downsample every 20: `0.291 s`, `0.45 MiB`.
- Final-only probabilities: `0.292 s`, effectively no full trajectory output.
- Final monitor-only: `0.289 s`, effectively no full trajectory output.

Recommendation:

For scans, prefer:

- `store_psis=False`
- `store_energies=False`
- `store_probabilities=False`
- `store_final_probabilities=True`
- `monitor_states=[...]` when only a few target states matter
- `store_final_monitor_probabilities=True`
- `store_every > 1` only when a time trace is required

Why this matters:

- Runtime improvement was modest in the small benchmark because eigensolves dominate.
- Memory reduction was large and will matter more in long scans, notebook sessions, and multi-parameter sweeps.
- Smaller result objects are easier to save, plot, and compare.

Next implementation targets:

- Add a documented `scan_defaults` or convenience method for final-only scan settings.
- Update example notebooks that perform scans to avoid full probability/energy/psi storage unless the notebook explicitly plots time traces.
- Add a result-size estimate utility for planned scan dimensions.

## Priority 3: Add Parameterized Benchmark And Profiling Scripts

Current benchmark gap:

- Existing scripts are useful, but key parameters are hard-coded.
- The real SPA2 GPU benchmark showed only `1.06x` speedup, while the synthetic benchmark showed `3.46x`.
- CPU model information could not be recorded because `Get-CimInstance Win32_Processor` returned access denied.
- The `slow_hamiltonian_batch(out=...)` microbenchmark timed out repeatedly and needs a better design.

Recommendation:

Create benchmark scripts that accept command-line options for:

- `N_steps` / `T`
- batch size
- Hilbert-space dimension / selected `Js`
- number of initial states
- storage mode
- CPU vs GPU mode
- microwave representation
- repeat count and warmup count
- output path for JSON/CSV results

Recommended scripts:

- `bench_storage_modes.py`: CPU storage settings and memory use.
- `bench_microwave_scan.py`: shared-slow scan vs repeated single runs.
- `bench_gpu_spa2.py`: CPU shared-slow vs GPU shared-slow with batch-size sweep.
- `bench_gpu_kernels.py`: CUDA-event microbenchmarks for slow-H construction and probability calculation.

Profiling should time at least these sections:

- field and Hamiltonian construction
- slow-Hamiltonian eigensolve
- rotating-frame eigensolve
- eigenvector reordering
- microwave matrix construction
- propagation matmuls
- probability calculation
- result storage

## Priority 4: Investigate Real-Workload GPU Bottlenecks

Benchmark evidence:

- Synthetic slow-H benchmark: GPU `0.871 s`, CPU `3.012 s`, speedup `3.46x`.
- Real SPA2-like benchmark: GPU `74.840 s`, CPU `79.255 s`, speedup `1.06x`.
- Real benchmark max final probability difference: `4.228e-06`.

Interpretation:

The GPU path is viable and numerically close, but the current real SPA2 shape does not use it efficiently enough to justify assuming GPU will always be faster. The synthetic result says the GPU can help; the real result says the real bottleneck is not solved yet.

Likely areas to profile:

- Per-step `cp.linalg.eigh` for `(B, n, n)` rotating-frame Hamiltonians.
- Shared-slow path overhead when batch is only moderate (`B=25` in the real benchmark).
- Microwave matrix construction and `cp.tensordot` cost per timestep.
- Repeated small kernel launches.
- Host/device conversions at function boundaries.
- Whether larger scan batches improve GPU occupancy.

Next implementation targets:

- Sweep batch sizes for the real SPA2 benchmark, for example `B=5, 25, 100, 250`.
- Compare `return_final_probabilities=True` vs monitor-only direct probability calculation.
- Add CUDA-event timing inside the GPU loop to separate eigensolve, microwave assembly, propagation, and final probability costs.
- Keep CPU shared-slow as the default for small scans until GPU shows a clear real-workload win.

## Priority 5: Fix Or Document Windows CuPy DLL Setup

Benchmark evidence:

- CuPy could see the RTX 5070 Ti.
- GPU execution initially failed because NVRTC could not open `nvrtc-builtins64_129.dll`.
- The DLL existed under `.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin`.
- Prepending the NVIDIA pip-package `bin` directories to `PATH` allowed GPU benchmarks to run.

Recommendation:

- Update `state_prep_gpu` Windows setup docs with the `PATH` workaround.
- Consider extending `_add_windows_cuda_dll_dirs()` so NVRTC builtins are found reliably before CuPy kernel compilation.
- Add a small GPU diagnostic command that verifies both device visibility and a trivial CuPy kernel allocation/operation.

This is not a numerical performance improvement, but it directly affects whether GPU acceleration is usable.

## Priority 6: Vectorize The CPU Shared-Slow Inner Loop

Current code:

- `src/state_prep/simulator.py:812`: pre-rotates microwave fields into the slow-eigenbasis.
- `src/state_prep/simulator.py:819`: loops over batch points in Python.
- `src/state_prep/simulator.py:825`: diagonalizes each rotating-frame Hamiltonian separately.

Why this remains worth doing:

- Shared-slow CPU is already a measured `1.90x` win over repeated runs.
- Its remaining obvious overhead is the Python batch loop for rotating-frame assembly, diagonalization, and propagation.
- Improving this path helps small and medium scans where GPU does not yet dominate.

Potential implementation:

- Build `H_rot_batch` with shape `(B, n, n)` using broadcasting or `np.einsum`.
- Use batched `np.linalg.eigh(H_rot_batch)` where practical.
- Apply propagation with batched matmul.
- Keep a scalar fallback if the batched eigensolve is slower on a specific BLAS/LAPACK build.

Validation:

- Compare against current `run_microwave_scan(...)` for a small scan.
- Require max final probability difference near the existing repeated-run comparison level.

## Priority 7: Avoid Full Probability Matrices For Monitor-Only GPU Outputs

Current GPU monitor handling computes all final probabilities in several paths and then gathers monitor states:

- `src/state_prep_gpu/simulator.py:190`
- `src/state_prep_gpu/simulator.py:356`
- `src/state_prep_gpu/simulator.py:690`

Recommendation:

If `return_final_probabilities=False` and monitor states are requested, compute amplitudes directly against the monitored tracked eigenvectors. This reduces final probability work from `(B, S, n)` to `(B, S, K)`.

Why priority is moderate:

- It is likely a clean memory and final-step compute reduction.
- It probably will not solve the main real SPA2 GPU runtime by itself because the loop eigensolves dominate.

## Priority 8: Precompute Field And Microwave Time Series For CPU Runs

Current pattern:

- `src/state_prep/hamiltonians.py:35`: `H_R` calls electric and magnetic field callables.
- `src/state_prep/hamiltonians.py:42`: `get_H_t_func` wraps trajectory and field evaluation.
- `src/state_prep/simulator.py:930` and `src/state_prep/simulator.py:1071`: timestep loops call the Hamiltonian function.

Potential improvement:

- Add an optional precomputed-field path that builds `E_t` and `B_t` once.
- Evaluate slow Hamiltonians from field-independent matrices in a flat NumPy path.
- Precompute microwave scalar envelopes and polarization along the same time grid.

Why priority is moderate:

- It should reduce Python callable overhead.
- The storage-mode benchmark suggests eigensolves dominate in small cases, so this needs profiling before major refactoring.

## Priority 9: Cache Microwave Coupling Matrices

Current pattern:

- `MicrowaveField.generate_coupling_matrices(...)` builds x/y/z coupling matrices.
- `make_H_mu(...)` loops over state pairs and evaluates Wigner/three-j terms.

Relevant locations:

- `src/state_prep/microwaves.py:199`
- `src/state_prep/microwaves.py:331`
- `src/state_prep/microwaves.py:354-357`
- `src/state_prep/microwaves.py:409`
- `src/state_prep/microwaves.py:426`

Potential improvement:

- Cache by `(Jg, Je, QN signature, polarization basis)`.
- Reuse cached matrices across repeated notebook setup and scans.
- Consider sparse storage only if profiling confirms meaningful sparsity benefits.

Why priority is lower:

- This affects setup time more than timestep evolution.
- It can still help notebooks that repeatedly reconstruct the same microwave fields.

## Priority 10: Revisit GPU Buffer Reuse After Better Microbenchmarks

`slow_hamiltonian_batch(..., out=...)` exists, but the attempted microbenchmark timed out at multiple sizes. Do not treat buffer reuse as a proven optimization yet.

Better benchmark design:

- Use CUDA events instead of wall-clock timing around Python loops.
- Force one-time CuPy kernel compilation before timing.
- Print progress or keep the run short enough to avoid silent timeouts.
- Measure inside a full simulation loop, not only isolated Hamiltonian construction.

If this later measures well:

- Preallocate `H_slow` outside GPU timestep loops.
- Reuse buffers for `H_mu`, `H_rot`, `tmp`, and probability scratch arrays where CuPy operations permit it.

## Lower-Risk Cleanup With Performance Side Effects

- Remove unused imports such as `scipy.differentiate.derivative` in `microwaves.py` if confirmed unused.
- Fix field addition type checks in `ElectricField.__add__` and `MagneticField.__add__`; they currently assert `type(other) != ElectricField/MagneticField`, which appears inverted.
- Consider `scipy.linalg.eigh(..., overwrite_a=True, check_finite=False)` or direct LAPACK calls where inputs are trusted and profiling shows overhead.
- Add tests for near-degenerate eigenvector tracking, especially CPU vs GPU ordering behavior.

## Updated Suggested Plan

1. Update scan notebooks to use `run_microwave_scan(...)` and final-only or monitor-only storage by default.
2. Add parameterized benchmark scripts and store benchmark output as JSON/CSV.
3. Add a Windows GPU diagnostic and document/fix NVIDIA DLL path setup.
4. Sweep the real SPA2 GPU benchmark over batch size before changing GPU internals.
5. Vectorize the CPU shared-slow inner batch loop if profiling confirms the Python loop is material.
6. Add direct monitor-only probability calculation in GPU paths.
7. Profile field/microwave precomputation and coupling-matrix caching before implementing larger refactors.
8. Revisit GPU buffer reuse only after a CUDA-event benchmark gives a reliable result.

## Follow-Up Investigation: 2026-06-29

This follow-up focused on areas not fully separated in the first scan: CPU
shared-slow internals, the new process-parallel scan option, batched CPU
rotating-frame diagonalization, and current GPU benchmark usability.

Environment for these short runs:

- Python: `3.11.13`
- NumPy: `2.3.2`
- SciPy: `1.17.1`
- CuPy: not installed in the active `.venv`
- OS/shell: Windows / PowerShell

The environment differs from the earlier benchmark note, so the numbers below
should be treated as directional.

### CPU Shared-Slow Runtime Split

Targeted monkeypatch timing of `run_microwave_scan(...)` with `N_steps=200`,
batch `8`, `n=64`:

- Total runtime: `0.989 s`
- `zheevd` calls: `1791`
- `zheevd` time: `0.748 s` (`75.6%` of runtime)
- `reorder_evecs` calls: `199`
- `reorder_evecs` time: `0.019 s` (`1.9%` of runtime)

Interpretation:

- The current shared-slow CPU hot path is dominated by eigensolves, especially
  the per-scan-point rotating-frame diagonalization.
- Replacing the Hungarian eigenvector reordering is not a priority for CPU
  runtime. It may still matter for GPU correctness/performance, but it is not
  the observed CPU bottleneck here.

### `zheevd` vs NumPy `eigh`

Same setup, `N_steps=200`, batch `8`:

- `eig_backend="zheevd"`: `0.992 s`
- `eig_backend="numpy"`: `1.028 s`

Interpretation:

- The current `zheevd` default remains slightly faster than scalar
  `np.linalg.eigh` calls on this machine.

### Batched CPU Rotating-Frame Prototype

A temporary prototype assembled rotating-frame Hamiltonians as `(B,n,n)` and
used batched `np.linalg.eigh(...)` plus batched propagation.

`N_steps=200`, batch `8`:

- Current-style loop: `1.045 s`
- Batched prototype: `0.995 s`
- Max final probability difference: `1.58e-9`

`N_steps=200`, batch `16`:

- Current-style loop: `1.879 s`
- Batched prototype: `1.848 s`
- Max final probability difference: `6.49e-9`

Interpretation:

- CPU vectorization of the shared-slow inner loop appears numerically safe in
  this small prototype, but the speedup is only about `2-5%` here.
- This downgrades the earlier "vectorize CPU shared-slow inner loop" item from
  high-confidence performance work to a moderate/low priority unless another
  BLAS/LAPACK build shows larger batched-eigensolve gains.

### Process-Parallel CPU Scan Behavior

Short `bench_microwave_scan.py` runs:

`N_steps=120`, batch `4`:

- workers `1`: `0.381 s`
- workers `2`: `1.437 s` (`0.27x` vs serial)

`N_steps=120`, batch `16`:

- workers `1`: `1.087 s`
- workers `2`: `1.780 s` (`0.61x` vs serial)
- workers `4`: `1.718 s` (`0.63x` vs serial)

`N_steps=500`, batch `16`:

- workers `1`: `4.627 s`
- workers `2`: `3.725 s` (`1.24x` vs serial)
- workers `4`: `2.829 s` (`1.64x` vs serial)

Interpretation:

- The `loky` path has substantial startup/pickling overhead and hurts small
  scans.
- It can help once each chunk has enough work. For the tested shape, the
  crossover was between `N_steps=120` and `N_steps=500` at batch `16`.
- Keep `workers=1` as the safe default. Document `workers` as a tuning knob for
  large scans, and include a worker sweep in benchmark output before using it
  in notebooks or production scans.

### GPU Follow-Up Blocked In Current Environment

`bench_gpu_kernels.py` currently skips because CuPy is not installed in the
active `.venv`:

- Error: `ModuleNotFoundError('CuPy is required for state_prep_gpu execution...')`

One benchmark-script issue was also observed:

- The parent process accepts `--gpu-timeout`, but does not pass it to the worker
  subprocess. If the worker succeeds, the emitted payload reports the worker's
  default `gpu_timeout` instead of the requested parent value.

### Updated Priority Changes

1. Keep `workers=1` as the default for CPU microwave scans; benchmark worker
   counts before using process parallelism.
2. Treat rotating-frame eigensolves as the main CPU shared-slow bottleneck.
   Improvements need to reduce eigensolve count/cost, not just Python loop
   overhead.
3. Do not prioritize CPU batched `np.linalg.eigh` conversion unless repeated on
   another machine or BLAS build shows a larger win.
4. Fix `bench_gpu_kernels.py` timeout propagation and reinstall/enable CuPy
   before refreshing GPU conclusions.
5. Keep direct GPU monitor-only probability calculation as a clean memory/work
   reduction, but expect limited impact on the real SPA2 runtime unless final
   probability calculation becomes a measured bottleneck.
