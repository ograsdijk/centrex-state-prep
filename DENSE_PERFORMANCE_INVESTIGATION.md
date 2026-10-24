# Dense-path performance investigation — 2026-10-24

The best measured candidate is **batching the shared slow eigensolves across
timesteps on GPU**. A 128-step tile reduced runtime by 18.7% at 250 scan points
and 49.3% at 25 points in the varying-electric-field case, including tile
preparation. This is additional to the existing sharing across scan points.
Full cross-call caches delivered little further benefit in these GPU tests.

The four investigations below record benchmark prototypes. The recommended
bounded timestep tiles were subsequently integrated into the GPU shared-slow
scan; see `src/state_prep_gpu/README.md` and the production tile entry in
`PERFORMANCE_BENCHMARK_RESULTS.md`. The native adapters and cross-call caches
remain benchmark experiments. Earlier block-helper changes are separate and
do not affect these dense fallback paths.

## Scope and measurement

The existing SPA2 setup was tested with non-collinear fields: `transverse_b`
adds Bx = 0.3 Bz and By = 0.2 Bz, while `varying_e` adds small, varying x/y
components to the electric field. Their exact definitions are in
`benchmarks/profile_transverse_fields.py`. Both detectors fall back to dense
solves. These are synthetic general-path probes, not measured field maps.

All matrices are **complex128, 64×64**, and all solves compute full
eigensystems. Frozen midpoint propagation, units, and matching objectives were
preserved. For each field case, 300 actual rotating-frame matrices were
captured from 12 timestep samples × 25 detunings in a 300-step production CPU
scan. Solver screening used those matrices, not random Hermitian matrices.

Benchmarks used the same Ryzen 7 9800X3D / RTX 5070 Ti Windows environment as
`PERFORMANCE_BENCHMARK_RESULTS.md`: Python 3.11.13, NumPy 2.3.2, SciPy 1.17.1,
PyTorch 2.12.1+cu132. Every timing comparison used one warmup, five repeats,
rotating interleaved order, and mean ± sample standard deviation. Paired 95%
intervals are in the JSON reports. Some CPU timings were highly variable;
small effects are explicitly treated as unproven. The short step counts test
performance and same-step agreement, not physical timestep convergence.

## 1. Alternative dense eigensolvers

The current CPU `zheevd` driver remains a good choice. Representative solver
screening over the 300 transverse-B matrices, with one BLAS thread:

| CPU solver | Mean ± std | Finding |
| --- | ---: | --- |
| Current `zheevd` | 0.0971 ± 0.0055 s | Reference |
| NumPy `eigh`, upper triangle | 0.0974 ± 0.0040 s | Equivalent; no measured improvement |
| LAPACK QR `zheev` | 0.1704 ± 0.0028 s | 75.5% slower |
| LAPACK MRRR `zheevr` | 0.3248 ± 0.0058 s | 234.6% slower |

The varying-E matrices gave the same conclusion: 0.1053 ± 0.0022 s for
`zheevd`, 0.1859 ± 0.0083 s for QR, and 0.3453 ± 0.0265 s for MRRR. Rejected
drivers were not carried into full propagation. Residuals, orthogonality, and
eigenvalue agreement were checked independently of eigenvector phases.

GPU default, explicit cuSOLVER, and requested MAGMA gave no replacement win.
Although the installed build advertises MAGMA, it emitted a warning that
`linalg.eigh` dispatches to cuSOLVER instead. Thus this was not a comparison
against an independent working MAGMA implementation.

A genuinely different GPU driver, native cuSOLVER batched Jacobi, accepted the
64×64 complex inputs and passed convergence/residual/orthogonality checks, but
was slower. At batch 250:

| Matrices | PyTorch default mean ± std | Native Jacobi mean ± std |
| --- | ---: | ---: |
| Transverse B | 13.69 ± 1.20 ms | 65.32 ± 4.10 ms |
| Varying E | 13.40 ± 0.67 ms | 68.67 ± 2.55 ms |

Jacobi was also slower at batch 25. No precision reduction was attempted.

## 2. CPU matrix layout, copies, and workspace

The installed SciPy wrapper exposes workspace **sizes**, not reusable scratch
buffers. Its default complex workspace at n=64 is 4,224 entries, matching the
queried optimal size; supplying that size explicitly showed no benefit.

Two stronger prototypes were tested:

- Assemble each rotating-frame matrix in Fortran order and allow LAPACK to
  overwrite that disposable matrix, avoiding its extra input copy.
- Call SciPy's installed Cython LAPACK capsule using persistent complex, real,
  and integer scratch arrays. Outputs remain separately owned. This is a
  serial benchmark adapter, not a proposed public/native ABI dependency.

Neither established a useful full-propagation gain. For transverse B at
300 steps and 25 points:

| CPU propagation | Mean ± std | Assessment |
| --- | ---: | --- |
| Current path | 3.145 ± 0.078 s | Reference |
| Persistent LAPACK scratch | 3.187 ± 0.048 s | Difference unproven |
| Fortran assembly + overwrite | 3.323 ± 0.096 s | 5.7% slower |

The other tested shapes were 150/300 steps × 8/25 points. Some short cases
had severe timing variation; none established an improvement. Final
probabilities were identical in these full CPU comparisons. This rejects the
tested adapters, not the possibility of a future lower-overhead native loop.

## 3. GPU solver-call overhead

The matching PyTorch source already calls the newer batched cuSOLVER API when
the relevant build support is enabled; wrapping that same API alone is not a
new algorithm. See the [PyTorch 2.12.1 implementation](https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp#L1315)
and [NVIDIA's solver API documentation](https://docs.nvidia.com/cuda/cusolver/).

A prototype reused the native cuSOLVER handle, parameter object, and device/host
scratch, and deferred convergence-status checks until the end of propagation.
All statuses were retained and checked before accepting the result. It used
the current CUDA stream and a separate workspace per stream/shape.

| Steps | Points | Current GPU mean ± std | Prototype mean ± std | Assessment |
| ---: | ---: | ---: | ---: | --- |
| 150 | 25 | 1.041 ± 0.047 s | 0.941 ± 0.013 s | 9.6% less runtime |
| 150 | 250 | 2.737 ± 0.058 s | 2.756 ± 0.186 s | Unproven |
| 300 | 25 | 2.017 ± 0.021 s | 1.854 ± 0.011 s | 8.1% less runtime |
| 300 | 250 | 5.438 ± 0.034 s | 5.245 ± 0.030 s | 3.5% less runtime |

Final probabilities were identical. These gains combine workspace/dispatch
changes with deferred checking; they do not isolate a single cause. Solver-only
screening had not established a win from persistent workspace alone.

The CUDA trace explains the limited large-batch headroom. Three default solves
at batch 250 took 45.60 ms profiled wall time, with 43.51 ms of actual kernel
intervals. CPU `cudaStreamSynchronize` time was 42.78 ms, overlapping those
kernels: most of that wait is required computation, not removable overhead.
The trace has 123 kernel records for those three solves. GPU parent annotations
are explicitly excluded from kernel totals to avoid double-counting. These
short traces are diagnostics, not repeated speedup measurements.

Native adapters add maintenance, error-handling, platform, threading, and
autograd limitations. The measured 3.5% large-batch gain does not yet justify
putting this benchmark adapter into production. Benefits must not be added to
time-batching gains: they partly target the same slow-solve calls.

## 4. Reuse across separate scans, and a better per-call alternative

The package already shares slow eigensystems across scan points **within** a
call. It has no reusable slow-eigensystem cache across separate scan calls.
Existing precomputed field arrays and Hamiltonian terms are not cached
trajectory eigensystems.

The cache benchmark alternated independent intensity-prefactor scans of 0.5
and 1.0, keeping fields and timestep grids identical. A cache of raw slow
eigenvalues/vectors was prepared once; label tracking still ran sequentially.
Inputs were checked/fingerprinted, including term matrices, sampled fields,
precision/device for GPU, and timestep information. Changed-field/timestep
invalidation checks passed. CPU cache entries do not include t=0 because that
initial solve is not intercepted; GPU entries include it.

CPU cross-call reuse helped the small case, but not reliably the larger one:

| Steps | Points | Baseline mean ± std | Cached mean ± std | Cache build | Finding |
| ---: | ---: | ---: | ---: | ---: | --- |
| 150 | 8 | 0.546 ± 0.027 s | 0.486 ± 0.020 s | 0.087 s | 11.0% less runtime after preparation |
| 300 | 25 | 2.975 ± 0.081 s | 2.947 ± 0.136 s | 0.130 s | Unproven |

The cache build is excluded from cached timings and must be amortized. The
small-case mean saving would recover its measured build cost after roughly two
calls; the noisy larger case does not establish a useful break-even point.

GPU has another option: diagonalize the independent slow Hamiltonians for
many timesteps in one batch **inside each scan**. Only their eigensystems are
precomputed; state propagation and label tracking remain sequential. This
amortizes many tiny solve calls and needs no persistent cross-call cache.

| Fields | Steps | Points | Current mean ± std | Prebatch inside each call | Retained cross-call cache |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transverse B | 150 | 25 | 1.014 ± 0.007 s | 0.577 ± 0.042 s | 0.545 ± 0.007 s |
| Transverse B | 300 | 250 | 5.771 ± 0.178 s | 4.632 ± 0.025 s | 4.604 ± 0.046 s |
| Varying E | 150 | 25 | 1.128 ± 0.114 s | 0.610 ± 0.068 s | 0.570 ± 0.026 s |
| Varying E | 300 | 250 | 5.776 ± 0.289 s | 4.665 ± 0.188 s | 4.600 ± 0.126 s |

Per-call prebatch timings include precomputation every invocation. Cached
timings exclude the one-time cache build. Measured GPU cache builds were
0.190–0.200 s for the first 150-step cache and 0.025–0.026 s for the subsequent
300-step cache in the same process. No extra cache-versus-prebatch speedup is
claimed: their smaller differences are within the combined spreads.

Full precomputation retains about 19.8 MB at 300 steps and would retain
660.5 MB (630 MiB) at 10,000 steps for these 64×64 complex128 eigensystems,
excluding scratch/input matrices. A 128-step tile bounds the retained slow
eigensystems to about **8.45 MB**, plus input matrices and solver scratch.
The tiled prototype was measured on varying E:

| Steps | Points | Current mean ± std | 128-step tile mean ± std | Runtime reduction |
| ---: | ---: | ---: | ---: | ---: |
| 150 | 25 | 1.128 ± 0.114 s | 0.572 ± 0.039 s | 49.3% |
| 300 | 250 | 5.776 ± 0.289 s | 4.696 ± 0.213 s | 18.7% |

The paired 95% intervals for tiled runtime changes were [-0.712, -0.400] s
and [-1.198, -0.962] s. Both mean improvements also exceeded combined spreads.
Across GPU prebatch/cache/tile comparisons, maximum common-physical-basis
population difference was 5.32e-9, physical-fidelity deviation was 3.41e-13,
and normalization error was 2.08e-13. CPU cached probabilities were identical.
This verifies same-step agreement, not physical convergence of these short runs.

Caches here live in one process. Separate launcher invocations start fresh
processes; sharing a retained cache between them would require additional
worker lifetime or persistence support. This is another reason to prefer the
self-contained per-call tile as the next small production improvement.

## Recommendation and reproduction

Implement bounded GPU slow-eigensolve time tiles first, with propagation tests
for dense transverse fields, changing field direction, spectrum crossings,
initial-state semantics, and non-default streams. Keep the CPU driver and
allocation path as they are. Consider an explicit cross-call CPU cache only
for repeated small scans with identical slow physics. Keep native adapters in
benchmarks until a larger workload benefit justifies their complexity.

All raw outputs are ignored under `results/dense_investigation_20261024/`.
The investigation is reproducible with the existing project environment:

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python benchmarks/bench_dense_investigation.py --phase cpu_screen --capture-dir results/dense_new/matrices --output results/dense_new/cpu_screen.json
& $python benchmarks/bench_dense_investigation.py --phase gpu_screen --capture-dir results/dense_new/matrices --output results/dense_new/gpu_screen.json
& $python benchmarks/bench_dense_investigation.py --phase cpu_propagate --cases transverse_b --steps 150 300 --batches 8 25 --capture-dir results/dense_new/matrices --output results/dense_new/cpu_propagate.json
& $python benchmarks/bench_dense_investigation.py --phase gpu_propagate --cases transverse_b --steps 150 300 --batches 25 250 --capture-dir results/dense_new/matrices --output results/dense_new/gpu_propagate.json
& $python benchmarks/profile_dense_dispatch.py --matrices results/dense_new/matrices/transverse_b.npz --output-dir results/dense_new/dispatch
& $python benchmarks/bench_slow_cache.py --device cpu --shapes 150:8 300:25 --output results/dense_new/cpu_cache.json
& $python benchmarks/bench_slow_cache.py --device gpu --shapes 150:25 300:250 --field-case transverse_b --output results/dense_new/gpu_cache.json
& $python benchmarks/bench_slow_cache.py --device gpu --shapes 150:25 300:250 --field-case varying_e --output results/dense_new/gpu_cache_varying_e.json
```

Each runner defaults to one warmup/five repeats. GPU native adapters use the
installed Windows PyTorch cuSOLVER DLL and require no new dependency, compiler,
or installation. They are benchmark-only complex128 adapters, not portable
production solver APIs. CPU native scratch reuse uses the installed SciPy
Cython LAPACK capsule and is restricted to serial use in these tests.
