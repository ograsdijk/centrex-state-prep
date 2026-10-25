# Performance Benchmark Results

## RC J12 background: integrated CPU/GPU paths (2026-10-24)

A further frozen-method check rejected the installed PyTorch CPU backend:
2.637 ± 0.038 s for SciPy versus 2.924 ± 0.036 s per-step or 2.911 ± 0.092 s
time-batched on PyTorch CPU (300 samples, 25 RC points, one thread). A GPU
tile/workspace sweep at 1,000 samples measured 1.895 ± 0.027 s for default
8/256 versus 1.827 ± 0.018 s for 16/512: 3.6% less time, but peak allocation
rose from 515 MB to 961 MB. Time-32/1,024 was unproven. Defaults remain
8/256; optional launcher controls are `--gpu-time-chunk` and
`--gpu-max-eigh-matrices`. All measurements use one warmup and five
interleaved repeats, with numerical agreement checked. Details and raw reports
are in the follow-up section of [RC_PERFORMANCE.md](RC_PERFORMANCE.md).

RC-specific measurements and implementation details are in
[RC_PERFORMANCE.md](RC_PERFORMANCE.md). The setup includes a distinct RC carrier
with mixed `z + 0.4 y` polarization; these results are not transferred from
the no-beat velocity scan. Defaults preserve the CPU frozen integrator.

One warmup and five interleaved repeats used 300 spatial samples, 25 points
(five SPA detunings × five background powers), 64 states, four initial states,
complex128, and RC offsets 0/-7.4 MHz from the zero-field transition:

| RC offset | CPU Magnus before → cached, one worker | CPU frozen, four workers → tiled GPU frozen |
| --- | ---: | ---: |
| 0 MHz | 2.623 ± 0.035 → 1.673 ± 0.045 s (36.2% less runtime) | 1.423 ± 0.016 → 0.631 ± 0.038 s (55.7% less) |
| -7.4 MHz | 2.440 ± 0.174 → 1.587 ± 0.030 s (35.0% less runtime) | 1.354 ± 0.022 → 0.598 ± 0.005 s (55.9% less) |

Every claimed improvement passes paired 95% and combined-spread checks. CPU
Magnus reuse preserves the existing integrator, including both beat senses and
physical step-start timing. GPU support uses dense full eigensolves, real
per-velocity beat times, bounded time/scan tiles and ordinary PyTorch status
checks. It is selected automatically by the launcher for same-manifold tones.
GPU array preparation and CPU setup construction are outside these timings;
per-call grouping, scan preparation, tile construction and propagation are
inside. Warm CPU workers and GPU completion synchronization are included.

The full suite passed (282 tests). At 2,000 samples, 27 parameter/velocity
points and both offsets, GPU frozen probabilities agreed with a CPU subset
within 3.82e-9; the CPU Magnus cache matched its previous implementation in
that longer check. An RC CPU/GPU launcher smoke check agreed within 3.31e-9
and resumed its checkpoints. These are same-grid agreement checks, not RC
timestep-convergence certification. Source, full conditions, raw-report paths
and reproduction commands are in the linked report.

## Velocity follow-up: four optimization candidates (2026-10-24)

Benchmark-only experiments against current production (`0d6e981`) examined
grouped state propagation, group-preserving CPU worker splits, full microwave
eigensolve time tiles, and native GPU solver handle/workspace reuse. Full
method, all cases, intervals, limitations and reproduction commands are in
[VELOCITY_PERFORMANCE_INVESTIGATION.md](VELOCITY_PERFORMANCE_INVESTIGATION.md).
Production defaults were not changed.

The larger case used 300 spatial samples, 25 microwave parameter combinations
× 10 velocities, 64 states, four initial states, complex128, varying transverse
electric fields, and dense solves. One warmup and five interleaved repeats on
the same Ryzen 7 9800X3D / RTX 5070 Ti measured:

| Experiment | Current mean ± std | Candidate mean ± std | Assessment |
| --- | ---: | ---: | --- |
| CPU grouped propagation, one worker | 4.804 ± 0.204 s | 3.947 ± 0.179 s | 17.8% less runtime |
| CPU group-preserving partition, four workers | 3.629 ± 0.028 s | 1.615 ± 0.014 s | 55.5% less runtime |
| GPU grouped propagation alone | 1.091 ± 0.038 s | 1.062 ± 0.023 s | Unproven |
| GPU full eigensolve time tile 8 | 1.091 ± 0.038 s | 0.662 ± 0.033 s | 39.3% less runtime |
| GPU native workspace/handles, checked status | 1.091 ± 0.038 s | 1.084 ± 0.027 s | Unproven |
| GPU native workspace/handles, deferred status | 1.091 ± 0.038 s | 1.033 ± 0.021 s | Unproven by combined-spread rule |

The CPU worker gain depends on velocity-major input scattering identical
microwave rows across processes; parameter-major input may already retain
most reuse. Worker timing uses warmed reusable processes. All non-native
improvements claimed above pass both paired 95% and combined-spread checks.

In the smaller 150-sample, five-parameter × eight-velocity case, full time-8
GPU tiles reduced runtime from 0.442 ± 0.014 s to 0.167 ± 0.005 s (62.3%).
Native deferred status checks helped that case by 15%, but did not provide
a proven extra win after time batching. A separate paired comparison of
time-8 versus time-8 plus grouped updates measured 0.704 ± 0.033 s versus
0.647 ± 0.021 s on the larger case: another 8.1% reduction. The smaller
combined case remained unproven. These percentages are not additive.

All repeated numerical comparisons passed. Twelve regression tests for the
prototypes passed, covering CPU ragged groups and worker ordering plus
CPU/CUDA histories, monitors and time-tile boundaries. Longer 1,000-sample,
250-point GPU checks matched current GPU probabilities exactly in a common
basis and a CPU subset within 2.13e-9; no speed claim uses that single check.

Raw repeated reports are ignored files
`results/velocity_followup_20261024.json` and
`results/velocity_followup_combined_20261024.json`. The experiments support
prioritizing CPU worker locality and bounded full GPU time tiling. Native
solver integration has no demonstrated benefit on top of the latter.

## Shared spatial-grid velocity scans (2026-10-24)

CPU `run_microwave_scan` and the GPU shared-slow scalar-envelope kernel now
accept per-point forward velocities. They sample a common spatial path and
scale each timestep by reference speed / speed. Identical microwave parameter
rows share the full frozen-Hamiltonian eigensolve, as well as the slow solve.
Distinct powers/detunings retain separate full solves. GPU grouping also checks
the complete per-point envelope and preserves the precision of shift inputs.
The CPU multitone route shares only slow solves and evaluates beat phases at
each point's physical elapsed time, including Magnus kernels when selected.
The scalar GPU route's existing lack of same-manifold beat support is unchanged.

One warmup and five interleaved repeats compared eight independent velocity
runs against one ensemble: five detuning combinations × eight speeds from
140 to 240 m/s, four initial states, 300 spatial samples, 64 states, complex128,
and the varying-E dense-Hamiltonian setup. Exact blocks were disabled on both
sides. CPU used one worker with one BLAS thread; GPU used one PyTorch CPU
thread. Environment: Ryzen 7 9800X3D / RTX 5070 Ti, Python 3.11.13,
NumPy 2.3.2, SciPy 1.17.1, PyTorch 2.12.1+cu132.

| Backend | Separate velocity runs, mean ± std | Shared ensemble, mean ± std | Speedup | Runtime reduction |
| --- | ---: | ---: | ---: | ---: |
| CPU | 6.588 ± 0.039 s | 1.057 ± 0.017 s | 6.23× | 83.9% |
| GPU | 6.954 ± 0.125 s | 0.880 ± 0.005 s | 7.90× | 87.3% |

Paired 95% intervals for ensemble minus independent runtime were
[-5.578, -5.484] s (CPU) and [-6.231, -5.918] s (GPU); both mean improvements
exceeded the summed baseline/candidate standard deviations. CPU setup objects
and GPU input preparation were outside timings. CPU scan preparation inside
`run_microwave_scan`, per-call GPU grouping, propagation, and output assembly
were included. GPU timings synchronized completion. These are ensemble
throughput gains relative to separate velocity runs, not gains for a single
velocity or for scans whose full Hamiltonians differ between velocities.

Common-basis final probability differences against independent trajectories
were at most 5.55e-9 on CPU and zero on GPU; CPU versus GPU differed by
1.09e-8. The GPU baseline used the existing per-point timestep capability
with one velocity per call. The CPU baseline used separately scaled physical
trajectories. The shared and separate runs used the same spatial samples;
this timing comparison does not establish timestep convergence at 300 samples.

Reproduce with a fresh output file:

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python benchmarks/bench_velocity_scan.py --n-steps 300 --velocities 8 --parameter-points 5 --warmup 1 --repeat 5 --output results/velocity_scan_new.json
```

Raw repeated timings are retained in ignored
`results/velocity_scan_20261024.json`. Regression tests compare separate
trajectories, frozen/Magnus propagation, multitone beat times, exact blocks,
CPU/CUDA tensor execution, histories/monitors, differing envelopes, grouping
precision, and parallel-worker ordering. Velocity arrays are also accepted by
the CPU/GPU launcher's parameter file and included in checkpoint fingerprints.

Validation completed with the full suite (233 passed), then 68 focused
CPU/CUDA tests after adding shift-key precision and transverse-path coverage.
The standard CPU/GPU comparison retained a maximum state difference of
1.95e-15. A mixed CPU/GPU launcher smoke run with two CPU workers agreed with
a CPU reference within 2.61e-9, retained velocity/time metadata, resumed matching
checkpoints, and rejected changed velocities without overwriting prior inputs.
The historical-baseline GPU tile benchmark also passed a functional smoke
check after the runner API extension; its single-run timing is not a speed claim.

## Production GPU slow-eigensolve tiles (2026-10-24)

The shared-slow GPU scan now batches independent slow eigensolves across
timesteps in bounded tiles. `slow_eigh_chunk_size=None` selects 128 on CUDA
and 1 on CPU tensors; `1` selects per-step solves explicitly. Tracking,
propagation, monitors, and history remain sequential. The true initial solve
is separate. Exact sector labels are retained per timestep and rotating-frame
groups are recomputed after every sorted spectrum, including tile boundaries.
Dense transverse fields require no symmetry assumption. Only the current tile
is retained, about 8.45 MB of eigensystems at 64 states/complex128, plus input
matrices, optional sector labels, and solver workspace.

Production comparisons load the pre-tile simulator from Git revision `b02b546`.
Both variants import the same current block and assignment helpers, isolating
the timestep change from the earlier helper optimizations. Runner preparation
is excluded; support detection and all tile construction/solves inside each
propagation are included. Four initial states and complex128 were used on the
same RTX 5070 Ti environment as the investigation below.

| Fields | Steps | Points | Repeats | Pre-tile mean ± std | Production tiles mean ± std | Assessment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Transverse B | 300 | 250 | 5 | 5.842 ± 0.212 s | 4.709 ± 0.130 s | 19.4% less runtime |
| Varying E | 150 | 25 | 7 | 1.237 ± 0.045 s | 0.626 ± 0.036 s | 49.4% less runtime |
| Varying E | 300 | 250 | 7 | 5.808 ± 0.125 s | 4.580 ± 0.066 s | 21.1% less runtime |
| Aligned | 150 | 25 | 7 | 2.987 ± 1.029 s | 1.723 ± 0.540 s | Unproven by combined-spread rule |
| Aligned | 300 | 250 | 7 | 6.054 ± 4.852 s | 3.068 ± 1.746 s | Unproven by combined-spread rule |

All comparisons used one warmup, interleaved rounds, and sample standard
deviations. The first five-round dataset suffered severe pauses, including
aligned-case measurements from 2.6 to 21.9 s, and a varying-E outlier. It is
retained as `results/gpu_slow_tiles_production_20261024.json`. Aligned and
varying-E cases were then repeated with seven rounds; no individual samples
were removed. The retry is `results/gpu_slow_tiles_retry_20261024.json`.
Aligned-case variability persisted, so no precise aligned-path speedup is
claimed. In the varying-E retry, paired 95% intervals for runtime changes were
[-0.632, -0.590] s at 25 points and [-1.356, -1.101] s at 250 points. The
transverse-B 250-point interval was [-1.531, -0.734] s.

All initial production cases agreed with the pre-tile path within 5.49e-9 in a
common physical readout basis and with a seven-point CPU reference subset
within 5.15e-9. The complete regression suite passed. New tests cover CPU/CUDA,
complex64/complex128, fully connected late transverse fields, exact blocks and
sector crossings, per-step/per-point nonuniform dt, per-point envelopes, distinct
initial fields, monitors, history, the partial tile, default selection, invalid
tile sizes, and a non-default CUDA stream. A bounded-sampling test confirms
tile widths of 128,128,3 for 259 propagated samples, excluding the terminal
sample. The existing adiabatic-label API warning remains expected.

A separate 1,000-step/250-point varying-E check agreed with the pre-tile path
within 3.74e-9 and the CPU subset within 2.67e-9 in common-basis populations;
physical-fidelity deviation from the pre-tile result was 8.76e-13. Its single
run is numerical validation only, not a speedup measurement, and is saved as
`results/gpu_slow_tiles_long_agreement_20261024.json`. The CPU/GPU comparison
example also passed (maximum common-basis probability difference 1.89e-15).

`slow_eigh` profiling now includes tile construction and solve time, amortized
over covered timesteps and weighting the partial tail correctly. All tiles are
profiled when tiling is active; other stages retain their sampling stride.
Historical solver/cache prototypes explicitly request chunk size 1 so their
baselines remain per-step after the production default changes.

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python benchmarks/bench_gpu_slow_tiles.py --cases aligned transverse_b varying_e --shapes 150:25 300:250 --warmup 1 --repeat 5 --output results/gpu_slow_tiles_new.json
& $python benchmarks/bench_gpu_slow_tiles.py --cases aligned varying_e --shapes 150:25 300:250 --warmup 1 --repeat 7 --output results/gpu_slow_tiles_retry_new.json
```

## Dense solver, overhead, and reuse experiments (2026-10-24)

All four follow-up investigations are documented in
[DENSE_PERFORMANCE_INVESTIGATION.md](DENSE_PERFORMANCE_INVESTIGATION.md).
Alternative CPU/GPU dense solvers did not beat the current drivers. CPU layout
and scratch reuse did not establish a gain. A native GPU dispatch/scratch
prototype saved 3.5% at 300 steps/250 points, while a 128-step slow-eigensolve
tile saved 18.7% on the dense varying-E path (5.776 ± 0.289 s to
4.696 ± 0.213 s), including per-call tile preparation. At 150 steps/25 points,
the same tile saved 49.3% (1.128 ± 0.114 s to 0.572 ± 0.039 s).
These paired results use one warmup/five repeats and agree in physical-basis
populations within 5.32e-9. Full cross-call caching delivered little further
GPU benefit beyond per-call prebatching; CPU cache benefit was limited to the
smaller tested scan. These are benchmark prototypes, not production API changes
or timestep-convergence claims. The detailed report records rejected options,
cache build cost, memory, invalidation checks, CUDA traces, and limitations.

## Non-collinear field profiles (2026-10-24)

The block-overhead improvements below were checked against synthetic
non-collinear variants of the same SPA2 benchmark. The reference microwaves,
carriers, initial states, and detunings were retained. Two field variants were
defined in the existing XYZ field coordinates and existing field units:

- `transverse_b`: keep Bz and add Bx = 0.3 Bz, By = 0.2 Bz; electric field stays
  on z. Electric and magnetic fields are therefore non-collinear.
- `varying_e`: keep Ez and add Ex = 0.02 Ez sin(phi), Ey = 0.01 Ez cos(phi),
  with phi = 2 pi (z + 0.08) / 0.28 along the original trajectory in meters.
  The magnetic field stays on z and the electric-field direction varies.

These are synthetic general-path probes, not measured experimental field maps.
Both CPU and GPU detectors returned no usable blocks for both variants and
executed dense solves. The aligned reference retained the nine SPA2 blocks.
Thus the earlier block-helper changes do not accelerate these dense paths.

At 300 steps, four initial states, one warmup and five repeats, uninstrumented
timings were as follows. CPU used one worker and 42 scan points, representative
of a six-worker chunk; GPU used 250 scan points. The CPU points were an evenly
spaced subset of the GPU grid, permitting physical output comparisons. These
columns have different batch sizes and are not a CPU-versus-GPU speed comparison.
Spreads here are population standard deviations (`ddof=0`, recorded in JSON).

| Fields | Usable blocks on both devices | CPU mean ± std, 42 points | GPU mean ± std, 250 points |
| --- | --- | ---: | ---: |
| Aligned reference | 12,14,12,8,8,4,4,1,1 | 1.761 ± 0.044 s | 2.445 ± 0.080 s |
| Transverse B | None | 4.439 ± 0.123 s | 5.611 ± 0.018 s |
| Varying E direction | None | 4.271 ± 0.074 s | 5.754 ± 0.221 s |

CPU timings include scan preparation. GPU timings exclude runner preparation
but include support detection inside propagation. These compare different
physical workloads; they are not an interleaved optimization speedup claim.

### Dense-path bottlenecks

Direct wrappers around CPU LAPACK calls separated native solve time from the
surrounding Python evolution loop, where `cProfile` otherwise hides it. A
separate warmed diagnostic run counted exactly 299 shared slow solves and
12,558 rotating-frame solves per case (one slow solve plus 42 point solves per
propagated step):

| CPU stage | Transverse B, profiled wall 4.833 s | Varying E, profiled wall 4.745 s |
| --- | ---: | ---: |
| Shared slow eigensolves | 0.096 s (2.0%) | 0.095 s (2.0%) |
| Rotating-frame eigensolves | 4.000 s (82.8%) | 3.923 s (82.7%) |

Sampled CUDA-event profiles gave:

| GPU stage | Transverse B, profiled wall 5.685 s | Varying E, profiled wall 5.556 s |
| --- | ---: | ---: |
| Shared slow eigensolves | 0.965 s (17.0%) | 0.930 s (16.7%) |
| Batched rotating-frame eigensolves | 4.211 s (74.1%) | 4.107 s (73.9%) |
| Eigenvector tracking | 0.163 s (2.9%) | 0.170 s (3.1%) |
| Microwave rotate/assembly | 0.120 s (2.1%) | 0.118 s (2.1%) |
| State update | 0.208 s (3.7%) | 0.206 s (3.7%) |

CUDA events sample every 25 steps (12 samples here) and include solver-call
launch/synchronization delays within each region. They do not distinguish
GPU arithmetic from idle/dispatch time. Native CPU wrappers and profiling also
add overhead, so these shares are diagnostics, not replacement benchmark times.

The main general-path target is therefore the dense rotating-frame solve on
both devices. Optimizing only shared slow work has a limited measured ceiling
for CPU chunks; tracking/field assembly also cannot explain most runtime.
The next solver experiment should use these genuinely complex 64x64 dense
matrices, benchmark alternatives end to end, and check full physical states.
No general dense-path speedup has been demonstrated by this investigation.

CPU/GPU probabilities in one fixed physical readout basis agreed within
9.12e-9 across all three cases; maximum physical state-fidelity deviation was
2.02e-13. No simulator source was changed for these profiles.

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python benchmarks/profile_transverse_fields.py --steps 300 --cpu-batch 42 --gpu-batch 250 --warmup 1 --repeat 5 --output-dir results/transverse_profile_new
```

Raw JSON, CPU cProfile dumps, and readable profiles are under the ignored
`results/transverse_profile_20261024/` directory. The native-call diagnostic
profiles are included in its `profile.json`.

## Block-solver bottlenecks and local fixes (2026-10-24)

Profiles of the merged `b02b546` production paths located the cost in block
eigensolves and their overhead. At 300 steps and 250 points, CPU `cProfile`
attributed 83% of its instrumented runtime to `_eigh_in_blocks`, including 37%
in `np.ix_`. At 42 points (representative of one six-worker chunk), the
corresponding shares were 77% and 34%. These are diagnostic profiler shares:
`cProfile` inflates frequent Python calls, so they are not estimates of the
speedup available in an uninstrumented run.

A warmed CUDA-event profile at 300 steps and 250 points estimated 1.18 s in
shared slow eigensolves and 1.65 s in rotating-frame eigensolves, against a
3.28 s profiled wall time. Together those stages account for about 86%; slow
eigenvector assignment took about 0.14 s (4%). These event regions include
launch/synchronization overhead and GPU idle time within a region, so this
does not establish that eigensolver arithmetic alone occupies 86% of the GPU.

Two local helper changes address that overhead without changing the numerical
method, blocks, propagation, tracking objective, or public API:

- CPU: use direct integer-array indexing and output column slices instead of
  rebuilding `np.ix_` arrays and `arange` columns for every block and point.
- GPU: batch equal-sized exact sectors into one eigensolver call, and handle
  1×1 blocks directly. The SPA2 layout needs four solver calls instead of nine
  in each block solve. Sector indices are still recomputed after every slow
  spectrum sort, and original sector labels are preserved.

Baseline and candidate were measured in one process/session, interleaved with
one warmup and five repeats per case, on the same Ryzen 7 9800X3D / RTX 5070 Ti
environment as below. Spreads are sample standard deviations. CPU measurements
use one worker and include scan preparation; GPU measurements exclude runner
preparation but include support discovery inside propagation. They measure
runtime, not timestep accuracy or full independent-job startup costs.

| Device | Steps | Points | Baseline mean ± std | Candidate mean ± std | Runtime reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 500 | 25 | 2.634 ± 0.169 s | 2.242 ± 0.038 s | 14.9% |
| CPU | 500 | 42 | 4.008 ± 0.130 s | 3.202 ± 0.096 s | 20.1% |
| CPU | 1000 | 25 | 5.030 ± 0.154 s | 4.156 ± 0.142 s | 17.4% |
| CPU | 1000 | 42 | 7.636 ± 0.492 s | 6.189 ± 0.461 s | 19.0% |
| GPU | 500 | 250 | 5.461 ± 0.182 s | 4.193 ± 0.250 s | 23.2% |
| GPU | 500 | 500 | 7.065 ± 0.030 s | 5.704 ± 0.004 s | 19.3% |
| GPU | 1000 | 250 | 11.140 ± 0.412 s | 8.195 ± 0.050 s | 26.4% |
| GPU | 1000 | 500 | 14.162 ± 0.268 s | 11.416 ± 0.152 s | 19.4% |

All paired 95% intervals excluded zero and all mean improvements exceeded the
summed baseline/candidate standard deviations. All final probabilities were
identical to baseline; maximum deviation of physical state fidelity from one
was 6.90e-13. This is evidence for these SPA2 workloads, not a guarantee for
dense fallbacks, other block layouts, or multitone Magnus.

Reproduce using the merged baseline revision:

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python benchmarks/bench_block_overhead.py --device cpu --steps 500 1000 --batches 25 42 --baseline-ref b02b546 --warmup 1 --repeat 5 --output results/cpu_block_overhead_new.json
& $python benchmarks/bench_block_overhead.py --device gpu --steps 500 1000 --batches 250 500 --baseline-ref b02b546 --warmup 1 --repeat 5 --output results/gpu_block_overhead_new.json
```

Raw reports and diagnostic profiles are retained under the ignored directory
`results/bottlenecks_20261024/`.

After the change, a separate warmed 300-step/250-point CUDA-event profile
estimated 0.70 s in slow eigensolves (27% of its 2.55 s wall time), 1.36 s in
rotating-frame eigensolves (53%), 0.16 s in tracking (6%), 0.12 s in microwave
assembly (5%), and 0.19 s in state updates (8%). Thus eigensolves remain the
main measured target. These sampled profiles diagnose stages; the repeated
paired measurements above establish the speedup. Further work should measure
solver/launch costs within those stages before introducing a new solver or
precomputation scheme.

Validation: the complete pytest suite passed, including new complex-matrix
reconstruction/sector-label tests on CPU and CUDA with repeated block sizes,
non-contiguous indices, 1×1 sectors, multiple batch dimensions, and both GPU
complex precisions. The CPU/GPU comparison example passed with maximum
common-basis probability difference 1.89e-15. One existing adiabatic-label
warning was emitted by the API test. No timestep or physics change was made.

## Independent CPU/GPU job pilot (2026-10-24)

Two separate SPA2 simulations used 250 detuning points each from -2 to +1 MHz,
four initial states, and intensity prefactors of 0.5 and 1.0 respectively for
both microwave fields. The half-power job ran on six CPU workers (one BLAS
thread each); the full-power job ran on the GPU with one PyTorch CPU thread.
Each had its own parameter file, launcher process, checkpoints, and results.
This is independent-job concurrency, rather than partitioning one scan.

On the Ryzen 7 9800X3D / RTX 5070 Ti environment documented below, at 1,000
steps, one warmup and five interleaved repeats measured the total time to finish
both jobs, including startup, preparation, output, and validation:

| Execution | Mean ± sample std | Total scan points/s |
| --- | ---: | ---: |
| CPU job followed by GPU job | 34.599 ± 1.011 s | 14.45 |
| CPU and GPU jobs simultaneously | 20.334 ± 1.351 s | 24.59 |

Concurrency reduced wall time by 41.2% (1.702x throughput). All five rounds
favored concurrency; the paired difference was -14.266 ± 0.304 s (standard
error), with 95% interval [-15.109, -13.422] s. This comparison uses the same
six CPU workers in both variants. It does not compare against assigning both
jobs to CPU only, or establish the optimum worker count or production speedup.

Final states were projected into one fixed physical readout basis rather than
compared by tracked labels. Sequential and concurrent final probabilities were
identical in the final round. Every run passed the agreement checks. A separate
CPU execution of the full-power job differed from GPU by at most 3.90e-9 in
probability. The largest state normalization error across all pilot jobs was
2.40e-12.

### Timestep accuracy remains a separate constraint

GPU convergence runs covered both full 250-point grids using uniform midpoint
frozen propagation. Maximum absolute changes in individual common-basis
populations between successive refinements were:

| Refinement | Half power | Full power |
| --- | ---: | ---: |
| 1,000 → 2,000 | 0.01464 | 0.02109 |
| 2,000 → 4,000 | 0.00790 | 0.01212 |
| 4,000 → 8,000 | 0.00169 | 0.00264 |

For 4,000 → 8,000, maximum changes in physical J=2 manifold populations were
0.00174 and 0.00262. Lineshape alignment found zero shift at its 0.1 kHz search
resolution for that pair; worst RMS residuals across initial states were
5.08e-4 and 6.92e-4. Alignment here is a diagnostic, not permission to discard
errors at the requested physical detunings.

The 1,000-step timing workload is not a precision production configuration.
The 8,000-step results also do not yet certify 0.001 absolute population
accuracy; finer references are needed before making that claim. No larger
production scan was launched from this pilot. These conclusions apply to the
benchmark SPA2 model, not the distinct-carrier multitone/Magnus workflows.

Reproduce with a fresh output directory:

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python benchmarks/bench_independent_jobs.py --steps 1000 --batch 250 --workers 6 --warmup 1 --repeat 5 --convergence-steps 1000 2000 4000 8000 --output-dir results/independent_pilot_new
```

Raw reports and checkpoints from this session are in the ignored directory
`results/independent_pilot_20261024/`, including `comparison.json`,
`convergence.json`, and per-job launcher logs. The runner refuses existing
output directories to prevent checkpoint reuse from becoming a false speedup.

## Local CPU/GPU updates (2026-10-24)

Measured on Windows with an AMD Ryzen 7 9800X3D (8 cores/16 threads), NVIDIA
RTX 5070 Ti, Python 3.11.13, NumPy 2.3.2, SciPy 1.17.1, and PyTorch
2.12.1+cu132. All tables below use one warmup and five interleaved timed rounds;
spreads are sample standard deviations. Other desktop workloads were active,
so small changes are treated conservatively. Commands and operational details
are in `PERFORMANCE_QUICKSTART.md`.

### CPU local changes

Compared revision `315af72` with the local candidate in the same interpreter,
using one worker, pinned BLAS, four initial states, exact blocks, final-only
output, and the real SPA2 fields. Magnus cases add the RC background at
-7.4 MHz. Both paths returned identical final probabilities in every case.
The changes hoist block grouping out of the point loop and omit unused frozen
assembly in multitone Magnus.

| Steps | Points | Propagator | Original mean ± std | Candidate mean ± std | Assessment |
|---:|---:|---|---:|---:|---|
| 500 | 5 | frozen | 0.793 ± 0.077 s | 0.737 ± 0.044 s | Unproven |
| 500 | 25 | frozen | 3.050 ± 0.066 s | 2.796 ± 0.062 s | 1.091x faster |
| 1000 | 5 | frozen | 1.782 ± 0.220 s | 1.746 ± 0.246 s | Unproven |
| 1000 | 25 | frozen | 4.707 ± 0.159 s | 4.282 ± 0.067 s | 1.099x faster |
| 500 | 5 | Magnus | 1.307 ± 0.155 s | 1.246 ± 0.085 s | Unproven |
| 500 | 25 | Magnus | 3.606 ± 0.413 s | 3.387 ± 0.415 s | Unproven |
| 1000 | 5 | Magnus | 2.784 ± 0.047 s | 2.675 ± 0.037 s | 1.041x faster |
| 1000 | 25 | Magnus | 7.832 ± 0.594 s | 7.187 ± 0.436 s | Unproven by combined-spread rule* |

*The paired confidence interval excludes zero, but the difference is smaller
than the summed standard deviations. Do not promote the nominal 1.09x ratio
to a robust speedup for this Magnus case.

Raw report: `results/cpu_performance_updates.json`.

### Production GPU exact blocks, with optimal assignment

Both variants use the corrected one-to-one matching and complex phase
alignment; these are not comparisons against the old greedy tracker. Runner
input preparation is excluded, while automatic block discovery inside each
simulation is included. All cases use four initial states and final-only output.

| Steps | Points | Dense mean ± std | Blocks mean ± std | Result |
|---:|---:|---:|---:|---|
| 500 | 25 | 3.435 ± 0.200 s | 4.756 ± 0.492 s | Blocks 38.5% slower |
| 1000 | 25 | 6.675 ± 0.076 s | 9.156 ± 0.593 s | Blocks 37.2% slower |
| 500 | 250 | 8.895 ± 0.249 s | 5.323 ± 0.188 s | Blocks 1.671x faster |
| 1000 | 250 | 17.802 ± 0.168 s | 11.190 ± 0.403 s | Blocks 1.591x faster |

Maximum dense/block probability difference was `1.495e-6`. Against the CPU
reference, the block GPU results differed by at most `5.77e-9` in probabilities
and `3.18e-13` in absolute deviation of state fidelity from one. Blocks are
therefore enabled automatically only at batches of at least 250; smaller batches
remain dense unless explicitly forced. This threshold is a conservative policy
for the measured 64-state SPA2 workload, not a universal crossover estimate.

Raw report: `results/gpu_performance_updates.json`.

### Device assignment versus SciPy round-trips

At 500 steps and 25 points, using dense production GPU propagation and optimal
matching in both variants:

| Assignment | Mean ± std |
|---|---:|
| On-device CUDA, float64 costs | 3.261 ± 0.029 s |
| SciPy with device/host/device transfers | 3.337 ± 0.119 s |

Final probabilities matched exactly. The paired 95% interval for CUDA minus
SciPy was `[-0.254, +0.101] s`, so a runtime difference is **unproven**. These
measurements do not support claiming that the transfers dominate this workload.
Production uses the device implementation, including full optimal conflict
resolution and phase alignment; there is no per-step CPU fallback.

Raw report: `results/gpu_assignment.json`.

### Concurrent CPU/GPU throughput

At 1,000 steps, 1,000 detunings, four initial states, uniform midpoint frozen
propagation, and final-only storage, fresh processes ran each variant with all
startup, preparation, checkpoint writing, merging, and shutdown included.
The hybrid assigned the first 600 scan points to six CPU workers and the
remaining 400 to the GPU (automatic exact blocks). GPU PyTorch CPU threads were
limited to one. One warmup and five interleaved repeats gave:

| Execution | End-to-end mean ± std |
|---|---:|
| CPU only, eight workers | 36.956 ± 3.257 s |
| CPU only, six workers | 42.507 ± 0.862 s |
| Six CPU workers + GPU, 60/40 point split | 31.314 ± 1.735 s |

Compared with the faster tested CPU-only setting, concurrent execution was
**1.180x faster (15.3% less wall time)**. The paired runtime difference was
`-5.642 ± 1.436 s` (standard error), with 95% interval `[-9.629, -1.656] s`.
Every timed round favored concurrent execution, and the mean difference also
exceeded the summed standard deviations. Maximum final-probability difference
was `4.569e-9`, after projecting physical states into the common CPU readout basis.

This establishes a useful concurrent setting for this workload; it does not
establish a globally optimal worker count or split. The GPU finished its chunk
before the CPU in the warmup, so a more GPU-heavy split may improve balance but
has not been timed with repeats. Smaller jobs may lose to process startup.

Raw summary and per-round timings: `results/concurrent_comparison/comparison.json`.
Individual backend checkpoints also record preparation and propagation times.

> **Historical results in the sections below are single-shot.** The repeated
> SPA2 comparison recorded later in this document is newer; read
> `IMPROVEMENTS.md` for other re-measured figures.
>
> These were recorded before the paired benchmark runner existed, so none carry
> a spread and several cannot be distinguished from no change at all. The
> `1.06x` real-SPA2 GPU result in particular has never been shown to differ
> from unity. Later runs with repeats and interleaving are in `IMPROVEMENTS.md`,
> together with the environment they were taken in.

Generated: 2026-06-14

This file records benchmarks run after the performance scan in `PERFORMANCE_IMPROVEMENTS.md`.

## Environment

- OS/shell: Windows / PowerShell
- Python: 3.11.13 from `.venv`
- NumPy: 2.4.0
- SciPy: 1.16.3
- CuPy: 13.6.0
- GPU: NVIDIA GeForce RTX 5070 Ti, driver 591.86, 16303 MiB
- CPU model: not recorded; `Get-CimInstance Win32_Processor` returned access denied.

GPU benchmark note: CuPy could see the GPU, but NVRTC initially failed to load `nvrtc-builtins64_129.dll`. The DLL exists under `.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin`, so GPU benchmark commands were run with the NVIDIA pip-package `bin` directories prepended to `PATH`.

## Summary

| Benchmark | Result |
| --- | ---: |
| Synthetic slow-H CPU vs GPU | GPU 3.46x faster |
| Real SPA2-like shared-slow CPU vs GPU | GPU 1.06x faster |
| CPU output storage: full vs final-only | final-only 1.17x faster, much less memory |
| CPU output storage: full vs downsampled every 20 | downsampled 1.17x faster, 20x less stored output |
| CPU microwave scan: shared-slow vs repeated runs | shared-slow 1.90x faster |
| GPU `slow_hamiltonian_batch(out=...)` microbenchmark | inconclusive; timed out repeatedly |
| Real SPA2, GPU vs 10-worker CPU (`N_steps=1000`, `B=250`) | baseline GPU 10.5x slower; refactored GPU 5.4x slower |
| Real SPA2, direct cuSOLVER Jacobi at `tol=1e-12` vs CuPy `eigh` (`N_steps=1000`, `B=250`) | custom GPU path 1.056x faster; close final-state agreement |

## Repeated SPA2 GPU Batch Scaling and 10-Worker CPU Comparison (2026-10-19)

This is a newer repeated benchmark than the historical single-shot results
below. It uses the real SPA2 setup (`n=64`), all four configured initial
states, a combined detuning × power scan, midpoint sampling, and final-only
output. The machine has an NVIDIA GeForce RTX 5070 Ti (16 GiB). CuPy 14.2.0,
NumPy 2.4.6, and SciPy 1.17.1 were used. The CPU scan used 10 `loky` workers
with one BLAS thread per worker. CPU and GPU measurements were interleaved,
with one warmup and five timed rounds.

The GPU-only batch sweep at 120 steps showed increasing throughput as the
batch grew. This was a GPU scaling check, not a comparison against a 10-worker
CPU:

| Scan points | GPU mean ± std | Throughput |
|---:|---:|---:|
| 25 | 1.594 ± 0.075 s | 15.7 points/s |
| 50 | 2.303 ± 0.097 s | 21.7 points/s |
| 100 | 3.784 ± 0.096 s | 26.4 points/s |
| 250 | 8.029 ± 0.267 s | 31.1 points/s |

At 1,000 steps, the completed pre-refactor paired case (`B=250`) strongly
favored the 10-worker CPU. The pre-refactor `B=500` case was stopped after
three of five rounds at the user's request; treat it as partial. The `B=1000`
case was not run.

| Round | CPU, 10 workers | GPU |
|---:|---:|---:|
| 1 | 12.6975 s | 116.6586 s |
| 2 | 11.8295 s | 148.5422 s |
| 3 | 11.9480 s | 142.8621 s |
| 4 | 12.2058 s | 84.9796 s |
| 5 | 12.2654 s | 148.0287 s |
| **Mean ± std** | **12.189 ± 0.326 s** | **128.214 ± 27.492 s** |

Pre-refactor partial `B=500` paired rounds (3 of 5):

| Round | CPU, 10 workers | GPU |
|---:|---:|---:|
| 1 | 21.5205 s | 255.1203 s |
| 2 | 25.6293 s | 129.0415 s |
| 3 | 22.2056 s | 132.9559 s |
| **Partial mean ± sample std** | **23.118 ± 2.201 s** | **172.373 ± 71.690 s** |

This partial `B=500` mean has only three rounds and a wide GPU spread, so it is
directional rather than a completed benchmark result. The `B=250` result is
the complete five-round comparison.

Before the refactor, the GPU mean was about **10.5× slower** than the 10-worker
CPU at this production-like step count and batch. At 120 steps GPU
points/second increased with batch size, but that did not offset the measured
1,000-step CPU advantage.

### GPU shared-basis refactor

The shared-slow GPU loop was changed to mirror the CPU's exact basis
factorization. Per timestep, it now rotates each microwave field into the
shared slow eigenbasis once, combines the rotated fields across scan points,
and propagates states in that basis before rotating them back once. This avoids
two per-point dense basis-change multiplies and avoids constructing a separate
`A = V @ V_rot` matrix for each scan point.

The same `N_steps=1000`, `B=250`, four-state setup was rerun with five
interleaved rounds and one warmup:

| Path | Mean ± std | Relative to 10-worker CPU |
|---|---:|---:|
| CPU, 10 workers | `11.698 ± 0.425 s` | baseline |
| GPU, before refactor | `128.214 ± 27.492 s` | `10.52x` slower* |
| GPU, after refactor | `62.687 ± 0.763 s` | `5.36x` slower |

*The earlier CPU mean was `12.189 s`; the before/after rows came from separate
benchmark sessions, so the ratio is approximate. The GPU refactor reduced its
mean runtime by about **2.05x**. Maximum tracked-probability difference from the
CPU was `1.618e-07`; maximum state infidelity was `6.66e-15`.

The 10-worker CPU remains faster for this case. The benchmark runner accepts
`--cpu-workers` and interleaves CPU/GPU rounds when both are enabled.

## Benchmark 1: Synthetic Batched Slow-Hamiltonian CPU vs GPU

Command:

```powershell
$nvidia = Join-Path (Get-Location) '.venv\Lib\site-packages\nvidia'
$bins = Get-ChildItem $nvidia -Directory | ForEach-Object { Join-Path $_.FullName 'bin' } | Where-Object { Test-Path $_ }
$env:PATH = (($bins -join ';') + ';' + $env:PATH)
.\.venv\Scripts\python.exe src\state_prep_gpu\benchmark_compare.py
```

Configuration:

- `T=200`
- `B=64`
- `S=1`
- `n=32`
- `dt=1e-7`
- warmup: 1
- repeat: 3

Results:

- CPU NumPy best: `3.012 s`
- GPU CuPy best: `0.871 s`
- Speedup: `3.46x`

Interpretation:

The array-based GPU path is clearly faster for a synthetic batched slow-Hamiltonian workload with moderate batch size. This supports keeping the flat GPU API for ensemble-style workloads.

## Benchmark 2: Real SPA2-Like CPU vs GPU

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'
$nvidia = Join-Path (Get-Location) '.venv\Lib\site-packages\nvidia'
$bins = Get-ChildItem $nvidia -Directory | ForEach-Object { Join-Path $_.FullName 'bin' } | Where-Object { Test-Path $_ }
$env:PATH = (($bins -join ';') + ';' + $env:PATH)
.\.venv\Scripts\python.exe src\state_prep_gpu\benchmark_spa2_real.py
```

Configuration:

- `N_steps=6000`
- detuning batch: `25`
- Hilbert-space dimension: `n=64`
- microwave frequency: `26.668808 GHz`
- CPU path: `Simulator.run_microwave_scan(...)`
- GPU path: `simulate_batched_scalar_microwaves_shared_slow(...)`

Results:

- CPU shared-slow scan: `79.255 s`
- GPU shared-slow scalar microwave scan: `74.840 s`
- Speedup: `1.06x`
- Max `|delta probabilities_final|`: `4.228e-06`
- CPU example depletion range: `0.000..1.000`

Interpretation:

The GPU implementation is numerically close for this benchmark, but only slightly faster. For this real SPA2 shape, the GPU path does not yet deliver the synthetic benchmark speedup. Likely reasons include small-to-moderate batch size, repeated per-step eigensolves, and overhead in the current shared-slow GPU implementation. Larger batches may improve GPU utilization, but this should be measured directly.

## Benchmark 3: CPU Output Storage Modes

Inline script used a real non-microwave field/Hamiltonian setup:

- `N_steps=2500`
- `n=36`
- initial states: `4`
- microwave fields: none
- eigensolver: `zheevd`

Results:

| Mode | Time | Stored Output | Saved Steps |
| --- | ---: | ---: | ---: |
| full default | `0.341 s` | `8.93 MiB` | `2500` |
| downsample every 20 | `0.291 s` | `0.45 MiB` | `126` |
| final-only probabilities | `0.292 s` | `0.00 MiB` | `2500` |
| final monitor-only | `0.289 s` | `0.00 MiB` | `2500` |

Relative to full default:

- Downsample every 20: `1.17x` faster and about `20x` less stored output.
- Final-only: `1.17x` faster and eliminates full trajectory output arrays.
- Monitor-final-only: `1.18x` faster and avoids full probability output.

Interpretation:

For this small non-microwave case, runtime is dominated by eigensolves, so storage settings only improve wall time modestly. The memory reduction is large and will matter more for long scans, many initial states, or notebook workflows that hold multiple results at once.

## Benchmark 4: CPU Shared-Slow Microwave Scan vs Repeated Single Runs

Inline script compared `Simulator.run_microwave_scan(...)` against a naive loop over `Simulator.run(...)` with one detuning per run.

Configuration:

- SPA2-like setup using the benchmark helper functions.
- `N_steps=1000`
- detuning batch: `5`
- Hilbert-space dimension: `n=64`
- microwave fields: `2`
- both paths used final-only probability storage.

Results:

- Shared-slow scan: `3.352 s`
- Naive repeated runs: `6.353 s`
- Speedup: `1.90x`
- Max absolute probability difference: `7.314e-10`

Interpretation:

The shared-slow CPU scan is a clear win for microwave parameter scans and agrees closely with repeated single-run results. This validates the recommendation to route detuning/power scans through `run_microwave_scan(...)` instead of notebook-level loops over `run(...)`.

## Inconclusive Benchmark: GPU `out=` Buffer Reuse

I attempted a focused benchmark comparing:

- `slow_hamiltonian_batch(terms, E, B)`
- `slow_hamiltonian_batch(terms, E, B, out=preallocated)`

The benchmark timed out repeatedly:

- `B=512`, `n=64`, `repeat=2000`: timed out after 300 s.
- `B=128`, `n=64`, `repeat=200`: timed out after 180 s.
- `B=32`, `n=32`, `repeat=50`: timed out after 120 s.

No timing result should be inferred from these runs. A better microbenchmark should print progress, force one-time CuPy kernel compilation before timing, and use CUDA events rather than wall-clock timing around many Python-looped calls.

## Actionable Conclusions

1. Keep and prefer `run_microwave_scan(...)` for CPU microwave parameter scans. It measured `1.90x` faster than repeated final-only single runs on a small SPA2-like scan.
2. Use final-only or monitor-only storage for scans. The speed gain was modest in the small benchmark, but memory reduction was substantial.
3. Keep the GPU path, but do not assume it is faster for every real workload. The synthetic benchmark showed `3.46x`, while the real SPA2-like benchmark showed only `1.06x`.
4. Fix or document the CuPy/NVRTC DLL path requirement on Windows. Without prepending NVIDIA package `bin` directories to `PATH`, GPU benchmarks fail before timing.
5. Add parameterized benchmark scripts so smaller and larger scan sizes can be swept without editing source or using inline scripts.
6. Revisit GPU buffer-reuse benchmarking with CUDA events and smaller, validated kernels before implementing broad buffer reuse based on timing claims.

## Parameterized Benchmark Scripts

Reusable benchmark entry points now live in `benchmarks/`. Each script writes JSON with top-level `benchmark`, `generated_at`, `environment`, `config`, and `results` fields. Passing `--csv` also writes flattened result rows as CSV.

Example commands:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_storage_modes.py --n-steps 500 --output results\storage.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_microwave_scan.py --n-steps 500 --batch 5 --compare-naive --output results\microwave.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_gpu_spa2.py --n-steps 1000 --batch-sweep 5 25 --cpu --gpu --output results\gpu_spa2.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_gpu_kernels.py --batch 32 --n 32 --output results\gpu_kernels.json --csv
$env:CUPY_CACHE_DIR = Join-Path $env:TEMP 'centrex-state-prep-cupy-cache'
.\.venv\Scripts\python.exe benchmarks\profile_gpu_spa2.py --n-steps 120 --batch 250 --output results\gpu_profile.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_gpu_eig_solvers.py --batch 250 --n 64 --output results\gpu_eig.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_gpu_precision.py --n-steps 120 --batch 250 --output results\gpu_precision.json --csv
# Historical command: this real-scan CuPy runner is now retired.
.\.venv\Scripts\python.exe benchmarks\bench_gpu_jacobi_tolerance.py --n-steps 120 --batch 250 --tolerances 1e-12 --output results\gpu_tol.json --csv
```

The real-scan Jacobi tolerance runner is retired because PyTorch does not expose
those solver controls. Use `bench_gpu_spa2.py` for current CPU/PyTorch scans, or
`bench_gpu_eig_tolerance.py` for the standalone legacy CuPy solver experiment.

Smoke-test commands run successfully for storage, CPU microwave scan, and CPU-only SPA2. The GPU kernel benchmark now runs CUDA work in a child process with `--gpu-timeout`; an earlier smoke run timed out while initializing CuPy kernels. The cache permission issue behind that timeout is diagnosed in the GPU Eigensolver Follow-up below.

## GPU Eigensolver Follow-up (2026-10-19)

The CuPy stalls above were caused by the kernel-cache location used in this
restricted run. `CUPY_CACHE_DIR` was unset, so CuPy tried to JIT-compile into
`%USERPROFILE%\\.cupy\\kernel_cache`; the active sandbox token could read that
directory but could not write to it. Host-to-device copies worked, while a
basic elementwise kernel and `eigh` stalled. Pointing `CUPY_CACHE_DIR` at a
fresh writable temp directory fixed both immediately. Reproduce GPU benchmarks
in this environment with:

```powershell
$env:CUPY_CACHE_DIR = Join-Path $env:TEMP 'centrex-state-prep-cupy-cache'
```

### Stage profile

Profiled the real SPA2 setup with `N_steps=120`, `B=250`, `n=64`, four initial
states, final-only output, and `complex128`. CUDA events sampled every 25th
propagation step; totals below are estimates from five sampled steps.

| Stage | Estimated total | Share of profiled stages |
| --- | ---: | ---: |
| Shared slow-H `eigh` | 0.408 s | 5% |
| Slow eigenvector tracking | 0.539 s | 7% |
| Microwave rotation and assembly | 0.117 s | 1% |
| Batched rotating-frame `eigh` | 6.704 s | 84% |
| State update | 0.173 s | 2% |

The batched rotating-frame eigensolve is the dominant GPU stage.

### Eigensolver choices

For 250 random Hermitian `64x64` complex128 matrices, with one warmup and five
repeats:

| Method | Mean ± std |
| --- | ---: |
| CuPy batched `eigh` (batched Jacobi) | 0.2029 ± 0.0032 s |
| Direct `cusolverDnZheevjBatched` at its default tolerance | 0.2004 ± 0.0043 s |
| 250 separate 2D CuPy `eigh` calls | 0.7815 ± 0.0094 s |

The 2D solver path is about 3.9x slower on this batch. CuPy's public `syevj`
wrapper does not expose tolerance or sweep settings, but the installed low-level
cuSOLVER bindings do. A benchmark-only adapter tested those controls. For the
same random matrix batch, loosening tolerance from `1e-7` to `1e-4` reduced the
eigensolve time from 0.1596 s to 0.1394 s, while the worst residual increased
from `3.5e-6` to `3.7e-3`. This matrix-only result did not predict propagation
accuracy; the real scan is the deciding check.

### Real-scan precision and tolerance checks

At 120 steps and 250 scan points, five interleaved GPU runs compared each
candidate with the native complex128 path:

| Candidate | Candidate time | Speedup | Max probability difference | Max state infidelity | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| complex64 | 7.480 ± 0.231 s | 0.989x | `3.747e-1` | `9.987e-1` | Reject |
| Jacobi `tol=1e-5` | 4.996 s | 1.488x | `9.373e-1` | `9.991e-1` | Reject |
| Jacobi `tol=1e-7` | 5.780 s | 1.279x | `5.065e-1` | `6.344e-1` | Reject |
| Jacobi `tol=1e-12` | 7.113 s | 1.059x | `4.056e-6` | `9.510e-11` | Promising |

The complex128 native time for the precision comparison was `7.395 ± 0.185 s`.
Looser Jacobi tolerances and complex64 are both fast or similar in a short scan,
but their state errors are too large for this trajectory.

The conservative `tol=1e-12` setting was then checked at production scan size:
`N_steps=1000`, `B=250`, four initial states, midpoint sampling, final-only
output, five interleaved rounds. The baseline and candidate timings were
`62.390 ± 0.783 s` and `59.105 ± 0.605 s`; every round favored the candidate.
The paired time change was `-3.285 ± 1.090 s` (candidate minus baseline), for a
`1.056x` speedup. Final max probability difference was `3.347e-6`, with max
state infidelity `3.276e-11`. This is within the repository's
`CROSS_BACKEND_TOL=1e-5`, though the probabilities differ more than in the
short scan.

The benchmark-only direct cuSOLVER path is therefore a small, measured GPU
improvement with close final-state agreement. It is not yet wired into the
public simulator API. Even with this improvement, the earlier 10-worker CPU
run remains roughly five times faster for this 250-point scan.

### PyTorch batched `eigh` experiment

The PyTorch 2.12 CUDA backend was tested on the same RTX 5070 Ti. For the
isolated solver check, both frameworks received the same batch of 250 random
Hermitian `64x64` complex128 matrices. Timings include a device synchronization
around each call; one warmup and five interleaved repeats were used.

| Method | Mean ± std | Max eigenvalue difference vs CuPy | Residual |
| --- | ---: | ---: | ---: |
| CuPy `eigh` | 0.1957 ± 0.0013 s | 0 | `3.38e-12` |
| CuPy direct cuSOLVER Jacobi | 0.1948 ± 0.0006 s | 0 | `3.38e-12` |
| PyTorch 2.12.1 `torch.linalg.eigh` | 0.0141 ± 0.0007 s | `1.06e-12` | `1.35e-13` |

For the real SPA2 propagation check, only `eigh` ran through PyTorch. CuPy still
performed Hamiltonian construction, basis tracking, microwave assembly, state
updates, and probability calculations. DLPack passed the matrices and results
between the libraries without host copies. The run used `N_steps=1000`,
`B=250`, `n=64`, four initial states, midpoint sampling, final-only output,
one warmup, and five interleaved repeats:

| Path | Mean ± std | Paired speedup | Max probability difference | Max state infidelity |
| --- | ---: | ---: | ---: | ---: |
| CuPy for all operations | 63.846 ± 0.709 s | baseline | — | — |
| PyTorch `eigh` + CuPy for other operations | 20.199 ± 0.180 s | 3.161x | `4.079e-8` | `1.11e-14` |

The paired time change was `-43.647 ± 0.330 s` (PyTorch candidate minus CuPy
baseline), with a 95% interval of `[-44.562, -42.732] s`; all five rounds
favored the hybrid. It reduced runtime by 68.4% and closely matched final
states. This is substantially better than the prior GPU path, though the
10-worker CPU result for the same case was about 11.7 s, so the hybrid GPU is
still slower than that CPU configuration.

The benchmark implementation is an experiment, not a production backend. See
`benchmarks/bench_gpu_torch_eigh.py` and `benchmarks/bench_gpu_torch_spa2.py`.
For reproducibility, use a CUDA-enabled PyTorch 2.12 build and set a writable
CuPy cache directory before running on Windows:

```powershell
$env:CUPY_CACHE_DIR = Join-Path $env:TEMP 'centrex-state-prep-cupy-cache'
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_eigh.py --batch 250 --n 64 --warmup 1 --repeat 5 --output $env:TEMP\torch_eigh.json
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_spa2.py --n-steps 1000 --batch 250 --scan-mode detuning --initial-states 4 --warmup 1 --repeat 5 --output $env:TEMP\torch_spa2.json
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_ops.py --batch 250 --dimension 64 --initial-states 4 --warmup 2 --repeat 7 --iterations 20 --output $env:TEMP\torch_ops.json
```

### Other GPU operations in PyTorch

To estimate the headroom beyond `eigh`, the remaining repeatedly used GPU
kernels were compared at the production shapes (`n=64`, scan batch 250, four
states). Each sample repeated an operation 20 times; the seven rounds were
interleaved. This was a kernel-level check with representative inputs, not a
complete all-PyTorch propagation run. Maximum output differences were zero for
these test inputs.

| Operation | CuPy mean | PyTorch mean | PyTorch speedup | Paired result |
| --- | ---: | ---: | ---: | --- |
| Slow-H assembly | 0.133 ms | 0.117 ms | 1.138x | Unproven; 95% interval crosses zero |
| Slow eigenvector reordering (`B=1`) | 0.262 ms | 0.116 ms | 2.262x | Improvement |
| Microwave rotation and assembly | 0.431 ms | 0.359 ms | 1.202x | Improvement |
| State update | 0.947 ms | 0.600 ms | 1.577x | Improvement |

These kernels are much smaller than the eigensolve in the profile, but the
state update and eigenvector reordering are worth moving with the solver. A full
PyTorch propagation benchmark was then run to test the combined effect.

### Full PyTorch propagation

The benchmark-only implementation in `benchmarks/torch_simulator.py` keeps all
evolution tensors in PyTorch for the whole loop: slow-H assembly, Hermitian
diagonalization, eigenvector tracking, microwave rotation and assembly, state
updates, and final probabilities. CuPy is used only by the comparison harness
to run its own baseline and inspect the PyTorch result through DLPack after the
propagation. Inputs are transferred to the GPU once before timing.

At `N_steps=1000`, `B=250`, `n=64`, four initial states, midpoint sampling,
final-only output, one warmup, and five interleaved rounds:

| Path | Mean ± std | Paired speedup | Max probability difference | Max state infidelity |
| --- | ---: | ---: | ---: | ---: |
| CuPy throughout | 64.244 ± 0.611 s | baseline | — | — |
| PyTorch throughout | 17.625 ± 0.330 s | 3.645x | `4.079e-8` | `1.11e-14` |

The paired time change was `-46.620 ± 0.250 s` (PyTorch minus CuPy), with a
95% interval of `[-47.313, -45.926] s`; all rounds favored PyTorch. This full
PyTorch result also beat the PyTorch-eigh/CuPy-rest hybrid in a separate
interleaved five-round comparison:

| Path | Mean ± std | Paired speedup |
| --- | ---: | ---: |
| PyTorch `eigh` + CuPy rest | 20.495 ± 0.327 s | baseline |
| PyTorch throughout | 17.645 ± 0.196 s | 1.161x |

The paired change was `-2.849 ± 0.143 s`, with a 95% interval of
`[-3.245, -2.453] s`; every round favored the full PyTorch path. Final
probabilities agreed exactly in this run, and max state infidelity was
`6.66e-16`. Against the earlier 10-worker CPU result of about 11.7 s, however,
the full PyTorch GPU path is still about 1.5x slower.

These measurements came from the benchmark prototype. The production
`state_prep_gpu` API now uses PyTorch for the full propagation and declares it
through the optional `gpu` extra; importing the package still works without
PyTorch. The benchmark now compares the PyTorch path with the CPU scan path.
The historical CuPy and hybrid measurements above remain recorded results.
Run the current comparison with:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_full_spa2.py --n-steps 1000 --batch 250 --scan-mode detuning --initial-states 4 --warmup 1 --repeat 5 --output $env:TEMP\torch_full_spa2.json
```

### Production PyTorch backend check (2026-10-23)

The migrated `state_prep_gpu` shared-slow implementation was measured against
the 10-worker CPU scan on the same RTX 5070 Ti, using one warmup and five paired
rounds at `N_steps=1000`, `B=250`, `n=64`, and four initial states:

| Path | Mean ± std | GPU speedup vs CPU | Max probability difference | Max state infidelity |
| --- | ---: | ---: | ---: | ---: |
| CPU scan, 10 workers | 13.046 ± 1.637 s | baseline | — | — |
| Production PyTorch GPU | 18.419 ± 0.423 s | 0.708x | `1.618e-7` | `9.99e-16` |

The migrated GPU path is about 1.41x slower than the CPU scan for this case,
while the final-state agreement remains close. Reproduce with:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_gpu_spa2.py --n-steps 1000 --batch 250 --scan-modes detuning --initial-state-sweep 4 --store-every 0 --repeat 5 --warmup 1 --cpu-workers 10 --output $env:TEMP\production_torch_spa2.json
```

### GPU time-axis batching for the full PyTorch propagation

The rotating-frame Hamiltonians for different time steps are known before
propagating the states. Their eigensystems can therefore be computed in a
small time chunk as one larger independent matrix batch, then applied to the
states sequentially in time. This increases eigensolver batch size while
preserving propagation order. The benchmark-only option is
`time_chunk_size` in `benchmarks/torch_simulator.py`.

At `N_steps=1000`, `B=250`, `n=64`, four initial states, midpoint sampling,
final-only output, one warmup, and five interleaved rounds, the measured GPU
times were:

| Time chunk | Mean ± std | Speedup vs one-step chunks | Max final-probability difference |
| ---: | ---: | ---: | ---: |
| 1 | 17.105 ± 0.476 s | baseline | — |
| 8 | 13.100 ± 0.348 s | 1.289x | `2.843e-9` |
| 16 | 12.953 ± 0.247 s | 1.304x | `2.843e-9` |

The paired changes were `-3.792 ± 0.069 s` for chunk 8 and `-3.938 ± 0.087 s`
for chunk 16; both 95% intervals excluded zero. Chunk 16 was not proven faster
than chunk 8, so chunk 8 is the smaller-memory choice. At batch 250, the GPU
chunk-8 mean is close to, but still above, the earlier 10-worker CPU mean of
`11.652 ± 0.289 s`.

Chunk size has a memory-dependent limit. At batch 250, chunk 32 took about
18.14 s in a three-round screen, so it was rejected. At batch 1,000, chunk 16
exceeded the 16 GiB GPU memory limit during eigensolver workspace allocation.
Chunk 4 fit, but its first paired round was slower than the CPU, so that run was
stopped without treating the single round as a completed comparison. The best
chunk size must be tuned for scan batch and available memory.

Screening at `N_steps=250`, `B=250` found the same trend: chunks 4 and 8 were
1.266x and 1.284x faster than chunk 1 over three rounds, with final
probabilities within `5.0e-9`. Reproduce the full GPU-only comparison with:

```powershell
$env:CUPY_CACHE_DIR = Join-Path $env:TEMP 'centrex-state-prep-cupy-cache'
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_time_batch.py --n-steps 1000 --batch 250 --scan-mode detuning --initial-states 4 --chunks 1 8 16 --warmup 1 --repeat 5 --output $env:TEMP\torch_time_batch.json
```

### Tiling time and scan axes for larger GPU batches

To fit a larger time chunk without allocating rotating-frame eigensolver
workspace for every scan point at once, the full PyTorch path was extended to
tile scan points inside each time chunk. At `N_steps=1000`, `B=1000`, four
initial states, and final-only output, a dense time-chunk-1 reference was
compared with time chunk 16 and scan tiles of 250. Both paths used all 1,000
detunings. The tiled candidate completed without out-of-memory errors:

| PyTorch CUDA path | Mean ± std | Speedup | Max final-probability difference |
| --- | ---: | ---: | ---: |
| Time chunk 1, full scan batch | 54.487 ± 0.042 s | baseline | — |
| Time chunk 16, scan tile 250 | 50.600 ± 0.021 s | 1.077x | `5.53e-9` |

Five interleaved repeats gave a paired runtime change of `-3.886 ± 0.015 s`
(standard error), with a 95% interval of `[-3.928, -3.845] s`. This is a
repeatable but modest 7.1% improvement. A short all-64-initial-states check
(`N_steps=16`, `B=4`, scan tile 2, time chunk 4) also passed, with maximum final
probability difference `3.45e-8`; production-size all-state performance has
not been measured. This dense method does not depend on block structure. Re-run
the 1,000-point comparison with:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_tiled_batch.py --n-steps 1000 --batch 1000 --initial-states 4 --time-chunks 16 --scan-tiles 250 --repeat 5 --warmup 1 --output $env:TEMP\torch_tiled_batch_1000.json
```

### CUDA Graph capture for the full PyTorch trajectory (unsupported)

Tested CUDA Graph capture on the full PyTorch trajectory with `N_steps=16`,
batch 4, four initial states, and time chunks of 4. A standalone matrix multiply
captured and replayed successfully, but trajectory capture failed at its first
`torch.linalg.eigh` with CUDA's `operation not permitted when stream is
capturing` error. Since the eigensolver runs at every timestep, a full-trajectory
CUDA Graph comparison could not be timed; no speedup is established. The
diagnostic runner is `benchmarks/bench_gpu_torch_cudagraph.py`.

### Exact Hamiltonian blocks and matrix-exponential propagation

For the real SPA2 trajectory, the electric and magnetic field time series were
checked at 101 points; only their z components are nonzero. The union of
nonzero couplings in the active slow-Hamiltonian terms (`Hff`, `HSz`, `HZz`)
and both microwave matrices separates the 64 uncoupled basis states into exact
connected components of sizes `14, 12, 12, 8, 8, 4, 4, 1, 1`. The same
components were found with an exact-zero threshold and relative thresholds up
to `1e-12`. Including all transverse Hamiltonian terms instead connects all 64
states, so the block structure depends on the actual z-only trajectory.

The sum of the block-size cubes is 7,354, compared with 262,144 for one dense
64-state solve. That is a theoretical eigensolver-work ratio of about 35.6x;
it is not a measured propagation speedup. An implementation must keep each
block's eigenvectors associated with the right detuning shifts while combining
the block results into the full adiabatic basis. This exact block path is the
next GPU implementation target.

The alternative `torch.linalg.matrix_exp` propagator was compared with the
full-spectrum `eigh` path on the same GPU. It propagated both the configured
four initial states and a complete 64-vector orthonormal basis:

| Case | Steps | Eigh mean ± std | Matrix-exp mean ± std | Matrix-exp relative time | Max probability difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four configured states, batch 250 | 250 | 4.259 ± 0.072 s | 6.834 ± 0.093 s | 1.605x slower | `1.264e-7` |
| All 64 basis states, batch 250 | 120 | 2.313 ± 0.010 s | 3.532 ± 0.017 s | 1.527x slower | `1.919e-7` |

Both comparisons used one warmup and three interleaved rounds; final
probabilities agreed within `1e-5`. The full matrix exponential is slower even
when all 64 input basis states are propagated, so this candidate is rejected
for this 64-state SPA2 setup. Reproduce with:

```powershell
$env:CUPY_CACHE_DIR = Join-Path $env:TEMP 'centrex-state-prep-cupy-cache'
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_matrix_exp.py --n-steps 250 --batch 250 --scan-mode detuning --initial-states 4 --repeat 3 --warmup 1 --output $env:TEMP\torch_matrix_exp.json
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_matrix_exp.py --n-steps 120 --batch 250 --scan-mode detuning --all-states --repeat 3 --warmup 1 --output $env:TEMP\torch_matrix_exp_allstates.json
```

### Full PyTorch batch-size sweep against the 10-worker CPU path

To check whether larger scans amortize GPU overhead, the full-PyTorch SPA2
propagator was compared with the CPU scan path at batches of 250, 500, and
1,000 detunings. Each case used four initial states, `N_steps=1000`, midpoint
sampling, final-only storage, one warmup, and five interleaved repeats. The CPU
used ten workers. Accuracy remained consistent across all three batches: the
maximum final-probability difference was below `1.62e-7`, and maximum state
infidelity below `1.6e-15`.

| Batch | CPU mean ± std | Full PyTorch GPU mean ± std | GPU/CPU time | GPU peak allocated | CPU points/s | GPU points/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 11.652 ± 0.289 s | 17.620 ± 0.303 s | 1.512x | 0.56 GiB | 21.46 | 14.19 |
| 500 | 22.226 ± 0.411 s | 30.932 ± 0.411 s | 1.392x | 1.08 GiB | 22.50 | 16.16 |
| 1,000 | 43.159 ± 0.862 s | 56.950 ± 0.348 s | 1.320x | 2.13 GiB | 23.17 | 17.56 |

The relative GPU gap narrows with batch size, but the GPU remains slower through
1,000 scan points. Throughput increases with batch size on both backends; in
this range the GPU throughput is still lower. The measured peak is PyTorch's
allocated tensor memory (not total device memory). Reproduce with:

```powershell
$env:CUPY_CACHE_DIR = Join-Path $env:TEMP 'centrex-state-prep-cupy-cache'
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_batch_sweep.py --n-steps 1000 --batches 250 500 1000 --scan-mode detuning --initial-states 4 --cpu-workers 10 --warmup 1 --repeat 5 --output $env:TEMP\torch_batch_sweep.json
```

### Dense versus exact blockwise SPA2 eigensolves on CPU and GPU

To measure the exact block structure directly, I assembled a batch of 250 real
SPA2 rotating-frame 64x64 Hermitian matrices at a representative trajectory
time. The active z-only slow terms and both microwave couplings split these
matrices into blocks of `[12, 14, 12, 8, 8, 4, 4, 1, 1]`; the measured maximum
off-block entry was exactly zero. Dense and blockwise paths both computed
eigenvalues and eigenvectors (`eigh`), with the blockwise results sorted back
into the full spectrum. One warmup and five paired, interleaved repeats were
used; the CPU path used NumPy and the GPU path PyTorch 2.12.1 on the RTX 5070 Ti.

| Device | Dense `eigh` mean ± std | Blockwise `eigh` mean ± std | Speedup | Max relative eigenvalue difference |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.0680 ± 0.0020 s | 0.01085 ± 0.00017 s | 6.27x | `1.33e-10` |
| GPU | 0.01099 ± 0.00052 s | 0.00411 ± 0.00025 s | 2.68x | `9.58e-11` |

The paired 95% intervals for runtime change (blockwise minus dense) were
`[-0.0597, -0.0546] s` on CPU and `[-0.00750, -0.00628] s` on GPU. This
establishes a solver-level win on both devices for the exact SPA2 structure. It
does not establish the end-to-end propagation gain: finding and tracking the
blocks, assembling each block's microwave matrices, and combining eigenvectors
would add work at every step. The structure is tied to z-only fields; transverse
field terms connect the sectors and remove this block advantage. Reproduce with:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_gpu_block_eigh.py --batch 250 --n-steps 1000 --warmup 1 --repeat 5 --output $env:TEMP\gpu_block_eigh.json --csv
```

### Automatic exact blocks in the CPU scan path

After adding automatic support detection to `Simulator.run()` and
`run_microwave_scan()`, the full CPU scan path was compared with block detection
forced off. This SPA2 batch used 25 detunings, four initial states, `N_steps=500`,
final-only output, the production `zheevd` backend, and one scan worker. The
detector checked the slow and both microwave Hamiltonians at every midpoint and
the initial time, finding exact blocks `[12, 14, 12, 8, 8, 4, 4, 1, 1]`. The
timings include that detection pass. One warmup and five paired, interleaved
repeats gave:

| CPU path | Mean ± std | Speedup | Maximum final-probability difference |
| --- | ---: | ---: | ---: |
| Dense | 3.778 ± 0.079 s | baseline | — |
| Automatic blocks | 2.217 ± 0.060 s | 1.704x | `2.43e-9` |

The paired runtime change was `-1.561 ± 0.024 s`, with a 95% interval of
`[-1.628, -1.493] s`; all five rounds favored blocks. This is an end-to-end
batch scan speedup for this SPA2 configuration, not a general guarantee: a
transverse field or off-axis microwave may merge enough sectors that the
detector falls back to a dense solve. Both public paths accept
`block_eigh=False` to force the dense reference. Reproduce with:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_cpu_block_eigh.py --path scan --n-steps 500 --batch 25 --workers 1 --warmup 1 --repeat 5 --output $env:TEMP\cpu_block_eigh_scan.json
```

### Full PyTorch GPU trajectory with cached blockwise eigensolves (rejected)

The blockwise solver was also tried inside the full PyTorch SPA2 trajectory,
using 250 detunings, four initial states, 1,000 steps, one warmup, and three
paired repeats on the RTX 5070 Ti. The dense path averaged `17.224 ± 0.044 s`;
the blockwise candidate averaged `11.213 ± 0.139 s` (1.536x faster). GPU
utilization sampled between 54% and 95% during this PyTorch run.

Do not use this speedup: the cached rotating-frame partition did not remain
exact over the trajectory. Its maximum relative off-block entry reached
`4.57e-7`, and the maximum final-probability difference from dense propagation
was `0.946` (maximum state-vector difference `1.31`). The apparent speedup is
therefore numerically invalid. The partition must account for all couplings
and their changing locations throughout the trajectory before it can be used.
The experimental runner is `benchmarks/bench_gpu_torch_block_trajectory.py`.

### Full PyTorch GPU trajectory using the CPU block detector

The trajectory benchmark was repeated with the exact same
`_find_exact_hamiltonian_blocks` detector used by the CPU simulator. It unions
exact nonzero support from the slow Hamiltonian and both microwave Hamiltonians
at the initial time and every midpoint in the 1,000-step trajectory. Detection
took 0.183 s on the CPU; propagation and all dense or blockwise eigensolves ran
in PyTorch on the RTX 5070 Ti. Both variants used 250 detunings, four initial
states, final-only output, one warmup, and five interleaved repeats.

| PyTorch CUDA path | Mean ± std | Speedup | Maximum final-probability difference | Maximum final state-vector element difference |
| --- | ---: | ---: | ---: | ---: |
| Dense | 17.589 ± 0.758 s | baseline | — | — |
| CPU-detector blocks | 10.477 ± 0.734 s | 1.679x | `1.62e-7` | `2.48e-8` |

The earlier cached-partition trial above is rejected because it did not use this
trajectory-wide support union. This run does use the CPU detector unchanged;
the CPU block-detection method was not altered for the GPU experiment. The
blockwise GPU solve agrees closely with dense PyTorch and saves an average of
7.111 s per trajectory in this benchmark. Reproduce with:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_gpu_torch_block_trajectory.py --n-steps 1000 --batch 250 --warmup 1 --repeat 5 --output $env:TEMP\gpu_torch_block_trajectory_cpu_detector.json
```
