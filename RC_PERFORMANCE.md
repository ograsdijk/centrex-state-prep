# RC J12 background performance

Implemented and measured 2026-10-24. This work specifically includes the RC
J12 background at a different carrier from SPA J12, with mixed polarization
`z + 0.4 y`, rather than transferring no-beat velocity-scan speedups to it.

Two changes are available:

- CPU multitone `propagator="magnus"` reuses oscillatory integrals across power
  points with identical shifts, velocities and beat frequencies. It is the
  same integrator and timestep rule; only repeated calculation is removed.
- The GPU has a beat-correct frozen multitone entry point with bounded time
  and scan tiles. The CPU/GPU launcher selects it for distinct carriers on
  one manifold. Each velocity retains its own beat phase and full eigensolve.

The default CPU frozen integrator is unchanged. A component-rotation/hoisting
prototype showed no reliable speedup and was not integrated. Native solver
adapters were not needed for the new GPU route.

## What was measured

The setup is SPA J12 + SPA background + RC J12 background, with both background
intensities initially 1/35 of the main beam's center intensity. The grid is
five SPA detunings (-2, -1, 0, 0.5, 1.5 MHz) × five background intensity
prefactors (1, 2, 4, 8, 16). Main and SPA-background detunings move together;
the RC detuning column stays zero, so its physical carrier remains fixed.
The RC carrier offset is specified relative to its zero-electric-field
transition frequency. The tested offsets 0 and -7.4 MHz correspond to
approximately +0.566637 and -6.833363 MHz relative to the SPA carrier.

Timings use 300 spatial samples, 25 parameter points, four initial states,
64 basis states, complex128, one warmup and five interleaved repeats on the
same machine/session. Tables report mean ± sample standard deviation. CPU
BLAS is pinned to one thread; GPU uses one PyTorch CPU thread and synchronizes
each timed run. CPU workers are warmed reusable loky processes. Setup-object
and GPU-array preparation are outside timings; per-call grouping, CPU scan
preparation, tile construction and propagation are inside.

Environment: Ryzen 7 9800X3D, RTX 5070 Ti (16 GiB), Windows, Python 3.11.13,
NumPy 2.3.2, SciPy 1.17.1, PyTorch 2.12.1+cu132. CPU before/after comparisons
read the old implementation from Git commit `0d6e981` and compare it with the
integrated implementation in the same session.

## CPU: cache integrals, not velocity-dependent propagators

For the frozen reference, native 64×64 eigensolves account for roughly 73–74%
of the profiled serial runtime. Mixed RC polarization leaves no useful exact
split in this setup; these runs execute full 64×64 solves. The investigated
component-rotation shortcut changes too little work to show a reproducible win.

Magnus already avoids the per-point full eigensolve, but was recomputing the
same oscillatory kernels for every intensity prefactor. Its new cache key
includes all reference-frame diagonal shifts, the traversal-time scale, and
every beat frequency. It excludes power, which only multiplies the integrated
coupling. Cache scope is one spatial step; unique points retain no cached
matrices and there is no full-trajectory or cross-run cache.

| RC offset | Previous CPU Magnus, one worker | Integrated CPU Magnus | Runtime reduction |
| --- | ---: | ---: | ---: |
| 0 MHz | 2.623 ± 0.035 s | 1.673 ± 0.045 s | 36.2% |
| -7.4 MHz | 2.440 ± 0.174 s | 1.587 ± 0.030 s | 35.0% |

Paired difference intervals were [-0.977, -0.922] s and [-1.034, -0.672] s.
Both pass the combined-spread criterion too. Common-basis probability changes
were at most 5.33e-15. This gain depends on repeated detunings/velocities across
power points; it is not a 35% gain for an arbitrary frequency-only scan.

CPU frozen output was unchanged, with no demonstrated runtime change. No
Magnus-versus-frozen accuracy or speed-at-matched-accuracy claim is made by
this comparison. Selecting an integrator or reducing step count remains a
separate accuracy decision for the desired RC scan.

## GPU: beat-correct full time tiles

The reference upper/lower expression is
`exp(-i phi) upper + exp(+i phi) lower`. An initial prototype represented this
with two Hermitian quadratures per field. The integrated array API takes upper
components directly, constructs the lower part as its adjoint, and generates
the phase weights on-device at each point's physical sample time. This avoids
storing a trajectory-sized `(T,B,n,n)` microwave array or a full `(T,B,M)` beat
envelope. No beat is dropped and velocities do not share full eigensystems
merely because they have the same detuning and power.

Slow eigensystems are tiled over up to 128 samples; full microwave solves over
up to eight. State updates and adiabatic label tracking remain sequential.
`max_eigh_matrices=256` caps every solver call, including slow tiles; scan-point
tiles are used when the full time tile would exceed that count. Retained full
eigensystems scale as time tile × batch × n², not trajectory length. The
solver is the installed PyTorch dense Hermitian eigensolver, with its ordinary
convergence checks. No native ABI adapter or aligned-field assumption is used.

| RC offset | CPU frozen, four workers | GPU per-step full solve | GPU eight-step full tiles |
| --- | ---: | ---: | ---: |
| 0 MHz | 1.423 ± 0.016 s | 1.115 ± 0.036 s | 0.631 ± 0.038 s |
| -7.4 MHz | 1.354 ± 0.022 s | 1.093 ± 0.021 s | 0.598 ± 0.005 s |

The integrated tiled GPU path takes 55.7–55.9% less time than four CPU workers
(2.26–2.27× throughput) for this workload. Paired GPU-tile minus CPU intervals
were [-0.826, -0.759] s and [-0.787, -0.727] s; both pass the spread criterion.
Final common-basis probability differences were at most 6.90e-9. Both GPU
columns use the integrated beat-correct path; the time-one variant isolates
the effect of full time tiling without comparing different physical models.

These are warm propagation comparisons, not standalone job startup timings.
Input preparation and factorization checks can dominate short jobs. The
launcher verifies fixed-matrix/scalar-envelope factorization at every actual
spatial sample and rejects unsupported changing polarization instead of
silently approximating it. The mixed RC polarization used here is supported.

## Numerical verification

- Full suite: **282 passed**, with the existing tracked-label crossing warning.
- Independent CPU matrices verify phase sign, units, per-point velocities,
  nonuniform timesteps, histories, monitors, partial time tiles, true initial
  fields and late transverse fields. Tests force small scan caps and verify
  the maximum number of matrices passed to each eigensolve.
- Additional tests cover CPU Magnus cache keys with differing RC frequencies
  and velocities, real RC fields at both offsets, and CUDA stream/profiling
  behavior. Zero beats agree with the existing scalar GPU path.
- At 2,000 samples, 27 detuning/power/velocity points and both RC offsets,
  frozen GPU probabilities agreed with a seven-point CPU subset within
  3.82e-9. Maximum GPU norm error was 4.16e-13 and overlap-infidelity deviation
  4.56e-13. CPU cached Magnus probabilities matched the previous implementation
  in that longer check; overlap deviation from one was at most 3.34e-11.
- An RC launcher check ran CPU workers and CUDA concurrently with different
  velocities, agreed with a CPU reference within 3.31e-9, retained metadata,
  and resumed its matching checkpoints.
- The existing CPU/GPU comparison example retained its 1.95e-15 maximum state
  difference. PyTorch remains optional at package import time.

The timing grids and longer checks establish agreement with the same
discretization, not physical timestep convergence of rapidly beating fields.
No speed claim uses a single run or a change of timestep/integrator.

## Using it

CPU callers keep `allow_multitone_same_manifold=True` and their chosen
`propagator`. Integral reuse is automatic for the multitone Magnus path.

The flat GPU API is
`state_prep_gpu.simulate_batched_multitone_shared_slow`; its inputs and units
are documented in `src/state_prep_gpu/README.md` and its docstring. Use
complex128 for long scientific runs. Complex64 is tested with looser
roundoff/norm tolerances and was not used for these scientific comparisons.

Existing CPU/GPU launcher setup files can now contain distinct same-manifold
carriers. Keep RC detuning zero in parameter rows when scanning the SPA carrier.
Optional `forward_velocities` `(B,)` preserve physical beat timing. The launcher
automatically selects the multitone GPU bridge and fingerprints its source
along with the simulation code for checkpoint reuse.

Reproduce into fresh output paths:

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python benchmarks/bench_rc_background.py --production --warmup 1 --repeat 5 --output results/rc_background_new.json
& $python benchmarks/check_rc_agreement.py --output results/rc_long_agreement_new.json
& $python -m pytest
```

Raw session reports are ignored files `results/rc_background_20261024.json`
(prototype screening), `results/rc_background_production_20261024.json`
(integrated repeated comparisons), and `results/rc_long_agreement_20261024.json`
(longer numerical checks). CPU profiles are saved beside the benchmark JSON.
Single-repeat smoke outputs are functional checks only.

## Follow-up: keep the frozen method, compare CPU backends and GPU tiles

The CPU Magnus gain above is specific to users already selecting that
integrator. It does not make the default CPU frozen method 35% faster.
Two further experiments retained the frozen method and its spatial grid.

### Installed PyTorch CPU backend: rejected

An alternative CPU path used the installed PyTorch build (Intel oneAPI MKL
2026.0) with one thread, versus SciPy/OpenBLAS 0.3.30 with one BLAS thread.
No new dependency was installed. A corpus of 75 actual RC rotating-frame
matrices, sampled at three trajectory positions, compared the same upper
Hermitian triangle. Residuals, orthogonality, and relative spectral errors
passed; relative eigenvalue differences were at most 2.80e-15. Relative
checks matter here because the frame matrices contain large rotational
energies. Full propagation was checked independently in a common physical
basis rather than relying only on eigensolver residuals.

One warmup/five interleaved repeats measured:

| CPU workload | SciPy reference | PyTorch CPU loop/per-step | PyTorch CPU batched/time-8 |
| --- | ---: | ---: | ---: |
| 75 captured eigensolves | 0.0204 ± 0.0004 s | 0.0266 ± 0.0008 s | 0.0246 ± 0.0003 s |
| Full frozen scan, 300 samples/25 points | 2.637 ± 0.038 s | 2.924 ± 0.036 s | 2.911 ± 0.092 s |

The full alternative CPU paths were 10.9% and 10.4% slower, respectively.
The paired intervals exclude zero and differences exceed summed spreads.
Final probabilities agreed within 9.03e-9. The alternative is not integrated;
SciPy remains the CPU default. These are same-session comparisons, not a
comparison with CPU timings from the earlier tables on another run.

### Larger GPU tiles: small additional gain at a memory cost

For the same 25-point RC scan at -7.4 MHz, now with 1,000 spatial samples,
one warmup/five interleaved repeats compared integrated GPU configurations:

| Full time tile / maximum matrices per solve | Mean ± std | Peak allocated memory | Assessment |
| --- | ---: | ---: | --- |
| 8 / 256 (default) | 1.895 ± 0.027 s | 515 MB | Reference |
| 16 / 512 | 1.827 ± 0.018 s | 961 MB | 3.6% less runtime |
| 32 / 1,024 | 1.839 ± 0.025 s | 1,852 MB | 2.9% lower mean, unproven |

The 16/512 paired interval was [-0.0941, -0.0419] s, passing both criteria.
The 32/1,024 interval crossed zero. Peak memory includes the shared prepared
inputs and retained final outputs; incremental peaks were 480, 927 and
1,817 MB, respectively. Reserved allocator memory is not used as a per-variant
requirement. Final probabilities matched across tile configurations; maximum
norm error was 9.31e-12. No step count or propagator was changed.

Defaults stay at 8/256 for the lower memory requirement. The launcher exposes
optional tuning without changing the physics:

```powershell
& $python scripts/run_cpu_gpu_scan.py --setup-file setup.dill --parameters points.npz --steps 10000 --gpu-time-chunk 16 --gpu-max-eigh-matrices 512 --output-dir results/rc_tuned
```

Both options apply to the multitone GPU route, are included in the request
fingerprint, and must be positive. The scalar no-beat GPU route is unchanged.
An end-to-end concurrent CPU/GPU check retained identical final probabilities,
resumed matching settings, and rejected changed settings for an existing
checkpoint. This is an optional workload-dependent tradeoff, not a universal
new default or an additive percentage on top of a different benchmark.

Reproduce these follow-ups with fresh output files:

```powershell
& $python benchmarks/bench_rc_frozen_cpu.py --warmup 1 --repeat 5 --output results/rc_frozen_cpu_new.json
& $python benchmarks/bench_rc_gpu_tiles.py --warmup 1 --repeat 5 --output results/rc_gpu_tiles_new.json
```

Raw session reports: `results/rc_frozen_cpu_retry_20261024.json` and
`results/rc_gpu_tiles_20261024.json` (both ignored).

## SPA referee simulations: a different RC workload

The current S18/S19 referee simulations in the SPA paper repository use AP1
(J0–J1), AP1 background, and switched RC J12 leakage, with AP2 absent.
This differs from the SPA2 plus RC J12 same-manifold beat benchmark above.
The two driven transitions use different excited manifolds; the same-manifold
multitone optimization is not the path used by S19.

The referee case has Jmax=4 (100 basis states), 16 propagated initial states,
34 physical frequency pairs, three velocity nodes, and four switching phases.
The RC polarization varies along the trajectory. Earth magnetic fields include
transverse components, so the aligned-field block shortcut does not apply.
Production midpoint sampling uses 16 steps per half-period of the 1 MHz
switching: about 48,696 samples for the central 184 m/s trajectory.

Each CPU worker handles a whole 34-frequency unit with one BLAS thread. The
GPU groups units sharing velocity and switching phase. Consequently, a
four-worker split of a 25-point SPA2 scan does not establish the relative
performance of the actual eight-to-twelve-worker referee queue.

`benchmarks/bench_referee_s19.py` reads the existing S19 physical definitions
and adapter without launching its pipeline or changing its checkpoints. It
compares the same twelve configurations (408 frequency points) across CPU
worker counts and GPU, with prepared physical inputs, one warmup, and five
interleaved repeats. GPU transfers and final-state retrieval are timed;
physical setup, checkpoint I/O, and worker startup are excluded. Reduced
timestep counts are timing pilots, not convergence checks or production ETA
measurements. A twelve-unit queue also has finite-queue load imbalance for
eight or ten workers, which must be retained when interpreting the results.

Run it from this repository with the paper's scripts directory explicitly:

```powershell
& $python -B benchmarks/bench_referee_s19.py --paper-scripts C:/Users/ogras/Documents/GitHub/CeNTREX-Argonne-SPA-testing/referee_response/scripts --steps 300 --workers 8 10 12 --repeat 5 --output results/referee_s19_fresh.json
```

This benchmark needs the paper scripts' dependencies and existing S19 solver
snapshot. It uses the current workspace CPU/GPU engines, not the snapshotted
CPU engine used by the paused production pipeline. No paper figure, fit,
checkpoint, claim, or production status is changed.

### Referee timing pilot, 2026-10-24

AMD Ryzen 7 9800X3D (8 cores, 16 logical processors), RTX 5070 Ti,
NumPy 2.3.2, SciPy 1.17.1, PyTorch 2.12.1+cu132. Both paths use frozen
propagation and complex128. Twelve configurations at the central velocity
(184 m/s), switching phase zero, and 300 full-trajectory samples were compared.
Each configuration contains all 34 frequency pairs and 16 initial states.

| CPU workers | CPU mean ± sample std | GPU mean ± sample std | GPU extra runtime |
| --- | ---: | ---: | ---: |
| 8 | 26.438 ± 0.760 s | 28.674 ± 0.356 s | 8.5% |
| 10 | 26.643 ± 0.869 s | 28.585 ± 0.883 s | 7.3% |
| 12 | 14.221 ± 0.417 s | 30.023 ± 0.576 s | 111.1% |

Each row has its own same-session, interleaved CPU/GPU comparison with one
warmup and five repeats. The GPU-minus-CPU paired 95% intervals are
[0.929, 3.542] s, [0.421, 3.463] s, and [15.075, 16.527] s, respectively.
All three differences also exceed the sum of the sample spreads. CPU and GPU
final probabilities agree within 8.30e-8 in a common physical basis.

For this twelve-unit pilot, twelve CPU workers finish about 2.11 times as
quickly as the GPU. The earlier 2.27-times GPU advantage over four CPU workers
on a smaller SPA2 beat problem does not describe this workload. The CPU worker
counts themselves were measured in separate groups rather than interleaved;
do not interpret their ratios as a controlled CPU scaling study.

These are propagation timings for a coarse pilot: production uses about
48,696 samples at the central velocity, and averages three velocities and four
switch phases. The pilot does not certify convergence, estimate a production
completion time, or measure simultaneous CPU/GPU throughput. In particular,
twelve jobs fit into one wave with twelve workers and require two waves with
eight or ten; this is a finite-batch completion comparison, not sustained
throughput for a long queue. Field preparation, readout, checkpoint I/O, and
worker startup remain outside the timer. CPU field evaluations inside its
propagator remain included; GPU field arrays are prepared in advance.

Raw report: `results/referee_s19_cpu_gpu_20261024.json` (ignored).

### Short referee follow-up: concurrent devices and GPU tiling

The follow-up uses the same central-velocity, phase-zero model, with 128
full-trajectory samples. Production remains paused. These are deliberately
coarse timing workloads; their final states are not paper simulation results.

Twelve distinct configurations were repeated three times to form 36 independent
jobs, each containing 34 frequency pairs and all 16 initial states. The
CPU-only variant uses twelve workers for all 36 jobs. The concurrent variant
uses the same twelve workers for 24 jobs and the existing scalar GPU worker
for the remaining twelve, returning identical ordered results.

| Same 36-job workload | Mean ± sample std |
| --- | ---: |
| CPU only, 12 workers | 17.987 ± 0.212 s |
| CPU 24 jobs + GPU 12 jobs, concurrently | 12.442 ± 0.244 s |

One warmup and five same-session interleaved repeats show 30.8% less runtime
(1.446 times throughput for this workload). The paired interval for the time
difference is [-5.634, -5.458] s, and the difference exceeds summed spreads.
Final probabilities agree within 9.53e-8. Worker startup, physical-input
preparation, readout, and checkpoint I/O are excluded; GPU transfers and final
state retrieval are included. This supports assigning independent jobs to both
devices. It does not establish a production ETA or the best split for a long
heterogeneous queue; one third of these jobs went to the GPU.

For the same twelve-configuration GPU batch, the alternative tested was the
existing bounded time-tiled kernel with zero beat frequencies. The S19 complex
Cartesian factorization was represented as half its assembled microwave
Hamiltonian plus its adjoint, preserving the changing RC polarization. Its
Hermiticity, reconstruction, final-state overlap, and probabilities were checked.

| GPU propagation variant | Mean ± sample std | Peak allocated memory |
| --- | ---: | ---: |
| Existing scalar worker | 11.991 ± 0.352 s | 1.42 GB |
| Time tile 8, solve cap 512 | 12.264 ± 0.340 s | 2.34 GB |
| Time tile 8, solve cap 1024 | 11.918 ± 0.235 s | 3.92 GB |

Both paired intervals cross zero: [-0.260, 0.807] s and [-0.589, 0.444] s.
Neither candidate provides a proven speedup. Probabilities agree within 2.54e-8
and norms within 1.44e-13. No scalar GPU default or production adapter was
changed; increased memory is not justified by these results.

Sampled CUDA event intervals attribute about 93% of measured GPU stage time to
rotating-frame eigensolves, about 5% to state updates, and only 0.23% to slow
eigensolves. A separate single-unit CPU profile, with Python wrappers around
the native LAPACK calls, attributes 3.192 of 3.965 profiled seconds (80.5%) to
the 4,318 rotating-frame eigensolves, versus 0.092 s (2.3%) to 127 slow solves.
Profiling overhead is included; these percentages are bottleneck attribution,
not additional paired timing claims. CPU wall profiling and GPU event intervals
use different clocks and should not be compared as absolute stage times.

Reproduce with a fresh output path:

```powershell
& $python -B benchmarks/bench_referee_followup.py --paper-scripts C:/Users/ogras/Documents/GitHub/CeNTREX-Argonne-SPA-testing/referee_response/scripts --steps 128 --repeat 5 --output results/referee_followup_fresh.json
& $python -B benchmarks/bench_referee_s19.py --paper-scripts C:/Users/ogras/Documents/GitHub/CeNTREX-Argonne-SPA-testing/referee_response/scripts --steps 128 --profile-cpu --output results/referee_cpu_profile_fresh.json
```

Raw reports: `results/referee_followup_20261024.json` and
`results/referee_cpu_native_profile_20261024.json`; the latter also has a
`.prof` artifact. All are ignored. No full production trajectory was run.
