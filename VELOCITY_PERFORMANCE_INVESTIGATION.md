# Velocity scan follow-up: four optimization experiments

Measured 2026-10-24 against production commit `0d6e981`. Production code and
defaults were not changed. Source-checked variants live in
`benchmarks/velocity_prototypes.py`; the paired runner is
`benchmarks/bench_velocity_followup.py`.

The strongest remaining opportunities are **keeping velocity groups together
in CPU workers** and **batching full microwave eigensolves over time on GPU**.
Grouped CPU propagation also helps the larger case. Native solver workspace
reuse alone did not help; deferred checks lost their advantage after time
batching. Gains below must not be added together.

## Workloads and method

Both cases use the current velocity-aware shared-slow scan, not independent
velocity runs. The baseline already shares full eigensolves for identical
microwave parameters. Each case uses the varying-E SPA2 setup, including
transverse electric-field components, dense solves (`block_eigh=False`),
64 states, four initial states and complex128:

| Case | Spatial samples | Microwave parameter rows | Forward velocities | Total points |
| --- | ---: | ---: | ---: | ---: |
| Small | 150 | 5 | 8 | 40 |
| Larger | 300 | 25 | 10 | 250 |

Speeds span 140–240 m/s. Input ordering is velocity-major: each velocity has
all microwave parameter rows. Uniform reference sampling corresponds to one
common spatial grid, with per-point physical timesteps. These sample counts
compare implementation cost and agreement on the same grid, not timestep
convergence of the underlying physics.

Every timing comparison used one warmup and five interleaved repeats in one
session. Tables give mean ± sample standard deviation. Claimed improvements
require both a paired 95% interval excluding zero and a mean difference larger
than the summed standard deviations. No outliers were removed. The native
deferred larger-case result passes the paired interval but fails the spread
criterion, and is therefore called unproven here.

Setup objects/input preparation are outside timings. Per-call grouping, scan
preparation, propagation and result assembly are inside. CPU BLAS threads are
pinned to one; GPU uses one PyTorch CPU thread and synchronizes every timed
call. Native convergence statuses are always checked before accepting output,
including in the deferred variant. CPU worker warmup uses reusable loky
workers; these timings do not measure launching cold standalone jobs.

Environment: Ryzen 7 9800X3D, RTX 5070 Ti (16 GiB), Windows, Python 3.11.13,
NumPy 2.3.2, SciPy 1.17.1, PyTorch 2.12.1+cu132.

## Current bottlenecks after velocity reuse

CPU profiling instrumented the actual native `zheevd` calls, checking that each
step makes exactly one slow solve plus one solve per unique microwave row.
The eigensolver fraction is about 51% for the small case and 53% for the larger
case, rather than the much higher pre-velocity-reuse fraction. In the larger
profile, 7,774 solves took 2.628 s of a 4.975 s profiled run: 0.103 s slow and
2.525 s microwave. Profiling adds overhead; these are diagnostic fractions.

Sampled production GPU event intervals put the full microwave eigensolve at
about 66% (small) and 70% (larger) of the sum of sampled stage means. State
updates are about 17% of that sum in the larger case. These CUDA intervals are
not an exclusive decomposition of end-to-end wall time, and do not justify
calling all synchronization time removable host overhead.

## 1. Grouped velocity propagation

CPU currently reuses an eigensolve but applies its eigenvectors separately to
each velocity's states. The prototype packs each group's velocity/state rows
into one matrix multiplication, applies individual phase factors, then packs
the reverse multiplication. Ragged groups and original output order are
preserved. The experiment targets the frozen propagator; Magnus and the
multitone path were not changed.

| CPU case | Current | Grouped updates | Assessment |
| --- | ---: | ---: | --- |
| Small | 0.5382 ± 0.0361 s | 0.4929 ± 0.0126 s | 8.4% lower mean, unproven |
| Larger | 4.8038 ± 0.2043 s | 3.9474 ± 0.1793 s | **17.8% less runtime** |

The larger paired difference interval was [-1.108, -0.605] s.

The GPU prototype avoids expanding eigenvectors over every velocity and packs
states into grouped matrix multiplications. It was not useful by itself:

| GPU case | Current | Grouped updates | Assessment |
| --- | ---: | ---: | --- |
| Small | 0.4419 ± 0.0139 s | 0.4436 ± 0.0095 s | Unproven |
| Larger | 1.0913 ± 0.0380 s | 1.0618 ± 0.0227 s | 2.7% lower mean, unproven |

After full timestep batching removes much of the solve cost, grouped updates
can matter more; see the separate combined comparison below. The GPU packing
prototype currently requires equal velocity-group multiplicities and rejects
ragged groups. Production grouping itself still handles ragged groups.

## 2. Keep groups together across CPU workers

The current contiguous split scatters identical microwave parameter rows when
input is velocity-major. Each worker then repeats those eigensolves. The
prototype assigns complete groups to worker bins, balances point counts, caps
workers at the number of groups, and restores every output to input order.
Each worker still repeats the slow solve, as in production.

| Case | Workers | Current contiguous split | Group-preserving split | Runtime reduction |
| --- | ---: | ---: | ---: | ---: |
| Small | 2 | 0.5015 ± 0.0428 s | 0.4027 ± 0.0211 s | 19.7% |
| Small | 4 | 0.5027 ± 0.0216 s | 0.3382 ± 0.0094 s | 32.7% |
| Larger | 2 | 3.9780 ± 0.0380 s | 2.6756 ± 0.0472 s | 32.7% |
| Larger | 4 | 3.6291 ± 0.0282 s | 1.6146 ± 0.0139 s | **55.5%** |

All four pass both significance criteria. For the larger four-worker case,
the paired difference interval was [-2.049, -1.980] s. Results also agree with
the serial path. This gain depends on input ordering and repeated parameter
rows; parameter-major input may already keep most groups together, and a
scan with no repeated rows cannot save eigensolves this way. Grouped CPU
propagation was not enabled in these worker comparisons, so no combined CPU
speedup has been established.

## 3. Full microwave eigensolves batched across time

Production already batches slow eigensystems in 128-step tiles. The new
prototype also builds and diagonalizes full microwave Hamiltonians for four
or eight known spatial steps at once, then tracks slow labels and propagates
states sequentially. A time tile is built over unique microwave rows, not the
full velocity-expanded batch. No state propagation is reordered in time.

| Case | Current | Full time tile 4 | Full time tile 8 |
| --- | ---: | ---: | ---: |
| Small | 0.4419 ± 0.0139 s | 0.2159 ± 0.0200 s | **0.1667 ± 0.0052 s** |
| Larger | 1.0913 ± 0.0380 s | 0.7165 ± 0.0110 s | **0.6622 ± 0.0325 s** |

Eight-step tiles reduced runtime by 62.3% (2.65× throughput) and 39.3%
(1.65×). Their paired difference intervals were [-0.290, -0.260] s and
[-0.494, -0.364] s. Both pass the spread criterion as well.

Memory scales with time tile × unique microwave rows × state dimension
squared, plus eigensolver workspace. The larger case's time-8 solve batch is
200 matrices, rather than 2,000 if velocities were redundantly included.
Observed peak allocated memory was about 1.15 GB for current versus 1.29 GB
for time-8 in the larger comparison process. These totals include common
resident inputs and warmed native-solver scratch; reserved allocator memory
is cumulative and is not an isolated per-variant requirement. Larger unique
parameter grids still need scan-axis tiling or a memory-aware fallback.

This benchmark prototype deliberately supports dense solves, 128-step slow
tiles and full time tiles dividing 128. It rejects profiling requests rather
than inventing stage timings. A production integration must retain exact-block
sector labels, bounded allocations, profiling semantics and fallback behavior.
The current scalar GPU route still does not support same-manifold beat phases.

## 4. Native solver handle/workspace reuse

The installed cuSOLVER adapter caches handles, parameters and workspace by
stream and shape. The checked variant reads convergence status after each
solve; the deferred variant checks all statuses before accepting the run.
Both retain the same native driver. This separates allocation/dispatch reuse
from avoiding per-solve status synchronizations.

| Case | Current PyTorch | Native checked | Native deferred |
| --- | ---: | ---: | ---: |
| Small | 0.4419 ± 0.0139 s | 0.4464 ± 0.0122 s | 0.3754 ± 0.0013 s |
| Larger | 1.0913 ± 0.0380 s | 1.0845 ± 0.0272 s | 1.0333 ± 0.0205 s |

Workspace/handle reuse with checked status alone has **no proven win**. Deferred
checks reduce small-case runtime by 15.0%. The larger 5.3% mean reduction is
unproven under the combined-spread rule, despite a paired interval just below
zero. This adapter is Windows/installed-CUDA-specific, complex128-only and
benchmark-only; it is not a replacement production dependency.

## Combined GPU comparisons

Another one-warmup/five-repeat interleaved comparison used time-8 as its
baseline. These are fresh paired timings, not percentages added to the first
comparison:

| Case | Time-8 baseline | Time-8 + grouped updates | Time-8 + native deferred |
| --- | ---: | ---: | ---: |
| Small | 0.1766 ± 0.0089 s | 0.1766 ± 0.0057 s | 0.1668 ± 0.0296 s |
| Larger | 0.7038 ± 0.0330 s | 0.6465 ± 0.0212 s | 0.6986 ± 0.0066 s |

Grouped updates save another 8.1% in the larger case, passing both criteria;
the paired difference interval is [-0.101, -0.013] s. The small case remains
unproven. Native deferred checks add no proven improvement after timestep
batching in either case. This makes native integration a low priority.

## Numerical checks and reproduction

All repeated variants passed same-physical-basis probability and state-fidelity
checks. The measured repeated cases reported zero final probability difference
against their corresponding baselines; norm/fidelity errors are retained in the
raw reports. Twelve tests passed for ragged CPU groups, output order after
worker regrouping, exact/dense CPU blocks, CPU/CUDA tensor execution, nonuniform
timesteps, histories, monitors, true initial fields, late transverse fields,
and partial time tiles crossing the 128-step slow-tile boundary. The GPU
combined grouping/time-8 variant is included in these checks.

A longer numerical check used 1,000 spatial samples, 25 microwave rows and
10 velocities (250 points). Both time-8 and grouped/time-8 matched production
GPU final probabilities exactly in the common basis; maximum state-infidelity
deviation was 8.66e-13 and norm error 4.33e-13. A seven-point CPU subset agreed
within 2.13e-9 in final probabilities. This was a numerical check, not a
single-run speed comparison; its report is
`results/velocity_followup_long_validation_20261024.json`.

Run comparisons with fresh output paths:

```powershell
$env:PYTHONPATH = "$PWD\src"
& $python benchmarks/bench_velocity_followup.py --warmup 1 --repeat 5 --output results/velocity_followup_new.json
& $python benchmarks/bench_velocity_followup.py --phase combined --warmup 1 --repeat 5 --output results/velocity_followup_combined_new.json
& $python -m pytest tests/test_velocity_prototypes.py
```

`--shapes` takes `steps:microwave-points:velocities`; `--phase` can select
grouped propagation, CPU partitioning, GPU experiments, or combined GPU checks.
The paired runner writes timing samples, spreads, intervals, numerical errors,
profiles and peak memory. CPU cProfile files are saved beside the JSON.

Raw reports from this session are in ignored
`results/velocity_followup_20261024.json` and
`results/velocity_followup_combined_20261024.json`. Smoke runs with one repeat
were functional checks only and are not performance evidence.

Recommended implementation order: group-preserving CPU worker partitioning,
bounded full GPU timestep tiles, grouped CPU propagation, then optional grouped
GPU updates where measurements justify them. Preserve the scalar GPU beat-phase
restriction and CPU multitone timing. Do not integrate the native adapter on
the strength of the pre-tiling small-case result.
