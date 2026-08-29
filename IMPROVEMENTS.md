# Improvements

Generated: 2026-08-15

This file records ease-of-use and performance findings from a review of the
current worktree (branch `batching`, including uncommitted changes).

Relationship to the existing documents:

- `REPO_SCAN.md` describes what the repository contains.
- `PERFORMANCE_BENCHMARK_RESULTS.md` records measured numbers.
- `PERFORMANCE_IMPROVEMENTS.md` is the original prioritized performance backlog.
- This file adds findings not in that backlog, re-prioritizes several of its
  items against the follow-up measurements, and covers ease-of-use separately.

Section headings carry their status. Items marked **DONE** are implemented and
verified; items marked **REJECTED** were measured and did not survive. Anything
marked *unmeasured* is an estimate from reading the code and should not be
believed until benchmarked. Every change proposed here is to be benchmarked with
repeats, before and after; see Benchmarking Protocol below.

Implemented so far: BLAS pinning, the `V @ V_rot` hoist, Ease Of Use A through
F, and the test suite. Measured and rejected: the split-step propagator, batch
threading in three forms, and the `eig_backend` default switch.

## The Measurement That Reframes The Backlog

The follow-up profiling in `PERFORMANCE_IMPROVEMENTS.md:317` is the single most
important number available:

| Component | Share of shared-slow scan runtime |
| --- | ---: |
| `zheevd` eigensolves | `75.6%` |
| `reorder_evecs` | `1.9%` |
| Everything else | about `22%` |

Consequences:

- Priorities 6 through 10 of the existing backlog all target the remaining
  `22%`. Their combined ceiling is therefore small, and the batched-`eigh`
  prototype already measured only `2-5%`.
- The loop structure is one *shared* slow eigensolve per timestep followed by
  `B` *per-scan-point* rotating-frame eigensolves per timestep
  (`src/state_prep/simulator.py:1006`). The inner eigensolve is the workload.
- A large win requires reducing the number or cost of eigensolves. No current
  backlog item does this.

## Benchmarking Protocol For Any Change

Every performance change proposed in this file must be benchmarked with repeats
before and after, and reported with its variability. A single timing pair is not
acceptable evidence: the differences being chased here are in the `2-20%` range,
which is the same order as run-to-run noise on this machine.

### What The Harness Already Provides

`benchmarks/common.py:110` (`time_call`) already implements this correctly and
returns:

- `time_best_s`
- `time_mean_s`
- `time_std_s`
- `repeat`
- `warmup`

All four benchmark scripts accept `--repeat` and `--warmup`. Nothing new needs
to be built; the protocol below is about using it consistently.

### Required Procedure

1. Run the baseline on the *current* code, on the same machine, in the same
   session as the candidate. Never compare against a number recorded in an
   earlier session or from a different environment.
2. Use at least `--warmup 1 --repeat 5` for CPU work. Use more repeats when the
   expected effect is below `10%`.
3. Interleave baseline and candidate runs rather than running all of one and
   then all of the other. Thermal drift and background load on this machine are
   slow-moving, so a blocked A-then-B design attributes drift to the change.
4. Report `time_mean_s` together with `time_std_s`, and state the speedup as a
   ratio of means. Report `time_best_s` as a secondary figure only.
5. Treat a change as unproven if the difference in means is smaller than the sum
   of the two standard deviations. Say so explicitly rather than quoting the
   ratio.
6. Record the numerical agreement alongside the timing for any change that can
   alter results, using the tolerances in the validation notes for each item
   below.
7. Sweep at least two batch sizes and two `N_steps` values. Several conclusions
   in `PERFORMANCE_IMPROVEMENTS.md` reversed sign between shapes, most visibly
   the `loky` worker results.

### Gaps To Close In The Existing Harness

- `benchmarks/bench_gpu_spa2.py:133` defaults to `--repeat 1`, which makes
  variability unmeasurable at default settings. Raise the default and record
  spread.
- The tables in `PERFORMANCE_BENCHMARK_RESULTS.md` quote single figures such as
  `1.90x` and `1.06x` with no spread. It is not currently possible to tell
  whether the `1.06x` real SPA2 GPU result is distinguishable from no change at
  all. Re-record these with repeats before using them to justify or reject work.
- ~~There is no paired before/after runner.~~ Built: `benchmarks/paired.py`.
  `compare_paired(...)` runs variants interleaved in one session, compares them
  by per-round paired differences, and reports a `95%` confidence interval and
  an explicit `improvement` / `regression` / `unproven` verdict. It also takes a
  `check` callback so numerical agreement is recorded alongside timing.
  `python benchmarks\paired.py --self-test` validates the runner itself against
  synthetic effects of known sign.
- GPU timings must use CUDA events and an explicit synchronize, as
  `PERFORMANCE_BENCHMARK_RESULTS.md:147` already concluded after the
  inconclusive buffer-reuse microbenchmark. `time_call` accepts a `sync`
  callback for this; use it.

### First Results From The Paired Runner

Measured 2026-08-15 with `benchmarks/paired.py`, SPA2 setup, `N_steps=200`,
batch `8`, `repeat=7`, `warmup=1`, interleaved.

`eig_backend` comparison, re-testing the single-shot claim at
`PERFORMANCE_IMPROVEMENTS.md:336` that `zheevd` (`0.992 s`) beats `numpy`
(`1.028 s`):

| Variant | Mean | Std |
| --- | ---: | ---: |
| `zheevd` | `1.0273 s` | `0.0426 s` |
| `numpy` | `1.0434 s` | `0.0247 s` |

- Paired delta: `+0.0161 s +/- 0.0086 s`
- `95%` CI: `[-0.0050 s, +0.0373 s]`
- Verdict: **unproven**

The confidence interval includes zero. The existing recommendation to keep
`zheevd` as the default is not supported by evidence; it is also not
contradicted. Keep the default, but stop citing it as a measured win. This is a
concrete example of the failure mode this protocol exists to prevent: the
single-shot difference was `3.6%`, and the round-to-round spread is `4%`.

### Where The Time Goes At Real Batch Sizes

Measured 2026-08-15. SPA2 setup, `N_steps=4000`, `repeat=3`, `eig_backend="zheevd"`.
Fitting `runtime(B) = a + b*B` separates shared per-timestep cost from
per-scan-point cost without monkeypatching.

| Batch | Mean | Std | Per point |
| ---: | ---: | ---: | ---: |
| `1` | `5.674 s` | `0.245 s` | `5.674 s` |
| `5` | `13.600 s` | `0.471 s` | `2.720 s` |
| `10` | `23.127 s` | `0.123 s` | `2.313 s` |
| `25` | `52.135 s` | `0.220 s` | `2.085 s` |
| `50` | `100.665 s` | `0.117 s` | `2.013 s` |

Fit: `runtime(B) = 3.789 s + 1.937 s * B`. The model predicted batch `50` at
`100.5 s` against `100.665 s` measured, so the separation is reliable.

Per-scan-point share of runtime:

| Batch | Share |
| ---: | ---: |
| `5` | `71.9%` |
| `25` | `92.7%` |
| `50` | `96.2%` |

Consequences:

- Notebook scans use `25` to `101` points, so `93%` or more of real runtime is
  the per-scan-point inner loop. The shared per-timestep work is `4-7%`.
- This retires Priority 8 (precompute field time series) and Priority 9 (cache
  coupling matrices) from the existing backlog outright: both optimize the
  intercept, which is nearly all amortized away at real batch sizes.
- It also revises Priority D of this file downward for single large scans,
  though cross-scan caching still helps a notebook running many scans in a
  session.
- The inner loop is the only worthwhile target, which is where the eigensolver
  choice, its threadability, and the `V @ V_rot` hoist all apply.

### Backend Agreement Is Looser Than Expected

The same run compared final probabilities between the two eigensolver backends:

- Max absolute difference in `probabilities_final`: `3.835e-07`

This is roughly `400x` looser than the `1e-9` tolerance that seemed plausible
a priori, though it sits in the same range as the `4.228e-06` already accepted
for CPU/GPU agreement on the real SPA2 benchmark. Two LAPACK routines
diagonalizing identical matrices diverge this much because near-degenerate
eigenvectors differ, and the adiabatic tracking then amplifies that over the
timestep loop.

Consequences for the regression test in Ease Of Use B:

- The `7.314e-10` agreement measured for shared-slow versus repeated runs holds
  only because both paths use the *same* eigensolver. It is not a safe tolerance
  for a test that allows the backend to vary.
- Any test comparing paths that differ in eigensolver, device, or ordering needs
  a tolerance near `1e-6`, and should pin `eig_backend` explicitly when a tight
  tolerance is wanted.
- This sensitivity is worth understanding before Priority A. If final
  probabilities already move by `4e-07` from an eigensolver swap, the acceptance
  threshold for the split-step propagator must be set against that floor rather
  than against zero.

## End-To-End Proof Of The Full Stack

Measured 2026-08-15. Real simulator, SPA2 setup, `N_steps=4000`, batch `25`,
`repeat=4`, configurations interleaved round-robin across subprocesses. Each
configuration adds exactly one change to the one above it, so contributions are
isolated rather than only the combined stack being demonstrated.

BLAS thread count must be set before NumPy imports, so each measurement ran in
its own subprocess with setup excluded from the clock. No warmup: every
subprocess is equally cold.

| Config | BLAS | Backend | Impl | Threads | Mean | Std | vs A | vs prev | Max diff |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | unpinned | `zheevd` | original | `1` | `56.399 s` | `0.518 s` | `1.00x` | - | `0` |
| B | `1` | `zheevd` | original | `1` | `31.918 s` | `0.115 s` | `1.77x` | `1.77x` | `0` |
| C | `1` | `numpy` | original | `1` | `32.190 s` | `0.079 s` | `1.75x` | `0.99x` | `4.378e-07` |
| D | `1` | `numpy` | hoisted | `1` | `30.467 s` | `0.094 s` | `1.85x` | `1.06x` | `4.378e-07` |
| E | `1` | `numpy` | hoisted | `8` | `12.019 s` | `0.097 s` | `4.69x` | `2.53x` | `4.378e-07` |

Paired comparisons against A, all verdict **improvement**:

| Config | Delta | `95%` CI |
| --- | ---: | --- |
| B | `-24.480 s` | `[-25.335, -23.625]` |
| C | `-24.208 s` | `[-25.146, -23.270]` |
| D | `-25.932 s` | `[-26.690, -25.174]` |
| E | `-44.380 s` | `[-45.310, -43.449]` |

### Reading The Result

- **BLAS pinning is worth `1.77x` on its own and is bitwise identical** (`0`
  difference). Largest and safest single change found, and it was not on the
  original list at all.
- **The backend swap is not a win by itself.** At `0.99x` it is marginally
  slower than `zheevd`, within noise. Its sole justification is that
  `np.linalg.eigh` is the only backend that releases the GIL, which is what
  makes config E possible. It should not be advocated on speed grounds.
- **The hoist contributes `1.06x` here versus `1.094x` measured earlier.** Not
  noise: `V @ V_rot` is a BLAS matmul, so under unpinned BLAS it was also paying
  the oversubscription penalty and removing it saved more. The two measurements
  agree once configuration is accounted for. The hoist is worth less after the
  BLAS fix, not more.
- **Threading gives `2.53x`, not the `6.6x` of the isolated eigensolve
  benchmark.** Amdahl: the shared per-timestep work does not thread. This gap is
  precisely why end-to-end measurement was required.
- Total numerical divergence is `4.378e-07`, entirely attributable to the
  backend swap, and consistent with the independently measured `3.835e-07`
  floor. The hoist and the threading each add exactly zero.

### The Same Ladder On The Multitone Path

Measured 2026-08-15. Multitone configuration (SPA2 + SPA background + RC
background on the same `Je=2` manifold), `N_steps=4000`, batch `25`, `repeat=3`.
The RC field's detuning column is zeros, matching
`scripts/analyze_spa2_bg_feature.py:227`, so the beat frequency varies with the
scan detuning as it does in the real analysis.

| Config | Mean | Std | vs A | vs prev | Max diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | `54.274 s` | `0.896 s` | `1.00x` | - | `0` |
| B | `34.467 s` | `0.169 s` | `1.57x` | `1.57x` | `0` |
| C | `34.927 s` | `0.039 s` | `1.55x` | `0.99x` | `1.701e-04` |
| D | `33.370 s` | `0.042 s` | `1.63x` | `1.05x` | `1.701e-04` |
| E | `15.741 s` | `0.040 s` | `3.45x` | `2.12x` | `1.701e-04` |

Step-by-step paired verdicts:

| Step | Change | Delta | `95%` CI | Ratio | Verdict |
| --- | --- | ---: | --- | ---: | --- |
| A to B | BLAS pinning | `-19.807 s` | `[-21.707, -17.907]` | `1.575x` | improvement |
| B to C | numpy backend | `+0.460 s` | `[+0.054, +0.867]` | `0.987x` | **regression** |
| C to D | hoist | `-1.557 s` | `[-1.652, -1.462]` | `1.047x` | improvement |
| D to E | threading | `-17.629 s` | `[-17.743, -17.515]` | `2.120x` | improvement |

Findings:

- Total `3.45x` versus `4.69x` on the shared-slow path. The gap is threading
  (`2.12x` versus `2.53x`): multitone doubles the shared per-timestep work by
  rotating upper and lower triangles separately, and that portion does not
  thread. Amdahl, as predicted.
- **The numpy backend is a measurable regression of about `1.3%`**, not a tie.
  The shared-slow run compared it only against A, which hid this. It remains the
  correct change because it is the only backend that releases the GIL, but it
  should be presented as a cost accepted to unlock threading, never as free.
- The hoist reproduces at `1.047x` against `1.06x` on the shared-slow path.
- Max difference is `1.701e-04` here versus `4.378e-07` on the shared-slow path.
  This is a property of the test shape, not of the multitone path: `N_steps=4000`
  is far into the under-converged regime for multitone (see below), where small
  eigensolver differences amplify. It has not been measured at production
  `N_steps`. A regression-test tolerance therefore cannot be a single global
  number.

### Cascade Detunings Did Not Accumulate Up The Ladder — FIXED 2026-08-16

`build_rotating_frame_shift` handled the *carriers* of a multi-rung cascade
correctly and the *detunings* incorrectly. Only scans with fields on two different
excited manifolds were affected, and no committed scan had that shape — every one
of them puts all its fields on a single manifold (`[mf12, mf12_bg]`, both `Je=2`).
**No existing result changes.**

For a cascade J=0 -(w01)-> J=1 -(w12)-> J=2, a level carries the sum of every
carrier below it, detunings included:

```
J=1: -(w01 + d01)
J=2: -(w01 + d01 + w12 + d12)
```

The carriers accumulated through `omega=sum(unique_omegas)`. The detunings were
subtracted through each group's own diagonal mask, and `generate_D` only writes
entries with `J == Je`, so `d01` landed on J=1 and never reached J=2. The 1-2
coupling was then left with a residual `exp(i*d01*t)` phase in the rotating frame
instead of being static. Measured, with `f01=13.3 GHz`, `f12=26.6 GHz`,
`d01=+1 MHz`, `d12=0`:

```
J=1 shift/2pi:  -13.301 GHz  = -(f01 + 1 MHz)   correct
J=2 shift/2pi:  -39.900 GHz  = -(f01 + f12)     wrong, should be -39.901 GHz
```

`Simulator.run` was always right, because there the detuning goes into `muW_freq`
via `set_frequency` and the cumulative sum picks it up. That divergence between the
two paths is what the fix removes.

The fix subtracts each group's detuning through a *cumulative* mask covering its own
manifold and every manifold above it, mirroring the carrier sum. With a single
frequency group the cumulative mask equals the group's own mask, which is why the
single-manifold results are untouched. Ordering is now validated too: the
frame-defining fields must form a chain (`Jg == prev.Je`), since `sum(unique_omegas)`
had always assumed that silently.

Covered by three tests in `tests/test_rotating_frame.py` against a new
`cascade_setup` fixture; the load-bearing one compares a `run_microwave_scan` sweep
of the upper rung against a loop of `Simulator.run` calls with the *lower* rung held
off resonance. With `d01 = 0` that test passes even without the fix, which is
precisely why this went unnoticed.

### Multitone: Adiabatic Label Swaps, Not Non-Convergence — RESOLVED 2026-08-16

**This supersedes and corrects the two sections below.** An earlier reading of
this data claimed the multitone runs were not converging and that published
numbers might be wrong. That was wrong. The physics converges; the eigenstate
*labels* permute.

Running the worst grid cell (`det = 0.0 MHz`, prefactor `1`) at batch `1` across
a step ladder, the pairwise differences fall into groups rather than a trend:

| | `10000` | `20000` | `40000` | `60000` | `80000` | `160000` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10000` | `0` | `4.2e-03` | `4.5e-03` | `7.5e-01` | `7.5e-01` | `1.0e-01` |
| `40000` | `4.5e-03` | `1.5e-03` | `0` | `7.5e-01` | `7.5e-01` | `1.0e-01` |
| `80000` | `7.5e-01` | `7.5e-01` | `7.5e-01` | `3.0e-04` | `0` | `7.5e-01` |
| `160000` | `1.0e-01` | `1.0e-01` | `1.0e-01` | `7.5e-01` | `7.5e-01` | `0` |

Inspecting where the differences live shows they are permutations. Between
`40000` and `80000`, population `0.75` moves from eigenstate label `32` to label
`29`, with the same magnitude. Between the cluster and `160000`, `0.10` moves
between labels `14` and `15`.

Summing over each pair confirms the populations are stable:

| `N_steps` | `P(14) + P(15)` | `P(34) + P(35)` |
| ---: | ---: | ---: |
| `10000` | `0.097692` | `0.785434` |
| `40000` | `0.100170` | `0.780903` |
| `80000` | `0.100865` | `0.779941` |
| `160000` | `0.100812` | `0.780013` |

Stable to about `0.3%` across a `16x` range in step count. `reorder_evecs`
matches eigenvectors by overlap, and for near-degenerate states either
assignment is equally valid, so which label a population lands under depends on
how the beat is sampled.

**Consequence for the analysis.** Observables that index a single eigenstate are
unreliable here; observables that sum over the degenerate group are fine. As
`scripts/analyze_spa2_bg_feature.py` computes them:

| `N_steps` | `depletion` | `transferred` |
| ---: | ---: | ---: |
| `10000` | `0.902308` | `0.879206` |
| `40000` | `0.899830` | `0.876260` |
| `80000` | `0.899135` | `0.880572` |
| `160000` | `1.000000` | `0.875540` |

- `transferred` sums the target and the monitors: spread `5.3e-03` across the
  whole ladder. Robust in practice, though structurally it still reads index
  `35` and would break if `34`/`35` ever swapped.
- `depletion` is `1 - P[initial, 14]`, a single index. At `160000` the
  population sits under label `15`, so depletion reads exactly `1.000000`
  instead of about `0.90`. That is a real defect in the observable, not in the
  simulation.

**Recommended fix, at the analysis level rather than the propagator:** identify
the near-degenerate group around the initial and target states and sum over it,
or select states by overlap with the intended physical state at the final time
rather than by a tracked index. No change to `N_steps` addresses this, and the
step count does not need increasing: `20000` is comfortably converged for the
population sums.

### Label Swaps Are Crossings, Not Degeneracy — DIAGNOSED AND FIXED 2026-08-24

**This supersedes the mechanism given above, and the fix recommended with it.**
The observation was right: labels permute and population sums are stable. The
explanation was wrong, and so was the proposed remedy.

**Not near-degeneracy — a swept crossing.** Measured on the SPA2 setup with the
microwaves off, which reproduces the whole effect without any scan:

| `N_steps` | index holding the population | its `(F, mF)` | population |
| ---: | ---: | --- | ---: |
| `2000` | `14` | `F=2, mF=0` | `0.999749` |
| `10000` | `14` | `F=2, mF=0` | `0.999990` |
| `40000` | `14` | `F=2, mF=0` | `0.999991` |
| `160000` | **`15`** | `F=2, mF=0` | `0.999991` |

The physical state is `F=2, mF=0` at every step count, holding `0.99999`. Only
the *index* moves. The initial state meets its `F=1, mF=0` partner at
`t/T = 0.691` with a minimum gap of about `3.2 Hz` against a Fourier width of
`657 Hz` (transit `1521.74 us`), and the passage is diabatic. `reorder_evecs`
labels adiabatically, so the label follows one branch while the population
follows the other.

**The direction matters and is counter-intuitive: a finer timestep makes the
label *more* likely to be wrong**, because it resolves the crossing that a
coarse step jumps over. That is why `160000` is the outlier while `10000`,
`40000` and `80000` agree — not because the coarse runs were under-resolved.

This is the same mechanism a separate `J=2, mF=0` analysis of the TlF
Hamiltonian arrives at from the opposite direction: `mF=0` levels cross exactly
when `B` is perpendicular to `E`, and only `B_parallel` opens a gap, at about
`1620 Hz/G`. Here `B = (0, 0, 1e-3) T` is exactly parallel to `E`
(`benchmarks/common.py`), so 10 mG predicts a gap of order Hz — which is what is
measured. The conclusion there — that the physical mapping through such a
crossing is the diabatic one — is precisely the mismatch.

**Summing over the near-degenerate group — the fix recommended above — is
wrong.** That group is the `F=2` `mF` multiplet:

| index | `E - E14` (Hz) | `F` | `mF` |
| ---: | ---: | ---: | ---: |
| `10` | `-3.2796` | `2` | `+2` |
| `11` | `+3.2786` | `2` | `-2` |
| `12` | `-1.6397` | `2` | `+1` |
| `13` | `+1.6394` | `2` | `-1` |
| `14` | `0.0000` | `2` | `0` |
| `15` | `-14538.78` | `1` | `0` |

Those sublevels are degenerate at readout only because the field has ramped off.
They have different selection rules, and a field applied at detection separates
them, so summing over them destroys exactly the information a state-preparation
simulation exists to produce. The simulation must not bake in what a particular
detector happens to be unable to resolve.

**The fix that shipped:** identify states by quantum numbers, not by index.
`probabilities_final[..., k]` is the population in `V_fin[:, k]`, and `V_fin` was
already returned, so nothing needed to be plumbed through — the defect was one
line reading a `t=0` index against a `t=T` array.

- `state_prep.utils.eigenstate_quantum_numbers(V, QN)` gives `(J, F1, F, mF)`
  and their *spreads* per eigenvector; `select_eigenstate` picks the unique
  match. `F1` is required: `(J, F, mF)` is not unique, since `J=1` has two `F=1`
  levels. Spreads matter because `F` is badly mixed while the Stark field is on
  (the "triplet" states run `F = 2.372 +- 4.899`) and only becomes good at
  readout.
- `SimulationResult.population(...)` / `MicrowaveScanResult.population(...)`
  select by quantum numbers. Prefer them over `get_state_probability`.
- `scripts/analyze_spa2_bg_feature.py` now selects this way and stores the full
  resolved distribution (`final_populations`, `final_quantum_numbers`) instead of
  collapsing to four scalars at write time.
- Verified: at `N_steps=160000` the old path gives `depletion = 1.000000` and the
  new one gives `0.000009`, matching `10000` and `40000`. Bit-identical wherever
  the tracking was already right.

**Crossings here are pervasive, not exceptional.** The new `LabelGapTracker`
diagnostic counts rank changes per label: **56 of 64 labels cross something**,
almost all between `t/T = 0.90` and `0.96` as the Stark field collapses. So
adiabatic labels in this system are broadly unreliable, which is the argument
for quantum numbers rather than a caveat about one bad crossing.
`result.unreliable_labels()` reports them, and the tracked-index accessors warn
once per result. A bare-gap threshold was tried first and is useless — it flags
63 of 64 — because it cannot tell a static end-of-trajectory degeneracy from a
swept crossing.

**Known issue, not fixed:** `src/state_prep_gpu/_reorder.py` matches by greedy
`argmax` rather than the Hungarian assignment the CPU path uses. Upstream's own
docstring notes that greedy matching silently produces an arbitrary ordering when
two eigenvectors claim the same reference column. GPU labels may therefore differ
from CPU ones. Left alone deliberately: there is no CuPy `linear_sum_assignment`,
so fixing it needs a host round-trip or a batched auction kernel, to repair
labels that should not be load-bearing now that quantum-number selection exists.

#### Grid Audit Closed — `transferred` Is The One That Breaks, 2026-08-24

Run with `scripts/audit_multitone_labels.py --n-steps 20000 40000 80000
--workers 8`, which reads every observable twice: once through a tracked `V_ini`
index, as the analysis used to, and once by quantum numbers. A swap shows up as
the two readings disagreeing.

| `N_steps` | elementwise | `depletion` (qn) | `depletion` (idx) | `transferred` (qn) | `transferred` (idx) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `20000` | `1.749e-02` | `2.546e-03` | `2.546e-03` | `2.676e-03` | `1.629e-02` |
| `40000` | `1.749e-02` | `1.438e-03` | `1.438e-03` | `1.183e-03` | `1.866e-02` |

Referenced to `80000`. Tracked indices disagreeing with the quantum-number
selection: **none** at `20000`, **none** at `40000`, **`monitor0`** at `80000`.

**First, a correction to how the question was posed above.** "Do the other 24
grid cells swap?" is not well formed. The batched path shares one slow
eigenbasis across the whole scan, so there is a single `V_fin` and a single set
of indices per run. A swap is a property of the *trajectory and step count*, not
of a detuning. The right question is which step counts swap, and the answer is
`80000`.

**`depletion` is clean.** The two readings are identical to every digit at both
step counts, so the `2.573e-03` residual is ordinary convergence error and not a
labelling artefact. No cell reads `1.000000`. The `det=0.0, pref=1` cell gives
`0.899910 / 0.899830 / 0.899135`, matching the earlier ladder.

**`transferred` is not clean, and it was the observable previously called
robust.** At `80000` the monitor label swaps and `transferred` inherits it: read
through tracked indices the error is `1.866e-02`, read by quantum numbers it is
`1.183e-03` — a factor of `16`. Most of the flat `1.749e-02` elementwise
"non-convergence" is this swap rather than physics, which is why the elementwise
metric refused to improve with step count.

**The pair involved is a known one.** `monitor0` is `J=2, F1=5/2, F=2, mF=0` and
the target is `J=2, F1=5/2, F=3, mF=0` — exactly the two branches picked out by
the separate `J=2, mF=0` analysis noted above, which shows they cross exactly at
`0.349 kV/cm` when `B` is perpendicular to `E` and are separated only by
`B_parallel` at about `1620 Hz/G`. That analysis and this one arrived at the
same crossing from opposite directions.

The swapped indices are **`32` -> `29`**, which is exactly the pair the original
single-cell diagnosis named ("population `0.75` moves from eigenstate label `32`
to label `29`"). That observation was correct and is reproduced here
independently; only its explanation — near-degeneracy rather than a crossing —
was wrong.

**Consequences.**

- Published figures at `N_steps` of `10000` or `20000` are unaffected: no swap
  occurs there, by either observable.
- `transferred` must be read by quantum numbers, not by index `35` plus tracked
  monitors. `scripts/analyze_spa2_bg_feature.py` already does this.
- The engine's `monitor_probabilities_final` still uses tracked indices and so
  still carries this defect; the accessors now warn, but the stored values are
  unchanged by design. Anything reading them at `80000` steps or finer should
  use `population(...)` instead.
- `20000` remains comfortably converged **against label swaps**, which is all
  this subsection measured. The step count never needed increasing to avoid a
  swap; raising it is what exposed one.

  > **Corrected 2026-08-29.** Do not read that as "`20000` is converged". For the
  > RC-background cases it is not even close: the beat runs at `5.4-8.9 MHz` over
  > a `1.52e-03 s` trajectory, so `beat*dt` is `2.6-4.3` rad per step and the
  > frozen propagator aliases it. `depletion` at `20000` sits `~1.7e-03` from a
  > converged answer. This sentence is why the SPA2 analyses were produced at
  > `20000`; they have since been regenerated at `160000` with
  > `--propagator magnus`. A convergence statement is only ever about the
  > quantity that was measured.

Full per-cell output in `results/multitone_label_audit.json`.

#### Superseded: the three questions this closed

All three are answered by the audit above. Kept for the reasoning, and because
the framing of the first one is instructive: it assumed labels are per-cell when
the batched path shares one eigenbasis across the scan.

The diagnosis above rests on **one grid cell** (`det = 0.0 MHz`, prefactor `1`),
chosen because it was the worst in the grid. It explains that cell cleanly, but
three things are unverified:

1. **Do the other 24 grid cells swap too, and at which step counts?** The swap
   partners seen here were `29`/`32` and `14`/`15`. Other detunings may have
   different near-degenerate pairs, or none.
2. **Are the published SPA2 background figures affected?** *Partly answered
   2026-08-24.* Both saved analyses (`results/spa2_bg_rc_analysis.json` and
   `results/spa2_right_bg_rc.json`, 363 points) were screened for the signature
   — `depletion` jumping toward 1 while `transferred` stays put — and are clean:
   no `depletion` is exactly `1.0`, and where it reaches `0.99999` `transferred`
   reaches `1.0000` alongside it, so the population really did leave. The only
   nonzero `|depletion - transferred|` gaps are in the `zy` family at *low*
   depletion (`0.31` vs `0.24`), smooth across the family rather than a
   single-point discontinuity, which reads as leakage into unmonitored states.
   This is a screen, not a proof: it would miss a swap of a population
   comparable to that leakage. The 5x5 multitone grid is a different set of runs
   and remains unchecked.
3. **Is `transferred` as safe as it looked?** It was stable to `5.3e-03` here,
   but it reads index `35` directly and would break if `34`/`35` ever swapped.
   That pair happened not to swap in this cell.

How to check: rerun the grid and, instead of comparing `probabilities_final`
elementwise, compare `population(J=..., F1=..., F=..., mF=...)` across step
counts. A swap shows up as a large elementwise difference with a stable
quantum-number population, which is the signature already seen. (Do **not** use
the group sum suggested in the superseded section above.) The saved arrays from
this run are only for the single cell, so the grid needs rerunning:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_convergence.py --multitone --n-steps 20000 40000 80000
```

Until that is done, treat existing multitone figures as probably fine but
unverified, and prefer `transferred` over `depletion` when reading them.

### Superseded: Reading This As A Convergence Failure

#### Original (incorrect) reading


**Measured 2026-08-16 with `benchmarks/bench_convergence.py --multitone`,
`N_steps` in `{10000, 20000, 40000, 80000}`, referenced to `80000`, over the
same `5x5` grid.** This supersedes the reading in the section below, which was
based on a `40000` reference.

| `N_steps` | Max abs error vs `80000` | Improvement ratio |
| ---: | ---: | ---: |
| `10000` | `7.494e-01` | - |
| `20000` | `7.459e-01` | `1.00` |
| `40000` | `7.451e-01` | `1.00` |

The error is flat at about `0.75` and does not fall at all, while `10000`,
`20000` and `40000` agree with *each other* to about `1e-2`. Three step counts
mutually agree and the fourth gives a completely different answer, on a scale
where probabilities are bounded by `1`.

The apparent convergence recorded below was therefore an artefact of comparing
under-resolved runs to one another. The earlier statement that `20000` carries
about `5.8e-03` error was too generous; the runs are not converging.

The structure locates it. Per-cell error at `N_steps=40000`:

| Detuning | Beat | Error |
| --- | --- | ---: |
| `-2.0 MHz` | `+2.567 MHz` | `3.4e-04` |
| `-1.0 MHz` | `+1.567 MHz` | `2.6e-01` |
| `0.0 MHz` | `+0.567 MHz` | `7.5e-01` |
| `+0.5 MHz` | `+0.067 MHz` | `2.9e-01` |
| `+1.5 MHz` | `-0.933 MHz` | `1.2e-01` |

The far-off-resonant cell converges cleanly; every cell in the resonant region
does not. This is not uniform discretisation error, it is confined to where
population is actually transferred, which is the regime the analysis is about.

Two explanations fit and this data cannot separate them:

1. The resonant multitone dynamics are genuinely hypersensitive - avoided
   crossings or near-degeneracies where the adiabatic eigenvector tracking makes
   different assignments depending on how the beat is sampled. Then no step
   count converges and the observable needs rethinking rather than refining.
2. Something specific breaks at `N_steps=80000`. Less likely on its face, since
   finer `dt` should help, but three runs agreeing and a fourth diverging is also
   the signature of a threshold effect.

Discriminating test: run `N_steps=60000` and `160000`. If `60000` joins the
`10000-40000` cluster and `160000` joins `80000`, there is a transition to
locate. If the answers keep jumping, it is hypersensitivity.

This is a physics question, not a performance one, and it should be settled
before the multitone analysis is trusted. Note that
`scripts/analyze_spa2_bg_feature.py` reduces to depletion and transfer rather
than raw `probabilities_final`, so the effect on any published quantity needs
checking against those reductions specifically.

### Superseded: Multitone Convergence Against A `40000` Reference

`scripts/analyze_spa2_bg_feature.py:380` doubles `N_steps` to `20000` when the
RC background is included. Testing whether that was conservative, over a `5x5`
grid of scan detuning by intensity prefactor (Rabi `1x` to `4x`), referenced to
`N_steps=40000`:

| `N_steps` | Max abs error | Worst cell |
| ---: | ---: | --- |
| `5000` | `6.257e-01` | `det=-1.0 MHz, pref=16` |
| `10000` | `1.496e-02` | `det=-1.0 MHz, pref=16` |
| `20000` | `5.763e-03` | `det=-1.0 MHz, pref=16` |

Conclusions:

- The doubling is **justified, not conservative**. There is no `2x` available
  here. `N_steps=10000` carries about `1.5%` error in the worst cell.
- `20000` may itself be insufficient. Convergence degrades from `42x` improvement
  (`5000` to `10000`) to `2.6x` (`10000` to `20000`), below the `4x` expected for
  second order. That is the signature of the `40000` reference not being
  converged either, so `5.763e-03` is a lower bound on the error at `20000`, not
  an estimate of it. Settling this needs an `N_steps=80000` run.
- Required step count does **not** track beat frequency. Beat runs from
  `+2.567 MHz` at `det=-2.0` to `-0.933 MHz` at `det=+1.5`, yet the worst cell is
  `det=-1.0` and both extremes are among the best converged. It appears to track
  where the drive is resonant and the dynamics are strongest.
- Coupling dependence is non-monotonic: error rises with prefactor at
  `det=-2.0, -1.0, +1.5`, but peaks at `pref=4` and falls at `det=0.0, +0.5`.
- Practical consequence: no single parameter point can certify a step count for
  this path. Any future convergence claim needs the grid.

### What Is Proven Versus What Is Ready To Ship

Proven: the gains above are real, isolated, and reproducible.

Applied:

- **BLAS pinning**, via a `threadpoolctl` context manager scoped to the
  evolution loops (`limit_blas_threads` in `src/state_prep/simulator.py`).
  `blas_threads` defaults to `1` on both `run()` and `run_microwave_scan()`, is
  threaded through the loky workers, and accepts `None` to leave BLAS alone.
  `threadpoolctl>=3.5.0` is now a hard dependency; a missing install is an
  `ImportError`, not a silent slowdown. Verified: limit applied inside the
  context (`[16,16]` to `[1,1]`) and restored on exit, `1.572x` end to end with
  agreement exactly `0.000e+00`.
- **The hoist**, to both batched paths. See Priority B above.

Closed since:

- **Threading: rejected.** The loky path already in the repository beats it at
  matched worker counts (`3.292x` versus `2.491x` at 8). See Priority C.
- **`eig_backend` default: leave as `zheevd`.** Enabling threading was its only
  justification.
- **Loky: keep and recommend.** `workers=8` gives `3.292x` with bitwise
  identical output, once BLAS is pinned. The `run_microwave_scan` docstring now
  carries the measured guidance.

Still open:

- No size-based crossover for loky. It is `0.29x` on small scans and nothing
  guards against that; the crossover has not been measured across `N_steps` and
  batch.
- A regression-test tolerance cannot be a single global number: `4.378e-07` in
  the converged shared-slow regime, `1.701e-04` in under-converged multitone.
- The multitone convergence question: `N_steps=20000` may be insufficient, and
  settling it needs an `N_steps=80000` reference run.

## Performance Priority A: Interaction-Picture Split-Step Propagator

**Status: measured and rejected in this form, 2026-08-15.** The proposal below
is kept because the diagnosis points at a specific alternative; the measurement
is recorded first.

### Measurement

Built the real rotating-frame Hamiltonian at production parameters and compared
the exact one-step propagator against the Strang-split one directly, rather than
bounding the commutator. SPA2 setup, `n=64`, `T=1.522e-03 s`, `9` sample times
along the trajectory, spectral norm of `U_exact - U_split`:

| `N_steps` | `dt` | Per-step Strang error | Times `N_steps` |
| ---: | ---: | ---: | ---: |
| `10000` | `1.522e-07 s` | `1.392e-03` | `1.39e+01` |
| `20000` | `7.609e-08 s` | `2.188e-04` | `4.38e+00` |
| `40000` | `3.804e-08 s` | `1.032e-04` | `4.13e+00` |

At the production `N_steps=10000` the accumulated error exceeds the trivial
bound `||U - U'|| <= 2`, i.e. the propagator is completely wrong, against a
measured numerical floor of `3.835e-07`.

### Why It Fails, And Why The Original Reasoning Was Wrong

The favourable ratio is real but irrelevant:

- max diagonal spread: `5.0276e+11 rad/s` (rotational manifold splittings)
- max `||H_mu_rot||_2`: `2.8362e+05 rad/s`
- ratio: `5.641e-07`

The coupling *is* six orders of magnitude smaller than the diagonal. But Trotter
error is not governed by that ratio, it is governed by `delta * dt`, which here
is about `7.65e+04` radians per step. The diagonal phase wraps roughly `12000`
times per timestep. The commutator `[delta, H_mu]` carries the large factor, so
splitting fails no matter how small the coupling is.

This is precisely the risk flagged in the original write-up, now quantified. The
existing algorithm needs a full eigensolve because it must resolve GHz-scale
energy differences exactly over `150 ns` steps, and it does.

### The Surviving Variant, Not Yet Checked

The diagnosis points at handling the diagonal *exactly* rather than splitting it:
move to the interaction picture with respect to `delta`, where the propagator
over one step involves only `H_I(s) = exp(i*delta*s) H_mu exp(-i*delta*s)`, with
`||H_mu|| * dt` about `0.043` radians. The oscillatory integral in the first
Magnus term is analytic elementwise:

```
integral_0^dt exp(i*(delta_i - delta_j)*s) ds = (exp(i*(delta_i-delta_j)*dt) - 1) / (i*(delta_i-delta_j))
```

This needs no eigensolve and no expansion of the large diagonal. Whether the
neglected second Magnus term is small enough is a separate question and needs
its own check before any implementation. Do not start writing this until that
check is run.

Observation:

The per-scan-point eigensolve at `src/state_prep/simulator.py:1018` exists only
to propagate the state vectors. Adiabatic tracking, probabilities, and monitor
states all use the *shared slow* eigenvectors, which are already computed once
per timestep at `src/state_prep/simulator.py:986`.

The matrix being diagonalized has favourable structure:

```
H_rot = H_mu_rot(t) + diag(D + D_mu_diag_batch[b])
        ^ small, dense    ^ large, diagonal, batch-dependent
```

- The diagonal part carries the slow eigenvalues and detunings, spanning GHz.
  Its exponential is exact and free: it is a vector of phases.
- The dense part is the microwave coupling, on the order of kHz to MHz.

Potential implementation:

- Strang splitting: `U ~ exp(-i*diag*dt/2) @ exp(-i*H_mu*dt) @ exp(-i*diag*dt/2)`.
- Only the small microwave term needs expansion, and it is well conditioned for
  a low-order Taylor or Chebyshev series.
- Cost per scan point drops from a full `n^3` eigensolve to a few `(n,n)@(n,S)`
  matmuls, where `S` is the number of initial states (about `4`). That is
  roughly `n^2*S` instead of order `10*n^3`.

Risks that must be validated before adopting:

- This is a Trotter splitting and introduces `[diag, H_mu]*dt^2` commutator
  error. The current scheme is exact for piecewise-constant `H`, though it
  already carries `O(dt^2)` time-ordering error, so the convergence order does
  not degrade, but the error constant may grow.
- The commutator is largest for far-off-resonant manifold couplings, which are
  exactly the terms generating the AC Stark shifts of interest.
- Larger `N_steps` may be required, which would eat into the gain.

Validation:

- Compare final probabilities against the current `run_microwave_scan(...)` for
  a small SPA2-like scan, at several `N_steps`, and confirm the difference
  converges at the expected rate.
- Require agreement at the level already used for the CPU/GPU comparison
  (`4.228e-06` or better on the real SPA2 benchmark).

This same change applies to the GPU path, where the batched rotating-frame
`cp.linalg.eigh` at `src/state_prep_gpu/simulator.py:175` and
`src/state_prep_gpu/simulator.py:341` is the likely explanation for the real
SPA2 GPU result being only `1.06x` while the synthetic slow-H benchmark reached
`3.46x`. Batched small eigensolves are the worst case for a GPU; batched matmuls
are the best case. This supersedes Priority 4 of the existing backlog by
proposing a specific cause and fix.

## Performance Priority B: Hoist The Basis Change Out Of The Batch Loop

**Status: APPLIED 2026-08-15**, to both batched paths
(`_time_evolve_mu_batched_shared_slow` and
`_time_evolve_mu_batched_shared_slow_multitone`). `_time_evolve_mu` is
deliberately unchanged: it has no batch loop, so `V @ V_rot` already runs once
per timestep there and the round trip would cost the same.

Verification of the applied code:

- Against outputs captured from the pre-edit code, over both batched paths and
  all of `probabilities_final`, `monitor_probabilities_final`, `psis_final`:
  worst difference `5.573e-14`. Exact, as the factorization requires.
- Paired A/B of the shipped implementation against the original inner loop,
  `N_steps=2500`, batch `20`, `repeat=5`: `16.2930 s` to `15.3764 s`,
  `1.060x`, CI `[-1.0148, -0.8185]`, agreement `6.972e-14`.

### Measurement

SPA2 setup, `N_steps=4000`, batch `25`, `repeat=5`, `warmup=1`, interleaved,
prototype applied by monkeypatch so both variants ran identical code apart from
the change under test.

| Variant | Mean | Std | Best |
| --- | ---: | ---: | ---: |
| baseline | `55.2840 s` | `0.3677 s` | `54.7619 s` |
| hoisted | `50.5538 s` | `0.3128 s` | `50.2784 s` |

- Paired delta: `-4.7301 s +/- 0.1904 s`
- `95%` CI: `[-5.2585 s, -4.2017 s]`
- Speedup: `1.094x`, `-8.6%` runtime
- Verdict: **improvement**
- Agreement: `1.331e-13`, confirming the transformation is exact

Also passes the conservative unpaired check: the `4.73 s` gap exceeds the
`0.68 s` summed spread, so significance does not depend on the paired design.

The `8.6%` sits just below the estimated `10%`. That is consistent with an
eigensolve costing roughly `10-20 n^3` flops against a complex matmul's `8 n^3`,
so `V @ V_rot` was somewhat cheaper than the eigensolve beside it. No anomaly.

### Applying It

The same factorization applies to the multitone path at
`src/state_prep/simulator.py:1163`, which has identical structure and is
unmeasured but follows the same argument.

It does *not* help `_time_evolve_mu` at `src/state_prep/simulator.py:1442`:
that path has no batch loop, so `V @ V_rot` already runs once per timestep and
the round trip would cost the same.

Expect the relative gain to grow once the eigensolve is made cheaper by
Priority C, since the absolute saving stays roughly constant while the total
shrinks.

### Original Rationale

Not present in the existing backlog. Exact, low risk, independently useful.

Current code:

- `src/state_prep/simulator.py:1026`: `A = V @ V_rot` runs inside the batch
  loop. This is a full `n^3` matmul per scan point per timestep, the same order
  as the eigensolve beside it.
- The same pattern appears at `src/state_prep/simulator.py:1163` (multitone
  path) and `src/state_prep/simulator.py:1442` (`_time_evolve_mu`).

Why it is avoidable:

`V` is the shared slow eigenbasis and is identical across the batch. It does not
need to enter the inner loop at all.

Potential implementation:

- Rotate `psis_batch` into the slow eigenbasis once per timestep, batched, at
  cost `B*S*n^2`.
- Propagate inside the loop using `V_rot` alone.
- Rotate back once per timestep.

Properties:

- Numerically exact. No change to results, so it can be validated by requiring
  bitwise-comparable output rather than a tolerance.
- Estimated *unmeasured* gain around `10%` on its own, since it removes one of
  the two `n^3` operations in the inner loop while the eigensolve remains.
- The gain grows substantially if Priority A lands, because removing the
  eigensolve makes this matmul the dominant remaining cost.

## Performance Priority C: Thread The Batch Loop Instead Of Using Loky

**Status: REJECTED 2026-08-15. Do not implement.** A controlled head-to-head at
matched worker counts shows the existing loky path beats threading. The
threading case below rested on an isolated eigensolve microbenchmark that did
not survive end-to-end measurement.

### Head-To-Head, Each Approach At Its Best

`N_steps=4000`, batch `25`, `repeat=4`, interleaved, BLAS pinned throughout.
Loky keeps `zheevd` (separate processes each have their own GIL); threading is
forced onto `numpy` and its measured `1.3%` penalty.

| Variant | Mean | Std | vs serial | Max diff |
| --- | ---: | ---: | ---: | ---: |
| `serial_zheevd` | `30.5646 s` | `0.061 s` | `1.00x` | `0` |
| `loky4_zheevd` | `11.8962 s` | `0.878 s` | `2.569x` | `0` |
| `loky8_zheevd` | `9.2846 s` | `0.982 s` | **`3.292x`** | `0` |
| `thread4_numpy` | `13.0924 s` | `0.119 s` | `2.335x` | `4.378e-07` |
| `thread8_numpy` | `12.2686 s` | `0.035 s` | `2.491x` | `4.378e-07` |

Direct paired comparison at matched worker count:

| Comparison | Delta | `95%` CI | Ratio | Verdict |
| --- | ---: | --- | ---: | --- |
| `loky8` vs `thread8` | `-2.984 s` | `[-4.501, -1.467]` | loky `1.321x` faster | improvement |
| `loky4` vs `thread4` | `-1.196 s` | `[-2.699, +0.306]` | loky `1.101x` faster | unproven |

Loky is also bitwise identical to serial (`0.000e+00`), because it keeps
`zheevd`. Threading carries the `4.378e-07` backend shift, reproducing the
figure measured independently in the ladder.

### Restructuring The Inner Loop Does Not Save Threading

Tested 2026-08-15, same shape (`N_steps=4000`, batch `25`, `repeat=4`). The
hypothesis was that threading lost because the inner loop makes about ten small
NumPy calls per scan point, each holding the GIL during Python-level dispatch,
so vectorising the assembly and calling `np.linalg.eigh` once per thread on an
`(chunk,n,n)` sub-stack would cut GIL transitions roughly sixtyfold.

| Variant | Mean | Std | vs serial |
| --- | ---: | ---: | ---: |
| `serial_zheevd` | `30.5445 s` | `0.114 s` | `1.00x` |
| `loky8_zheevd` | `8.1076 s` | `0.205 s` | **`3.767x`** |
| `thread8_naive` | `12.1482 s` | `0.033 s` | `2.514x` |
| `thread8_batched` | `12.3755 s` | `0.100 s` | `2.468x` |
| `thread8_tiled` | `11.5480 s` | `0.214 s` | `2.645x` |

- Materialising the full `(B,n,n)` stack was *slower* than the naive loop
  (`12.38 s` versus `12.15 s`): streaming `1.6 MiB` past L2 cost more than the
  saved GIL transitions returned.
- Per-thread tiling recovered locality and beat the full stack by `6.7%`, but
  beat the naive loop by only `5%`.
- Loky remains `1.42x` faster than the best threaded variant.

**Correction, measured 2026-08-16.** An earlier version of this section claimed
the binding constraint was a `29%` serial fraction, obtained by inverting Amdahl
from the measured `2.645x`. That reasoning was circular: it assumed Amdahl was
the only limit and solved for the parameter that made it true. Direct
instrumentation (`benchmarks/bench_serial_fraction.py`) shows otherwise:

| Batch | Shared (serial) | Amdahl ceiling at `8` threads | Threading achieved |
| ---: | ---: | ---: | ---: |
| `5` | `31.1%` | `2.52x` | - |
| `25` | `10.0%` | `4.70x` | `2.65x` |
| `50` | `6.4%` | `5.53x` | - |

At batch `25` the serial fraction permits `4.70x`, and threading reached only
`2.65x`, so the serial fraction does **not** explain the plateau. The real limit
is GIL contention inside the batch loop, which the batched and tiled
restructurings relieved by only about `5%`. The original hypothesis was right
about the mechanism and wrong about the magnitude.

This also corroborates the `runtime(B)` fit above (`7.3%` intercept at batch
`25`) rather than the Amdahl inversion.

Breakdown of the shared `10%` at batch `25`, `N_steps=1500`:

| Section | Share |
| --- | ---: |
| `microwave_rotate` | `3.3%` |
| `slow_eigensolve` | `3.2%` |
| `psis_rotate_in` + `psis_rotate_out` | `2.0%` |
| `reorder_evecs` | `0.9%` |
| `h_slow_eval` | `0.8%` |

Nothing in that list is worth optimising: the whole shared path is `10%`, and
loky already parallelises it by repeating it concurrently.

This also inverts the original reasoning about loky. Loky fits
`30.54/8 + ~4.3 s startup ~= 8.1 s`: it parallelises *everything*, including the
shared work, by repeating it redundantly per worker. The duplication first
described as loky's weakness is in fact its decisive advantage, because it
converts a serial bottleneck into parallel redundant work. It also predicts
loky's lead grows with `N_steps`, since its only real cost is fixed startup.

### Why The Threading Case Was Wrong

Two errors, both from reasoning rather than measuring:

1. **The duplicated slow diagonalization costs CPU, not wall time.** It was
   counted as pure waste, but the workers run it *in parallel*: eight workers
   each spending `3.8 s` on the shared portion still finish in `3.8 s` of wall
   clock. The old "repeats the shared slow diagonalization per chunk" framing
   overstates the cost.
2. **Threads do not get a clean run at the inner loop.** NumPy releases the GIL
   *inside* LAPACK, but the `H_rot` assembly between those calls is a sequence of
   small NumPy operations whose Python-level dispatch holds it. That serializes
   threads across a meaningful fraction of the loop. The `6.34x` eigensolve
   microbenchmark measured only the part that threads well.

Shipping on the microbenchmark would have added a threading backend slower than
the loky path already in the repository, forced an unnecessary backend switch,
and introduced a `4.378e-07` output shift in exchange for a regression.

### Consequences

- **Do not implement threading.** The pool-lifetime, API-exposure, and
  loky-coexistence questions are all moot.
- **Do not change the `eig_backend` default.** Enabling threading was its only
  justification. `zheevd` stays, so the `1.3%` penalty and the output shift do
  not arise.
- **Loky's historical numbers were a BLAS oversubscription artifact.** The
  `0.27x` to `1.64x` range was measured with BLAS unpinned, i.e. workers times
  16 OpenBLAS threads on 16 cores. With the pinning now shipped, the same code
  reaches `3.292x` at `workers=8` with no further change.
- Remaining unknowns are documentation, not design: loky's variance is high
  (`0.982 s` against threading's `0.035 s`, from process startup), it is still
  catastrophic on small scans (`0.299x` at `N_steps=250`, batch `8`), and the
  best worker count at other batch sizes is unmeasured.

### Serial Fallback That Already Exists

`workers=1` is the default and runs the serial path. Two guards fall back to it:

- `src/state_prep/simulator.py:725`: loky is only used when `workers != 1` *and*
  `batch > 1`.
- `_run_microwave_scan_parallel_loky`: `n_jobs = min(effective_n_jobs(workers),
  batch)`, and `n_jobs <= 1` recurses with `workers=1`.

Both guards key on a degenerate batch. There is **no size-based crossover
heuristic**, so `workers=8` on a short scan still pays the loky penalty. Adding
one would need the crossover measured across `N_steps` and batch first.

### Superseded: The Original Threading Case

The measurements below stand as isolated microbenchmarks; they simply do not
predict end-to-end behaviour.

### Full Eigensolver Grid

Measured 2026-08-15. Microseconds per eigensolve, `repeat=3`, `16` CPUs. `proc4`
generates matrices inside each worker, so compare backends within that column,
not across columns.

`n=64`:

| Backend | BLAS | serial | thread4 | thread8 | proc4 |
| --- | --- | ---: | ---: | ---: | ---: |
| `zheevd` | `1` | `285.2` | `292.6` | `292.0` | `216.2` |
| `np_eigh` | `1` | `288.5` | `77.0` | `43.9` | `105.9` |
| `sp_eigh_default` | `1` | `343.8` | `368.0` | `390.2` | `114.6` |
| `sp_eigh_evr` | `1` | `344.5` | `367.0` | `373.9` | `121.1` |
| `sp_eigh_evd` | `1` | `290.0` | `311.5` | `317.9` | `103.3` |
| `zheevd` | default | `494.3` | `465.3` | `437.4` | `275.9` |
| `np_eigh` | default | `489.1` | `145.6` | `86.8` | `158.0` |

`n=100`, serial column only:

| Backend | BLAS `1` | BLAS default |
| --- | ---: | ---: |
| `zheevd` | `779.1` | `5514.2` |
| `np_eigh` | `764.5` | `5768.1` |
| `sp_eigh_default` | `894.4` | `5866.8` |
| `sp_eigh_evd` | `781.8` | `5456.8` |

### Why `zheevd` Was Originally Chosen, Resolved

`sp_eigh_default` and `sp_eigh_evr` are the same number (`343.8` versus
`344.5`), confirming SciPy's `eigh` defaults to the `evr` driver, which is about
`20%` slower than `zheevd`. Choosing `zheevd` over `scipy.linalg.eigh` was
correct and remains correct.

It does not extend to `np.linalg.eigh`, which uses the `evd` driver internally
and ties `zheevd` in every serial cell. The `proc4` column also shows no `zheevd`
advantage under loky: it is `2x` *slower* at `n=64` (`216.2` versus `105.9`) and
a tie at `n=100`.

### Unpinned OpenBLAS Is A Large Unforced Loss

| Shape | BLAS `1` | BLAS default | Penalty |
| --- | ---: | ---: | ---: |
| `n=64` serial | `285-288` | `489-494` | `1.7x` |
| `n=100` serial | `765-781` | `5514-5768` | `7.2x` |

This applies to every backend uniformly. No thread environment variables are set
in this venv, and both NumPy and SciPy link the same `scipy-openblas 0.3.30`, so
production runs unpinned. Notebooks using `Js = [0,1,2,3,4]` (`n=100`) are paying
roughly `7x` on every eigensolve for nothing.

Pinning BLAS to one thread is a configuration change with no algorithmic risk and
should be validated end to end before anything else in this file.

### Combined Headroom

Current default (`zheevd`, unpinned, serial) versus best measured
(`np_eigh`, BLAS `1`, `8` threads):

- `n=64`: `494.3` to `43.9` us, `11.3x`
- `n=100`: `5514.2` to `120.2` us, `45.9x`

This is the eigensolve in isolation. End-to-end gain will be smaller, but
eigensolves are about `75%` of the inner loop and the inner loop is `93%` of
runtime at batch `25`, so the ceiling is large. Measure it; do not assume it.

### Earlier Threading Measurement

`n=64`, `1500` independent eigensolves, BLAS pinned to one thread, `16` CPUs,
`repeat=5`:

| Threads | `zheevd` speedup | `numpy.linalg.eigh` speedup |
| ---: | ---: | ---: |
| `1` | `1.00x` | `1.00x` |
| `2` | `0.96x` | `1.93x` (`96%` efficiency) |
| `4` | `0.96x` | `3.75x` (`94%` efficiency) |
| `8` | `0.96x` | `6.34x` (`79%` efficiency) |

The current default eigensolver is the one that cannot be threaded.
`scipy.linalg.lapack.zheevd` is an f2py-generated wrapper that holds the GIL;
NumPy's linalg ufuncs release it. Single-threaded the two are indistinguishable
(`0.4223 s` versus `0.4284 s`), which the paired scan benchmark independently
confirmed at the whole-scan level.

### Consequence

- Switching the scan default to `eig_backend="numpy"` costs nothing measurable
  single-threaded and unlocks near-linear thread scaling of the batch loop.
- The batch loop body is eigensolve plus BLAS matmuls, and NumPy releases the
  GIL for both, so the whole inner loop should thread, not just the eigensolve.
- This requires no change to the numerics beyond the backend swap, whose effect
  is the already-measured `3.835e-07`.
- Expected gain at realistic batch sizes is far larger than anything else in
  this file, and unlike Priority A it is not an algorithmic risk.

Revises the existing backlog, which concluded only that `workers=1` should stay
the default.

Measured behaviour of the current `loky` path:

- `N_steps=120`, batch `4`: `0.27x` versus serial, i.e. `3.7x` slower.
- `N_steps=120`, batch `16`: `0.61x`.
- `N_steps=500`, batch `16`: `1.64x`.

Cause:

- `_run_microwave_scan_parallel_loky` pickles `self`, which carries
  `centrex_tlf` objects, field callables, and closures.
- Each chunk repeats the shared slow diagonalization, discarding the main
  advantage of the shared-slow design.

Potential implementation:

- The inner loop is almost entirely LAPACK. `zheevd` and `numpy.linalg.eigh`
  release the GIL, so a threading backend can parallelize the batch loop
  directly.
- Threads share the slow eigendecomposition instead of recomputing it per chunk,
  and require no pickling.
- `parallel_backend` currently hard-rejects anything but `"loky"`
  (`src/state_prep/simulator.py:610`); this would add `"threading"`.

Caveats:

- BLAS threads must be pinned to one per worker thread, e.g. with
  `threadpoolctl`, to avoid oversubscription.
- At `n=64` most BLAS builds already run single-threaded, so the risk is
  moderate, but it must be checked on this machine.

Validation:

- Extend `benchmarks/bench_microwave_scan.py` with a backend sweep alongside the
  existing worker sweep, at the shapes where loky currently loses.

## Performance Priority D: Cache The Slow Time Series Across Scans

Broader than Priority 9 of the existing backlog, which caches only microwave
coupling matrices.

Observation:

A notebook session that runs a power scan, then a detuning scan, then a
background-intensity scan over the same trajectory, fields, and `Js` recomputes
per timestep, every time:

- `H_slow(t)` construction
- the shared slow eigensolve
- `reorder_evecs`
- the microwave rotation `Vh @ H_mu_t(t) @ V` at
  `src/state_prep/simulator.py:1004`

That is the entire shared portion of the runtime, repeated per scan.

Potential implementation:

- Cache keyed on trajectory, electric field, magnetic field, `Js`, and
  `N_steps`.
- In-memory for a session; optionally on disk for notebook restarts.
- Makes the second and subsequent scans in a session substantially cheaper
  without changing any per-scan-point work.

This is also an ease-of-use improvement, since it removes the incentive to
manually restructure notebooks around a single scan call.

## Performance Priority E: Cheaper Timesteps

**Read this section first.** What follows below it is a chronological record that
includes conclusions later overturned. Every claim here carries the class of
evidence behind it, because in this investigation the evidence class -- not the
number -- was what decided whether a conclusion survived.

### A naming hazard, now fixed

`exact` used to name two unrelated things, and the collision misled readers
repeatedly:

- **the reference** -- `U_exact`, `Model.exact_states`, the closed-form solution;
- **a propagator option** -- which freezes `H` at the sample point and
  exponentiates *that frozen matrix* exactly. The word described the matrix
  exponential, not the evolution.

So a column headed "exact error" meant *the error of the frozen propagator*, not
*the exact error*, and it is entirely normal for Magnus to beat it. Both are
second-order approximations of the same time-ordered exponential and both
converge to the same closed form; to reach `1e-03` in population on the multitone
model the frozen propagator needs `256,000` steps against Magnus's `32,000`.

**The option is `propagator="frozen"`**; `PROPAGATORS == ("frozen", "magnus")`.
It was called `"exact"` throughout the investigation and was renamed before it
ever shipped, so there is no alias and nothing to migrate -- `"exact"` is simply
rejected. In prose below, "frozen propagator" means `propagator="frozen"`, and
"exact" always means the closed-form reference.

This was not cosmetic. The name talked this investigation into writing "multitone
must fall back to the exact propagator" into a plan as a *safety rail* -- fencing
off the more accurate path in favour of the one whose name sounded authoritative.

### Evidence classes

| class | meaning | trustworthy? |
| --- | --- | --- |
| **exact** | closed-form solution, or a fine product verified against one | yes |
| **geometry** | a property of the grid or the fields; no propagation | yes |
| **timing** | wall clock, interleaved, with repeats | yes |
| **theory** | arithmetic on a measured profile | as an arithmetic bound |
| **cross-scheme** | reference from a *different* discretisation | usually |
| **self-referenced** | reference from a finer run of the *same* scheme | **no -- biased** |
| **pointwise** | populations compared at a fixed detuning | **no -- slope-dominated** |

Two classes produced every wrong conclusion in this document. **Self-referenced**
comparisons flatter the scheme the reference came from: scoring against `U128`
made graded look worse, then scoring against `G256` -- itself graded -- made it
look `10-100x` better, and both were the same mistake with the sign flipped.
**Pointwise** comparisons near a steep lineshape measure the slope: at
`det=-1.0` `d(population)/d(detuning)` reaches `6e-03` per kHz, so a `0.2 kHz`
effective error reads as `1e-03` of population and refuses to converge.

### Verdicts

| question | verdict | evidence |
| --- | --- | --- |
| Is midpoint second order? | **Yes**, `2.00` measured; left-endpoint `1.00` | exact |
| Why does it not look that way in production? | `delta*dt >> 1` **and** non-commuting `H`: the neglected terms are time-ordering commutators carrying powers of `delta*dt`. Commuting `H` keeps order `2` at any scale | exact |
| Is a convergence order measurable at production settings? | **No.** `delta*dt ~ 7.65e+04`; reaching `~1` needs `N ~ 7.6e+08`. Richardson is therefore unavailable, and CF4/Filon cannot be expected to deliver formal orders either | exact + theory |
| Does midpoint help in practice? | **Yes.** On a commuting model at production `delta*dt`, left-endpoint saturates at `4e-01` with no convergence while midpoint reaches `9.0e-03` -- `45x` | exact |
| Does a graded grid help **on SPA2**? | **No -- it loses `1.8-4.2x`.** Measured on `spa_like`, whose level motion (`4.5e-05` of the spread) matches SPA2's measured `9.8e-05`, at production `delta*dt` against a closed form | exact |
| Does it help anywhere? | **Yes, where the active fraction is small**: `100-245x` on a sech pulse (`~3%` active). The ceiling is `1/f` for active fraction `f`; SPA2 is `63%` active, so `1.58x`, which is too little to pay for what grading gives up | exact + theory |
| Why does it lose by so much rather than merely failing to help? | Equidistribution *works* -- it cuts the summed local error `1.7-2.2x`. But a uniform grid's leading error telescopes to a boundary term, worth `14-18x`, and grading forfeits that. Bad trade unless `1/f` clears the cancellation ratio | exact |
| How much can grading buy at most? | `1.58x` at `p=1`, `1.39x` at `p=2` -- a bound on step *placement*, not a measured speedup | theory |
| What does grading cost? | `+0.92%` at matched `N_steps` | timing |
| Is the Magnus propagator faster? | **Yes**, and the gain grows with batch because it removes the per-scan-point `O(n^3)` eigensolve. On the **real SPA2 shape**: `2.10x` at batch `25`. On a *synthetic* model with sparser `H_mu`: `3.23x` at `25`, `1.89x` at `5`, `1.23x` at `1` -- quote the real figure, not the synthetic one | timing |
| Is Magnus as accurate? | **Yes** -- matches midpoint to four significant figures at production `delta*dt`, on two models, at every step count | exact |
| Does a graded grid break Magnus? | **No.** Identical to midpoint on every grid. But `||A|| = ||H_mu||*dt` rises with grading when the beam sits away from the density peak (`0.044 -> 0.32` on SPA2), and the Taylor guard trips near `0.5` | exact + geometry |
| Are Strang and Lie still rejected? | **Yes, now on a measurement**: `16x` worse than midpoint at production `delta*dt`, while Magnus is more accurate *and* faster. Correct implementations -- they show orders `2.00` and `1.00` in the clean regime. The old norm bound called them "completely wrong"; `16x` worse is bad but not meaningless | exact |
| Why is `det=-1.0` so hard? | Not numerics. Two resonance crossings, the near one `2.5 sigma` out in the beam flank, giving *partial* Landau-Zener transfer. The observable swings `0.177` across `100 kHz` | geometry |
| Do the four SPA2 Magnus regressions have an explanation? | **Yes, resolved: they are not a property of Magnus.** Compared at the *same* `N_steps`, where both schemes share one label path, `|frozen - magnus|` shrinks in every reported cell -- `4.69e-03 -> 6.85e-05 -> 9.85e-06` over `N = 4000, 16000, 64000` -- and tracks the control cells exactly. A systematic defect would not shrink. The original `<= 2.2e-04` differences sit below the `~5e-03` discretisation floor both schemes share at `N=4000` | cross-scheme |
| Can a fine run of either scheme settle it? | **No.** `probabilities_final` is indexed by tracked labels whose *meaning* shifts with `N_steps` (see the three rows below), so the same population can land in a different column. Comparing `N=4000` against `N=128000` returns `0.999` in cells whose physics is nearly identical | geometry |
| What does `reorder_evecs` actually do? | Keeps the ordering the run started in, by chaining each step's overlap match against the *previous* step (`V_ref = evecs`). Matching against `t=0` was considered and rejected -- see `simulator.py:707` -- because the eigenvectors rotate completely here (the SPA2 initial state is a Stark mixture `F = 2.372 +- 4.899` at `t=0`, a pure `F=2` state with the field off), so `t=0` overlaps go meaningless away from crossings | code |
| Does it hide the diabatic hop? | **No -- it displays it.** Label `k` is the continuous deformation of initial state `k`, so a population that traverses diabatically appears to move *between labels*: it keeps its character while the continuation of its own label swaps character. That is the intended reading, and it is what makes the labels useful for plotting where population goes | code |
| Then what is actually unsafe? | Only this: near a crossing the tracker does not resolve, the chained match pairs by character rather than by adiabatic branch, so **what the labels mean depends on `N_steps`**. `utils.py:126` states the direction -- a finer timestep resolves the crossing and makes the adiabatic label more likely to differ from where the population is. With `56` of `64` labels crossing, successive crossings resolve at different `N`, giving stable stretches broken by en-masse flips: `0, 16, 0, 16` swapped cells across `20000..320000`. So never compare tracked columns *across* step counts; within one run they are exactly what they claim to be | geometry |
| Do the saved `results/` need regenerating? | **Yes, and for a worse reason than sampling.** `left` vs `mid` at `N_steps=20000` moves `depletion` by `6.5e-03`, `465x` the `1.4e-05` the report quotes. But the RC cases were also *aliased* -- see below | cross-scheme |
| Was the RC-background data resolved at all? | **No.** `rc_offset=-7.4 MHz` against a `-2.0..+1.5` scan puts the beat at `5.4-8.9 MHz`, and `T = 1.52e-03 s` over `N=20000` gives `beat*dt = 2.6-4.3` rad -- most of a rotation per step. A frozen propagator does not approximate that, it aliases it; it needs `N ~ 6.4e+05` for `beat*dt ~ 0.1` | geometry |
| What does Magnus buy on the real analysis? | It integrates the beat, so `beat*dt` no longer gates it. Clean second order (`2.4e-03, 3.9e-04, 9.9e-05` over `N = 20000..160000`, ratios `6.19` then `3.94`), reaching an *extrapolated* `~2.5e-05` at `N=160000` (Richardson from the `80000/160000` pair, not measured) against the saved data's `~1.7e-03`, itself measured against a reference carrying `~1e-04` | cross-scheme + theory |
| Is the frozen propagator the exact solution? | **No.** It exponentiates a *frozen* `H` exactly; `"magnus"` exponentiates an *integrated* `H` approximately. Both are second order and converge to the same closed form. To reach `1e-03` in population on the multitone model: `"frozen"` needs `256,000` steps, `"magnus"` `32,000`. It was called `"exact"` during the investigation and renamed before shipping, after the name caused a third misreading; see the naming note above | exact |
| Is the Magnus propagator shipped? | **Yes**, `propagator="magnus"` on both entry points; `"frozen"` remains the default and stays bit-identical | timing + exact |
| Does multitone Magnus work? | **Yes**, and it is now the *more accurate* path. It was landed refusing `magnus` because freezing the beat phase is wrong; the kernel now integrates it | exact |
| Is the frozen beat phase actually a problem? | **Yes, severely.** In *populations* at `beat*dt = 0.17` rad the frozen propagator errs `1.4e-01` against Magnus's `1.7e-03`. Over `N = 4000..16000` its error does not fall at all (`2.2e-01, 1.2e-01, 1.4e-01`) while Magnus is clean order `2` | exact |
| Was the first beat kernel correct? | **No**, it anchored the beat prefactor at the midpoint while the kernel integrates from the left endpoint. The effect is subtler than first reported: with one coupled pair the stray phase is a *gauge transformation* and changes no population. It is physical only once a second field is present -- then `5e-03` at `N=4000`, falling with `dt` | exact |

### How the multitone reference is established

Worth stating explicitly, because "closed form" is a claim, not evidence, and a
wrong closed form is more dangerous than an unconverged reference -- it does not
wobble, so nothing looks suspicious.

`rotating_coupling` is checked two independent ways:

1. **It solves the ODE, verified analytically.** `U = A B` with constant `a`,
   `b`, so `i Udot U^H = a + A b A^H` in closed form -- no finite differences, no
   time stepping, no propagator. Worst residual over `t` is `4.0e-14` absolute,
   `2.7e-17` relative to `|H|`; `U(0) = I`; unitary to `4e-16`.
2. **A fine product converges to it** at ratio `4.00` per refinement, i.e. the
   independent discretisation agrees and does so at the expected order.

Check 1 is the one that matters: it shares no code with the schemes it judges.
Check 2 could in principle agree with a wrong reference if the product inherited
the same error, which is exactly how this investigation went wrong before.

The reference is good to `~1e-16` against a smallest measured error of
`6.5e-06`, so it is roughly `1e+10` from being the limiting factor. Quote that
ratio whenever a reference is used; the rule in this section exists because
several earlier tables quoted errors *below* their reference's own resolution.

### Numbers that did not survive

- **The Magnus speedup `3.23x`.** Measured on a synthetic model whose `H_mu` is
  sparser than the real one. On the SPA2 shape it is `2.10x` at batch `25`. Both
  are honest timings; the synthetic one is not the production number.
- **"Multitone must fall back to the exact propagator."** Written as a safety
  rail while the beat integral was underived, and correct at the time. It is now
  backwards: `propagator="frozen"` is the one that freezes the beat, and in
  populations it is ~`80x` worse than Magnus where `beat*dt` approaches `1` rad.
  Note how the *name* argues for the wrong choice here.

- **`10-100x` for grading, the `120x` at `N=500`, the `5.6x` effective-shift
  figure.** All self-referenced to `G256`, a graded run.
- **The `1.1-7.7x` replacement figure**, measured on a model built from SPA2's
  density *profile*. Its magnitudes were wrong by four orders: it moved the
  eigenvalues by `~0.5` of the spectral spread where SPA2 moves them by
  `9.8e-05`. Matching the shape is not enough. On `spa_like`, which matches
  both, grading **loses**.
- **Any error below `2e-04`** quoted anywhere in the older subsections. That is
  the resolution floor stated in *How Well Truth Is Actually Known*, and several
  tables quote figures one to two orders below it.
- **"Midpoint is rejected"**, in *Superseded: Midpoint Sampling Measured With The
  Broken Metric*. It is the shipped default and is `45x` better than
  left-endpoint where they differ cleanly.
- **"Why the scheme stays first order is unresolved."** It is resolved: see the
  `delta*dt` mechanism above.

### What the multitone model changed

For most of this investigation the multitone path had **no exact reference at
all** -- all six analytic models drive the single-tone loop. Its Magnus branch
was therefore scored against the frozen propagator, which agreed with it to
`1.5e-02` and looked like confirmation. It was not: both schemes were wrong, in
the same direction, and the comparison could not see it.

`rotating_coupling` in `benchmarks/analytic_models.py` closes that gap. It is the
driven two-level problem,

    H(t) = (Delta/2) sz + (Omega/2) (exp(-i*w*t) s+ + exp(+i*w*t) s-)

which is exactly a raising/lowering pair carrying a beat phase -- the structure
the multitone loop assembles -- and it is constant in the frame rotating at `w`,
so

    U(t) = exp(-i*w*t*sz/2) exp(-i[((Delta - w)/2) sz + (Omega/2) sx] t)

A fine product converges to this at ratio `4.00` per refinement, reaching
`2.283e-05` at `1.6e+06` steps, which is what licenses using it as truth.

Two ways it can be mis-parameterised, both of which produced confident wrong
readings here before being caught:

1. **Sign of the beat relative to the detuning.** `detuning - beat = -5800`
   against `rabi = 400` is counter-rotating and transfers `0.005` of the
   population. Every scheme then scores terribly against dynamics that are not
   there. `test_multitone_drive_is_near_resonance` asserts against this.
2. **Scoring amplitudes where only populations are physical.** A constant phase
   on the raising part with its conjugate on the lowering part is `R^H (.) R` for
   diagonal `R`. With a single coupled pair it telescopes across steps and
   changes nothing observable -- yet it is plainly visible in the complex
   amplitudes. Measured that way the beat-anchor defect looked like a clean
   collapse from order `2.00` to `0.99`; in populations the two anchors were
   **bit-identical**. Compare `abs(psi)**2`. A model with one coupled pair and
   one field cannot see a phase convention at all, so it cannot be used to size
   one.

The payoff was real but narrower than it first looked: a genuine
misfactorisation in shipped code, invisible to agreement between the two schemes,
and invisible *again* to a single-field model that could only express it as
gauge. Sizing it physically needs a second field, and there `rotating_coupling`
runs out -- adding a static coupling breaks the closed form, so that measurement
is differential (old anchor vs new), not against truth.

### Phase C, closed

Both open questions from the propagator plan are now answered, and the method
that answered them is the same in both cases: **stop looking for a better
reference and find a comparison that needs none.**

**C1, the four SPA2 regressions.** The plan treated this as possibly
unanswerable, because no closed form survives a `D_mu`-carrying Hamiltonian at
production `delta*dt`. That framing was the mistake -- it assumed the question
needed truth. It needed only a comparison free of the bias: run both schemes at
the *same* `N_steps`, so they share one label path and no reference is involved,
then watch the difference as `N_steps` grows. It shrinks in every cell, tracking
the controls, so the regressions are two unconverged runs differing below their
shared floor.

The first attempt at this did use a fine reference, and it failed loudly enough
to be instructive: differences of `0.999` in cells whose physics barely moves.
That is the adiabatic label tracking, not the propagator. Any comparison of
`probabilities_final` across different `N_steps` on this setup is invalid for the
same reason, which is a sharper statement than the existing warning about
comparing populations at a fixed detuning.

**C2, `results/` regeneration.** Answered by measuring sensitivity rather than
regenerating: run the saved configuration under both samplings and compare. The
observables move `243x` and `224x` the convergence the report quotes for them,
so the files are stale in substance. Note what that also implies -- a convergence
figure obtained by refining `N_steps` under one sampling measured
self-consistency, not accuracy, and understated the true discretisation error by
more than two orders of magnitude.

### The RC background was aliased, not merely stale

Regenerating `results/` turned up something bigger than the sampling default.

The RC tone is fixed while SPA2 is scanned, so the beat runs at `5.4-8.9 MHz`
across the scan. The trajectory lasts `1.52e-03 s`, so at the `N_steps=20000` the
analysis used, **the beat advances `2.6-4.3` radians per step**. That is not a
coarse approximation of the beat; it is aliasing. The report's Numerical Notes
already record that a `2500`-step run produced "a false large far-right peak" and
that `20000` removed it -- but `20000` is still deep in the same regime, so that
artifact was reduced rather than eliminated.

Two things follow, and the second is a trap worth naming:

1. **Digits in the peak table beyond the third decimal are not meaningful.** The
   saved data sits `~1.7e-03` from a converged answer while quoting values like
   `0.999877` to six decimals.
2. **Among aliased results, comparisons between them are arbitrary.** Scoring
   `left` and `mid` at `N=20000` against a reference suggested `left` -- the
   *old* setting -- was closer, which contradicts midpoint being second order.
   It was coincidence between two aliased runs. Establishing `beat*dt` first
   would have shown that no ordering there could mean anything, and it is
   cheaper than any of the runs that produced the misleading numbers.

`beat*dt` is geometry: computable from the scan configuration and the trajectory
duration, with no propagation at all. Check it before choosing `N_steps` for any
multitone scan, the way `grading_ceiling` is checked before enabling grading.

### What a reader should take away

1. **Do not reason about convergence order at production settings.** There is no
   asymptotic regime there. Rank by measured error at matched wall clock.
2. **Never reference a scheme against a finer run of itself.** Use a closed form,
   or at minimum a different discretisation, and report the reference's own
   uncertainty next to the result.
3. **Never compare populations at a fixed detuning near a steep feature.**
   Compare lineshapes; report the effective detuning shift and the residual.
4. **Report the density dynamic range beside any grading result.** Below about
   `10x` the comparison is meaningless -- a model with `1.01x` range produced a
   confident "grading never helps" that was purely an artefact of the setup.

### The chronological record

Everything below is kept in the order it was measured, including the parts later
overturned, because the *way* each conclusion failed is the reusable lesson.
Subsections superseded by the analytic-model work carry a banner.

### The Reference Is The Result

> **Superseded.** the `G256` reference is a **graded** run, so every accuracy ratio scored against it is self-referenced and flatters grading. The `1-2e-04` truth floor derived later in this section also invalidates both operands of the `3.422e-05` vs `1.446e-04` argument. Superseded by *Graded Grids Against Exact Solutions*.

**`N_steps=128000` on the uniform grid is not converged, and using it as a
reference inverts conclusions.** Candidate references on the `3x2` grid, uniform
(`U`) and graded (`G`) at `128000` and `256000` steps, pairwise `max|diff|`:

| | `U128` | `U256` | `G128` | `G256` |
| --- | ---: | ---: | ---: | ---: |
| `U128` | `0` | `1.446e-04` | `2.618e-04` | `2.458e-04` |
| `U256` | `1.446e-04` | `0` | `1.172e-04` | `1.011e-04` |
| `G128` | `2.618e-04` | `1.172e-04` | `0` | `3.422e-05` |
| `G256` | `2.458e-04` | `1.011e-04` | `3.422e-05` | `0` |

This is a **reference-free** comparison: each scheme against a finer version of
itself, with no shared arbiter.

- graded self-agrees to `3.422e-05` between `128000` and `256000`
- uniform self-agrees only to `1.446e-04`, `4x` worse at the same step counts

The graded grid is therefore the more accurate scheme at matched step count, and
`U128` sits `2.5e-04` from the converged cluster.

An earlier pass in this session scored everything against `U128` and concluded
grading was *worse* above `N_steps=16000`. That was uniform being flattered by a
reference drawn from its own error structure, which partially cancels for
uniform and not at all for graded. The effect is negligible where errors are
`1e-02` and dominant where they approach the reference's own error, i.e. exactly
where the false conclusion appeared. **Any future convergence claim here needs a
reference from a different discretisation, not a finer run of the same one.**

`G256` is the reference below, good to about `3.4e-05`.

### The Worst Cell Sets Everything, And It Is One Cell

> **Superseded.** the per-cell table is self-referenced to `G256`, and its `1e-06` to `5e-05` entries sit one to two orders below the `2e-04` truth floor stated in *How Well Truth Is Actually Known*. The qualitative point -- that one cell sets worst-case error -- stands; the numbers do not.

Per-cell error against `G256`:

| `N_steps` | uniform `det=-1.0 p4` | graded `det=-1.0 p4` | uniform `det=+1.5 p1` | graded `det=+1.5 p1` |
| ---: | ---: | ---: | ---: | ---: |
| `500` | `2.256e-02` | `1.201e-02` | `2.413e-02` | `1.316e-03` |
| `1000` | `2.194e-02` | `4.313e-03` | `5.708e-04` | `4.454e-05` |
| `4000` | `9.142e-03` | `2.529e-03` | `2.159e-05` | `1.881e-06` |
| `24000` | `3.607e-04` | `1.029e-03` | `1.714e-06` | `7.512e-06` |

Four of the six cells (`det=+0.0` and `det=+1.5`, both prefactors) converge
cleanly and monotonically under both grids, reaching `1e-06` to `5e-05` by
`24000`. **The two `det=-1.0` cells oscillate and hold the worst-case error near
`1e-03` at every step count tested.** That is the same cell the existing
convergence data names as worst, and its selected indices are stable, so this is
numerics rather than labelling.

### Grading Is More Accurate, And Still Cannot Be Cashed In

> **Superseded.** self-referenced to `G256`. The `10-100x` and `120x` figures do not survive; measured against exact solutions the gain is single digits on the SPA2 profile. Superseded by *Graded Grids Against Exact Solutions*.

On the cells that behave, grading is worth a large reduction in error at a given
step count -- `120x` at `N_steps=500` on `det=+0.0 p1`, `13x` at `1000` on
`det=+1.5 p1` . Combined with the reference-free result above, grading is
the better discretisation on every measurement that is not a single point.

The obstacle is that a scan is certified by its **worst** cell, and at
`det=-1.0` -- the partial-transfer regime where `remaining` is about `0.69` and
the physics is interesting -- the worst-cell error oscillates by more than the
difference between the two grids. Over `N_steps` from `500` to `24000` the
worst-cell ratio swings between `6.4x` in favour of grading and `3x` against,
with `graded@1000 = 4.3e-03` against `graded@1500 = 1.5e-02` -- worse with 50%
more steps.

**Those swings are the oscillation, not a property of either grid.** No
single-point comparison in this range supports a claim in either direction, and
three readings taken during this session -- a `2x` for grading, a claim that
grading was worse above `16000`, and an `8x` step saving inferred from
`graded@1000` -- were all nodes. The correct statement is that grading is more
accurate but its step-count saving is *unmeasurable* on the certifying cell, not
that it is absent.

### The Ceiling Is `1.58x`, And The Step Cap Was Hiding It

A region where `H` is constant needs no steps at all: `exp(-i H dt)` is exact for
any `dt`. So quiet stretches of the trajectory *must* tolerate long steps, and
any measurement suggesting otherwise is measuring something else. The first pass
here capped graded steps at `max_ratio=4` on the assumption that long steps
degrade the eigenvector overlap `reorder_evecs` tracks by. That cap was
**binding** -- the produced grid ran `0.28x` to `4.07x` -- so the grid was
forbidden from doing the one thing grading is for, and the assumption behind it
is wrong exactly where it matters: a region quiet enough to earn a long step is
one whose eigenvectors are barely rotating.

Sweeping the cap, worst-cell error against `G256`:

| `N_steps` | uniform | `mr=4` | `mr=16` | `mr=64` | `mr=256` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1000` | `2.194e-02` | `4.313e-03` | `1.662e-02` | `6.058e-03` | `6.133e-03` |
| `2000` | `9.883e-03` | `7.844e-03` | `9.072e-03` | `1.589e-02` | `1.590e-02` |
| `4000` | `9.142e-03` | `2.529e-03` | `5.317e-03` | `3.731e-03` | `3.796e-03` |
| `8000` | `3.547e-03` | `2.699e-03` | `2.617e-03` | `4.149e-03` | `4.197e-03` |

Selected indices stayed `(14, 35)` throughout, so the label tracking the cap was
protecting never needed protecting. Equidistribution saturates at `46.97x` the
uniform step, so `mr=256` and `mr=1024` return the same grid as `mr=64`.

**Raising the cap costs nothing and gains nothing.** Steps up to `47x` in the
quiet regions land in the same error band, which confirms the physics -- those
regions really are free -- while showing the freed budget is too small to matter.

Why it is too small, quantitatively. Median `||dH/dt||` relative to its own peak:

| `z` | share of path | median density |
| --- | ---: | ---: |
| `-80..-40 mm` | `14.3%` | `7.0e-01` to `1.8e-01` |
| `45..72 mm` | `9.6%` | `2.1e-02` |
| `72..120 mm` | `17.1%` | `4.0e-02` |
| `120..200 mm` | `28.6%` | `4.1e-03` |

Two things bound the payoff:

1. **No stretch of this trajectory is static.** The quietest is `4.1e-03` of
   peak, not zero -- the ring-electrode Stark field decays smoothly and never
   flattens within the flight.
2. **The square root destroys the leverage.** Local error goes as `g*dt^2`, so
   `1/250` the density buys `sqrt(250) ~ 16x` longer steps, not `250x`. Under
   *uncapped* allocation that `28.6%` of the path still consumes `10.7%` of the
   step budget, so deleting it entirely would save `10.7%`.

Integrating the equidistribution bound over the whole trajectory:

| order | `N_uniform / N_optimal` |
| --- | ---: |
| `p=1` | `1.58x` |
| `p=2` | `1.39x` |

So the honest figure for a graded grid on this problem is about **`1.5x`** --
real, and comparable to the `1.572x` from BLAS pinning that was adopted, but an
order of magnitude below what the `63774x` dynamic range in `||dH/dt||`
superficially suggests. It cannot be certified on the current observable because
`det=-1.0` oscillates by more than a `1.5x` change in step count produces.

This ceiling was computed in the first diagnostic of the investigation (`1.59x`
at `p=1`) and then lost track of through a series of confounded comparisons. It
was the right answer from the start.

### What `det=-1.0` Actually Is, And Why It Never Converged

It is not a numerics problem. The cell is physically ill-conditioned, and the
metric used throughout this investigation divides by that conditioning.

The Stark-shifted `J=1 -> J=2` transition frequency **sweeps down and back up**
along the flight: `+0.34 MHz` relative to the carrier at `z=-80 mm`, down to
`-3.03 MHz` near `z=-33 mm`, back to `+0.59 MHz` by `z=130 mm` and flat after.
Any detuning inside that range is therefore crossed **twice**, and the detuning
selects *where*:

| detuning | resonance crossings | field at the crossing | outcome |
| --- | --- | --- | --- |
| `+1.5 MHz` | none (above the `+0.587` maximum) | -- | no transfer; saturated at `0` |
| `0.0 MHz` | `z = -68.3 mm`, `+28.6 mm` | second is exactly the beam centre | complete transfer; saturated at `1` |
| `-1.0 MHz` | `z = -62.5 mm`, `-1.1 mm` | `8.4 sigma` and `2.75 sigma` out | **partial** transfer; unsaturated |

`det=0.0` and `det=+1.5` converge to `1e-05` and `1e-06` because they are
saturated: the observable sits against a floor or a ceiling and its derivative
with respect to everything is small. `det=-1.0` crosses resonance at
`z = -1.1 mm`, `2.75 sigma` from the beam centre, where the field amplitude is
about `15%` of peak. Weak coupling at a finite sweep rate is the intermediate
Landau-Zener regime, where transfer is steeply sensitive to coupling and sweep
rate.

Measured directly, holding the grid fixed at `N_steps=8000` and perturbing only
the detuning: `d(remaining)/d(detuning)` is `2e-03` per kHz over a `30 kHz`
window and reaches `6e-03` per kHz nearby, with the observable swinging `0.177`
across `100 kHz`. **The `1e-03` "convergence error" chased through this whole
investigation corresponds to an effective detuning error of about `0.2 kHz`** --
`0.02%` of the `1 MHz` detuning. No timestep scheme will ever drive that to
`1e-06` in population, because the observable amplifies a sub-kHz effective error
by three orders of magnitude.

A predicted Stuckelberg interference between the two crossings was **not
confirmed**: the accumulated phase implies a `~3 kHz` fringe, and a `0.5 kHz`
scan over `30 kHz` shows a smooth broad feature (dominant Fourier period equal to
the window) with only `~0.002` ripple near the sampling limit. Steep slope alone
accounts for the conditioning; no interference is needed to explain it.

### A Metric That Works

> **Superseded.** the lineshape metric itself is sound and is still the right tool, but this table's reference is a graded `N=64000` run, so the graded rows are self-referenced. Read the method, not the numbers.

Comparing populations at a fixed detuning near a steep lineshape feature
measures slope, not discretisation error. Comparing whole *lineshapes* and
reporting the best-fit **effective detuning shift**, with the residual after
alignment, is well conditioned. Over `det` from `-1.10` to `-0.90 MHz`, `21`
points, against a graded `N=64000` reference:

| `N_steps` | scheme | `max|dP|` | effective shift | residual |
| ---: | --- | ---: | ---: | ---: |
| `1000` | uniform | `2.156e-02` | `2.708 kHz` | `9.244e-03` |
| `1000` | graded | `6.904e-03` | `0.482 kHz` | `3.532e-03` |
| `2000` | uniform | `9.777e-03` | `-0.530 kHz` | `4.587e-03` |
| `2000` | graded | `1.137e-02` | `0.345 kHz` | `4.941e-03` |
| `4000` | uniform | `4.536e-03` | `0.014 kHz` | `2.971e-03` |
| `4000` | graded | `4.770e-03` | `-0.291 kHz` | `2.813e-03` |
| `8000` | uniform | `2.459e-03` | `-0.034 kHz` | `1.510e-03` |
| `8000` | graded | `2.938e-03` | `0.119 kHz` | `1.625e-03` |
| `16000` | uniform | `1.045e-03` | `-0.029 kHz` | `6.472e-04` |
| `16000` | graded | `7.200e-04` | `-0.028 kHz` | `4.915e-04` |

**The residual converges cleanly and monotonically** for both schemes --
`9.2e-03, 4.6e-03, 3.0e-03, 1.5e-03, 6.5e-04` for uniform, ratios near `2` per
doubling, i.e. first order, exactly what left-endpoint sampling predicts. The
propagator was converging correctly the entire time; `max|dP|` could not see it.

Graded is `5.6x` better in effective detuning at `N_steps=1000` (`0.48 kHz`
against `2.708 kHz`) and a wash above that on this window. Note the window lies
*entirely* inside the hard partial-transfer regime; the well-conditioned cells,
where grading wins `10-100x`, are not represented in it.

**Consequence for any future convergence work here.** A convergence check that
compares `probabilities_final` at fixed detunings will report false
non-convergence wherever a scan point sits on a steep part of the lineshape.
`benchmarks/bench_convergence.py` does exactly this, and its detuning grid
includes `-1.0 MHz`. This is also a live candidate explanation for the
unresolved multitone convergence puzzle recorded earlier in this document, whose
detunings are likewise near resonance. Before treating either as a numerical
failure, check the local `d(observable)/d(detuning)`.

### Implemented: `step_density` On Both Entry Points

Shipped as opt-in. `build_time_grid(T, N_steps, *, density=None, order=1,
max_ratio=64.0, probes=400)` and `field_variation_density(...)` are exported
from `state_prep`; `run()` and `run_microwave_scan()` take `step_density`
(`None`, `"auto"`, or a callable), threaded through the loky workers. The
evolution loops are untouched -- they already read `dt` per step -- and
`density=None` returns `np.linspace(0, T, N_steps)`, verified `array_equal`.
Results carry `time_grid` (`"uniform"`/`"graded"`).

One implementation subtlety worth keeping: the `max_ratio` floor has to be
solved for as a fixed point of `f = mean(max(w, f)) / max_ratio`. Flooring the
weight also raises the mean weight, which lengthens every step, so a floor taken
from the unfloored mean overshoots the cap -- measured `1.43x` over at
`max_ratio=2` before the fix.

End-to-end through the public API, SPA2, `3x2` grid, batch 6, `repeat=3`,
interleaved, against the `G256` reference:

| case | mean | std | worst-cell error |
| --- | ---: | ---: | ---: |
| uniform `N=10000` | `22.49 s` | `0.05 s` | `6.624e-04` |
| graded `N=10000` | `22.70 s` | `0.05 s` | `8.021e-04` |
| graded `N=6329` | `14.50 s` | `0.09 s` | `3.358e-03` |

**Grading costs `+0.92%` at matched `N_steps`**, so the extra density pass is
free in practice and the option is safe to leave on.

**The `1.58x` is a ceiling, not a delivered speedup, and this table is the
warning.** Dividing `N_steps` by `1.58` does run `1.55x` faster, but the
worst-cell error goes from `6.6e-04` to `3.4e-03` -- five times worse, not equal.
At matched `N_steps` graded is also marginally behind uniform in the worst cell
(`8.0e-04` against `6.6e-04`), inside the oscillation. The ceiling is realised on
the cells that converge cleanly, not on `det=-1.0`, and a scan is certified by
its worst cell. Anyone reducing `N_steps` on the strength of `step_density`
should confirm the reduction on the cells they actually care about, with
`benchmarks/bench_step_grid.py`.

### Which Quantity To Grade On: Three Alternatives, All Rejected

`||dH/dt||` was criticised above for grading on a quantity the propagator
already handles exactly: most of it is the *diagonal* Stark variation, it
outweighs the microwave term by about `3e+08` at the median, and the resulting
grid puts `31%` of a 10000-step budget into the Stark ramp entrance (`14%` of
the path) and only `11%` into the beam crossing. Three alternatives were built
and measured. **The criticism was wrong, and the measurement is what shows it.**

- `comm` -- `||[H, dH/dt]||_2`, the Magnus time-ordering term. In `H`'s
  eigenbasis `[H, H']_ij = (E_i - E_j) H'_ij`, purely off-diagonal, gap-weighted.
- `nac` -- `||dV/dt||_F`, the eigenbasis rotation rate, from a finite difference
  of phase-aligned eigenvectors. This is what `reorder_evecs` must track.
- `mu` -- `||dH_mu/dt||_2` alone, the beam envelope. The control that tests
  whether the error is beam-driven.

Step placement, as a share of a 10000-step budget (path share in brackets):

| monitor | max/median | `z` in `-80:-40` (`14.3%`) | `z` in `15:45` (`10.7%`, beam) |
| --- | ---: | ---: | ---: |
| `dh` | `15.6` | `31.1%` | `11.4%` |
| `comm` | `15.6` | `31.1%` | `11.4%` |
| `nac` | `37.3` | `29.4%` | `12.7%` |
| `mu` | `36395.5` | `3.2%` | `29.5%` |

Worst-cell error against `G256`:

| `N_steps` | uniform | `dh` | `nac` | `mu` |
| ---: | ---: | ---: | ---: | ---: |
| `500` | `2.413e-02` | `1.271e-02` | `6.008e-03` | `4.113e-01` |
| `1000` | `2.194e-02` | `4.313e-03` | `6.345e-03` | `1.317e-01` |
| `2000` | `9.883e-03` | `7.844e-03` | `8.038e-03` | `3.160e-02` |
| `4000` | `9.142e-03` | `2.529e-03` | `9.971e-03` | `1.510e-02` |
| `8000` | `3.547e-03` | `2.699e-03` | `1.001e-03` | `4.677e-03` |
| `16000` | `1.041e-03` | `1.359e-03` | `9.080e-04` | `1.558e-03` |

Three results:

1. **`comm` is redundant with `dh`.** The two grids differ by `6.7e-09 s`, under
   `5%` of one uniform step. The dominant energy differences are the rotational
   splittings, which are near-constant along the trajectory, so the commutator is
   essentially proportional to `||H'||` and equidistribution -- which sees only
   relative density -- returns the same grid. There was never a second monitor
   here.
2. **`nac` is not clearly better than `dh`.** Best at `500`, `8000` and `16000`,
   worse at `1000` and `4000`. Within the oscillation, which is the same
   amplitude as the differences between schemes.
3. **`mu` is far worse than everything, and this is the informative result.**
   Concentrating on the beam and starving the Stark ramp is catastrophic:
   `4.1e-01` at `N_steps=500`, worse than uniform at *every* step count tested.
   Per-cell at `4000` it degrades every cell to about `1e-02`, including the easy
   `det=+1.5` cells that reach `2e-05` under uniform and `2e-06` under `dh`.

Result 3 settles the design question the critique raised. **The error is not
beam-driven. It is dominated by the region `dh` already weights**, and starving
that region to feed the beam destroys accuracy globally. The frozen-`H` step is
exact for a constant diagonal, but the diagonal *changes* within a step, and the
resulting relative-phase error between levels is a real error -- it is not
removed by the exactness of `exp(-i D dt)`. `||dH/dt||` was a defensible monitor
and the objection to it was wrong.

Per-cell at `N_steps=4000`, showing both the point above and the cell that
blocks everything:

| scheme | `det=-1.0 p1` | `det=-1.0 p4` | `det=+0.0 p1` | `det=+1.5 p1` |
| --- | ---: | ---: | ---: | ---: |
| uniform | `6.525e-04` | `9.142e-03` | `2.567e-04` | `2.159e-05` |
| `dh` | `2.090e-03` | `2.529e-03` | `3.439e-04` | `1.881e-06` |
| `nac` | `1.491e-03` | `9.971e-03` | `1.653e-04` | `1.835e-05` |
| `mu` | `6.362e-03` | `1.510e-02` | `9.107e-03` | `9.862e-03` |

No monitor systematically fixes `det=-1.0 p4`. That cell is not a grid problem.

### How Well Truth Is Actually Known

The reference-free argument above compared each scheme against a finer run of
*itself*, and `G128`/`G256` share a monitor, so their `3.4e-05` agreement
measures that family's self-convergence rather than its distance from truth.
Independently constructed discretisations at high step count spread further:

| against `G256` | |
| --- | ---: |
| `G128` (same monitor) | `3.422e-05` |
| `U256` | `1.011e-04` |
| `mu`-graded at `128000` | `1.180e-04` |
| `nac`-graded at `128000` | `1.686e-04` |

Truth is therefore known to roughly `1-2e-04`, not `3e-05`, and **no error below
about `2e-04` in any table here is resolvable.** The conclusions above rest on
errors of `1e-03` and larger, which are safely above that, but a future run that
needs finer discrimination needs a better reference first.

### Implemented: `time_sampling`, Defaulting To Midpoint

Shipped 2026-08-28, and this one **changes results by default**. `run()` and
`run_microwave_scan()` take `time_sampling` (`"mid"`, the default, or `"left"`),
threaded to all four evolution loops and through the loky workers. Only the
argument to `H_slow_t(...)` / `H_mu_t(...)` moves; `dt` and everything
downstream are untouched. `_sample_offset` validates the value. The multitone
loop's beat phase `exp(-i * delta_omega * t)` is evaluated at the sampled time
too, since it is part of the Hamiltonian's time dependence, and
`gap_tracker.update` records energies against the sampled time rather than the
grid point. Results carry `time_sampling`.

Default flipped rather than left opt-in because midpoint is more accurate at
every step count measured and costs nothing per step. `"left"` remains available
and reproduces pre-option results exactly.

**Consequences to be aware of.**

- Every result changes. Measured `max|mid - left|` of `0.112` at `N_steps=300`
  on a 3-point scan; the difference shrinks with `N_steps` and is not
  measurable by `16000`, but it is not zero at production settings.
- **The saved analyses in `results/` were produced with left-endpoint sampling
  and no longer correspond to what the code now produces.** They have not been
  regenerated. Anything comparing new output against them must either pass
  `time_sampling="left"` or regenerate.
- The full suite passes unchanged (`43 passed, 1 skipped`), including
  `test_state_labelling.py`, whose tracked-index swap at `N_steps=160000`
  survives the change. The bitwise tests still hold because they compare two
  runs of the same settings.

### The Ceiling Is `1/f`, And Uniform Grids Get A Cancellation Bonus

Two results that together predict when grading is worth enabling, without
running anything.

**The ceiling is the reciprocal of the active fraction.** For a trajectory that
is active over a fraction `f` and static elsewhere, equidistribution can buy at
most `1/f`. Verified against the formula directly:

| active `f` | ceiling `p=1` | `1/f` |
| ---: | ---: | ---: |
| `1.00` | `1.00x` | `1.00` |
| `0.50` | `2.00x` | `2.00` |
| `0.10` | `10.02x` | `10.00` |
| `0.02` | `50.60x` | `50.00` |

Crucially this depends on how much of the run is **active**, not on how deep the
quiet is. SPA2's density spans `63774x`, which sounds decisive, but its effective
active fraction is `63%`, so the ceiling is `1.58x`. A deep but narrow quiet
region contributes almost nothing to the integral that sets the bound.

**Uniform grids cancel most of their own error.** Measured per-step against the
closed form on `spa_like` at production `delta*dt`:

| grid | summed local error | global error | cancellation |
| --- | ---: | ---: | ---: |
| uniform | `3.763e-03` | `2.129e-04` | **`17.7x`** |
| graded `p=1` | `2.166e-03` | `8.746e-04` | `2.5x` |
| graded `p=2` | `1.731e-03` | `3.816e-04` | `4.5x` |

Equidistribution does exactly what it claims -- it reduces the *summed* local
error by `1.7-2.2x`. It still loses, because uniform sampling cancels `94%` of
its local error and grading only `60-78%`.

**The mechanism is telescoping, not phase.** On a uniform grid the midpoint
rule's leading error sums as `Sum (dt^2/24) f''(t_i) dt -> (dt^2/24)[f'(T) -
f'(0)]`, a **boundary term**: interior contributions cancel pairwise because
every step has the same `dt`. Vary `dt` and the sum becomes a weighted integral
of `f''` instead, and nothing telescopes.

Demonstrated independently on the commuting model, by moving a bump so the
boundary derivative stops vanishing: interior bump gives uniform `3.31e-13`,
a bump at `t/T = 0.06` (where `f'(0) = 3.6e+03`) gives `9.77e-06`. Seven orders,
from the boundary term alone.

An earlier explanation in terms of a geometric series in the fast phase is
**wrong**: the cancellation is already `17.6x` at `delta*dt = 1e-02`, where
phases barely rotate. Phase rotation only widens uniform's advantage from `3.5x`
to `7x` between `delta*dt = 1e-02` and `8e+04`; it does not create it.

**Practical rule.** Grading is worth enabling when `1/f` comfortably exceeds the
cancellation ratio being given up -- on the order of `14x`, not `1x`. Compute
`1/f` from the field profile before running anything:

```python
d = field_variation_density(H_slow, muw_hams)(ts)
ceiling = T * np.trapezoid(d, ts) / np.trapezoid(np.sqrt(d), ts) ** 2
```

SPA2 gives `1.58x` and grading loses. A run that is `90%` static gives `10x` and
should be re-measured rather than assumed either way.

**Evidence class**: exact solution, plus arithmetic on measured profiles.

### Graded Grids Against Exact Solutions

**Measured 2026-08-28, after two invalid attempts.** `step_density` shipped on
accuracy figures scored against `G256`, which is *itself a graded run* -- the
same self-reference bias this section identifies for `U128`, mirrored with the
sign flipped in grading's favour. Every `10-100x`, the `120x` at `N=500` and the
`5.6x` effective-shift figure inherit it and **do not survive**. The honest
figure is single digits.

**Verdict: grading helps, by `1.1x` to `7.7x`, on the structure the real problem
has.** Non-commuting `H`, density profile taken directly from the measured SPA2
`||dH/dt||` (`41278x` dynamic range against the real `63774x`, `61%` of the
flight quiet), enveloped microwave coupling in `muw_hams`, density built by the
shipped `field_variation_density`, grids by the shipped `build_time_grid`, truth
from a fine product with self-convergence at `1.4e-07`:

| `N_steps` | median gain | min gain over 3 seeds |
| ---: | ---: | ---: |
| `500` | `2.52x` | `2.37x` |
| `1000` | `2.47x` | `1.83x` |
| `2000` | `3.46x` | `2.73x` |
| `4000` | `1.18x` | `1.14x` |

Grading wins in **all 12** rows. No single `order` dominates: `order=1` and
`order=2` take five rows each, `order=4` two, and at `N=4000` the aggressive
grids can lose to uniform while `order=4` still wins. Gains above the `1.58x`
equidistribution ceiling are real but come from uniform being *erratic* at large
`delta*dt`, not from grading being more efficient than the bound allows.

### Two Invalid Tests, Recorded So They Are Not Repeated

Both produced confident wrong answers before the setup was checked.

**A flat density.** The first model used `H_slow = R(Gt) H0 R(Gt)^H`, a *uniform*
rotation, whose `||dH/dt||` is constant. Measured dynamic range: **`1.01x`**,
against the real trajectory's `63774x`. Testing a step-*placement* optimisation
on a problem with uniform step requirements measures nothing, and it produced
the conclusion "grading never wins" plus a recommendation to delete the feature.
**Always report the density dynamic range alongside any grading result**; below
about `10x` the comparison is meaningless.

**A commuting `H`.** For commuting `H(t)` the propagation collapses to a scalar
quadrature of `F = integral of f`, and composite midpoint on a smooth interior
bump enjoys Euler-Maclaurin cancellation -- the leading error term is
`(dt^2/24)[f'(T) - f'(0)]`, which vanishes when the bump is interior. Uniform
then reaches `1e-12` while any graded grid, which destroys the cancellation,
sits near `1e-02`. This is a genuine property, not an artefact, but it is
specific to commuting `H` and says nothing about the real problem. Moving the
bump does not rescue it: at `11 sigma` from the boundary `f'` still vanishes
there. **The `scalar` model cannot evaluate grading at all.**

The second point has a cost: `scalar` is the only model here with a closed form
at arbitrary spectral scale, so it is the only one that could reach production
`delta*dt`. Since it cannot evaluate grading, **the grading result above is
limited to `delta*dt` between `5` and `40`**, and a brute-force oracle cannot go
further -- resolving `delta*dt ~ 7.65e+04` needs the same `~7.6e+08` steps that
make the regime unreachable in the first place. Extrapolating the `1.1-7.7x` to
production settings is an assumption, not a measurement.

**Evidence class**: exact solution (fine-product oracle, self-convergence
verified) at `delta*dt <= 40`. Supersedes every `G256`-referenced grading claim.

### Magnus At Production `delta*dt`, Against Closed Forms

**Measured 2026-08-28. This closes the Magnus question.** `rotating` and
`scalar` have closed forms at *any* spectral scale, so production `delta*dt` is
reachable with exact truth and no oracle to validate. `split_model` puts the
coupling in `muw_hams` while keeping the sum equal to the model, so the closed
form survives. `n=64`, batch `25`, `delta*dt = 7.65e+04`, `||H_mu||/spread =
5.0e-07` and `||A|| = 0.038 rad`, both matched to SPA2.

| model | `N_steps` | `left` | `mid` | `magnus` | magnus time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rotating` | `1000` | `2.212e-02` | `2.205e-02` | `2.205e-02` | `3.59 s` vs `12.47 s` |
| `rotating` | `4000` | `1.555e-02` | `1.556e-02` | `1.556e-02` | `14.35 s` vs `49.85 s` |
| `scalar` | `1000` | `3.869e-01` | `1.397e-01` | `1.397e-01` | `3.02 s` vs `11.89 s` |
| `scalar` | `4000` | `4.050e-01` | `9.033e-03` | `9.033e-03` | `12.08 s` vs `47.88 s` |

**Magnus matches `mid` to four significant figures against exact truth**, at
every step count on both models, while running `3.5-4x` faster.

Speed, measured separately with warmup and three interleaved repeats,
`n=64`, `N=2000`:

| batch | shipped | magnus | speedup |
| ---: | ---: | ---: | ---: |
| `1` | `4.34 +- 0.02 s` | `3.53 +- 0.01 s` | `1.23x` |
| `5` | `8.02 +- 0.01 s` | `4.25 +- 0.02 s` | `1.89x` |
| `25` | `26.03 +- 0.01 s` | `8.05 +- 0.01 s` | `3.23x` |

The scaling with batch is the mechanism: Magnus removes the *per-scan-point*
`O(n^3)` eigensolve and leaves the shared slow one, so the gain grows with batch
and is `1.23x` at batch `1`. Consistent with the independent `2.49x` measured on
the real SPA2 workload in `NATIVE_PORT_INVESTIGATION.md`.

**A self-inflicted regression, caught late.** The scaling-and-squaring guard
added to `apply_expm_taylor` used `np.linalg.norm(A, 2)` -- a spectral norm, so
an SVD, `182 us` at `n=64` against the `420 us` eigensolve the propagator exists
to avoid. It ate `43%` of the benefit and sat through several measurements
unnoticed. Now `np.linalg.norm(A, "fro")` at `3.1 us`: Frobenius bounds the
spectral norm from above, which is all scaling and squaring needs.

**Two side results.** On the *commuting* model `mid` holds order `1.96`/`2.00`
even at `delta*dt = 7.65e+04` -- with no time-ordering error there is nothing for
large `delta*dt` to spoil, which is the mechanism stated exactly. And `left` is
saturated there at `4e-01` with no convergence at all, making midpoint `45x`
better and independently justifying the default flip.

**Evidence class**: exact solution (closed form) at production `delta*dt`, plus
direct interleaved timing. No reference requiring validation.

### Strang And Lie, Measured Rather Than Bounded

**Measured 2026-08-28.** Both were rejected on a propagator-norm bound --
accumulated error `1.39e+01` against a trivial bound of `2` -- with no observable
ever run. Since this document's recurring lesson is to conclude from a
measurement, they are now implemented (`make_splitting_loop` in
`benchmarks/bench_analytic.py`) and scored against exact solutions.

**The implementations are correct.** In the clean regime (`delta*dt = 1e-03`)
Strang measures order `2.00` and Lie order `1.00`, their textbook values, with
Strang matching midpoint and Magnus:

| variant | `N=1000` | `N=2000` | order |
| --- | ---: | ---: | ---: |
| `mid` | `1.222e-08` | `3.051e-09` | `2.00` |
| `magnus` | `3.665e-08` | `9.152e-09` | `2.00` |
| `strang` | `3.644e-08` | `9.101e-09` | `2.00` |
| `lie` | `4.502e-05` | `2.250e-05` | `1.00` |

**At production `delta*dt` they lose, and the rejection stands.** On `rotating`
at `delta*dt = 7.65e+04`, `N=4000`:

| variant | error | vs midpoint |
| --- | ---: | ---: |
| `mid` / `magnus` | `2.280e-03` | -- |
| `strang` | `3.675e-02` | `16x` worse |
| `lie` | `3.560e-02` | `16x` worse |

**But the bound overstated the case.** It concluded the propagator is
"completely wrong", the accumulated error exceeding the trivial bound of `2`.
Measured on an observable they are `4-16x` worse than midpoint -- bad, and
disqualifying given Magnus is simultaneously more accurate *and* `3.2x` faster,
but not meaningless. A bound that says "worse than useless" where the
measurement says "16x worse" is the failure mode this document keeps recording.

**A trap in the `engineered` model, found here.** That model is constructed as
`U = V(t) exp(-i Phi(t))` -- a product of exactly two exponentials, which *is*
the Lie-split form -- and its natural split (`H_slow = V diag(w) V^H`,
`H_mu = theta' G`) is aligned with that factorisation. Lie therefore reproduces
it almost exactly and appears `73000x` better than midpoint. Splitting the same
`H(t)` arbitrarily instead:

| split | `mid` | `strang` | `lie` |
| --- | ---: | ---: | ---: |
| natural, `V`/`Phi` aligned | `2.691e-02` | `4.323e-02` | `3.677e-07` |
| arbitrary, misaligned | `2.691e-02` | `9.800e-02` | `1.020e-01` |

**Do not evaluate splitting methods on `engineered` with its natural split.**
The same caution applies to any model built by choosing `U` first: the
factorisation used to construct it will flatter whichever integrator shares that
structure.

**Evidence class**: exact solution, both regimes.

### Analytic Ground Truth, And Why No Order Was Ever Measurable

**Added 2026-08-28: `benchmarks/analytic_models.py`, `tests/test_analytic.py`.**
Six time-dependent Hamiltonians with known exact solutions. Nothing in them
imports `centrex_tlf`, so these are the first checks in this repository against
an answer that is true independently of the code -- closing the gap `AGENTS.md`
names, that the suite "checks internal consistency, so a change to the
underlying `centrex_tlf` Hamiltonian would move every result together and leave
all tests passing".

Every reference used earlier in this investigation was a finer run of some
scheme, and each time the reference's own error turned out to be comparable to
what was being measured. That is now unnecessary.

| model | form | exact answer | verified to |
| --- | --- | --- | ---: |
| `rotating` | `H = e^{-iGt} H0 e^{iGt}` | `U = e^{-iGt} e^{-i(H0-G)t}` | `2.2e-11` |
| `scalar` | `H = f(t) H0` | `U = exp(-i F(t) H0)` | `1.6e-11` |
| `rosen_zener` | sech pulse, fixed detuning | `sin^2(pi Om0 tau/2) sech^2(pi D tau/2)` | `2.3e-09` |
| `allen_eberly` | sech pulse, tanh chirp | `1 - cos^2(pi/2 sqrt((Om0 tau)^2-(D0 tau)^2))/cosh^2(pi D0 tau/2)` | `6.8e-13` |
| `landau_zener` | linear sweep, fixed coupling | `exp(-pi Om^2/2a)`, **asymptotic only** | `2.0e-03` |
| `spa2_replica` | Gaussian envelope, U-shaped sweep | none; fine `n=2` product | `~1e-11` |

`allen_eberly` is the microwave-transfer analogue: envelope *and* sweep, which is
SPA2's structure. `spa2_replica` reproduces the geometry that makes `det=-1.0`
hard -- two crossings, the near one `2.5 sigma` out in the Gaussian flank, giving
partial transfer of `0.68`-`0.82` (the real cell sits at `0.69`).

**Landau-Zener's closed form is not an oracle.** It is a `t -> +/-inf` asymptote
and the sweep never switches off, so a finite window leaves the population still
oscillating toward the limit: `2.0e-03` off at the default window and `8.6e-02`
if the window is halved. Use it as a physics check at the `1e-2` level, never for
accuracy work. A two-level product propagator reaches `~1e-11` and is the oracle
instead.

### The Mechanism: Order Collapse Is A Time-Ordering Effect

Driving the shipped batched loop on these models, with `delta*dt` dialled by the
spectral scale:

| model | `H` commutes with itself? | `delta*dt` at `N=1000` | observed order |
| --- | --- | ---: | ---: |
| `scalar` | yes | `1.0e-03` | `2.00` |
| `scalar` | yes | `2.0e+01` | `2.00` |
| `scalar` | yes | `2.0e+02` | `2.00` |
| `rotating` | no | `1.0e-03` | `2.00` |
| `rotating` | no | `2.0e+01` | `-0.45` |
| `rotating` | no | `2.0e+02` | `0.18` |

**The midpoint scheme is second order.** It shows exactly `2.00` whenever the
problem is resolved, and it keeps `2.00` at any scale when `H(t)` commutes with
itself at different times, because then the only error is quadrature of the
scalar factor. The collapse happens only in the non-commuting case, once
`delta*dt` exceeds about `1`: the neglected terms are time-ordering commutators
carrying powers of `delta*dt`, and beyond that point no asymptotic power law is
available to measure.

SPA2 runs at `delta*dt ~ 7.65e+04` with a non-commuting `H`. Reaching
`delta*dt ~ 1` would need `N ~ 7.6e+08` steps.

**This single mechanism explains the whole convergence story in this document:**

- why midpoint never showed its formal second order on the real problem;
- why measured orders came out erratic and sometimes negative (`-1.09` for `mid`
  at `4000/8000/16000`);
- why the error oscillates in `N_steps` rather than falling monotonically;
- why every reference was so hard to establish, and why references built from a
  finer run of the same scheme were systematically misleading;
- why **Richardson extrapolation is unavailable** -- it needs a clean order to
  build its coefficients from, and there is none;
- why **CF4, Filon quadrature and modified Magnus cannot be expected to deliver
  their formal orders either**. They are derived from the same expansion that
  fails to converge here. Any of them must be judged by measured error at
  matched cost, never by order.

The practical consequence for anyone extending this work: **do not reason about
convergence order at production settings.** Rank propagators by measured error
per unit wall clock, and use the clean regime only to confirm an implementation
is correct.

### Propagators Ranked Against Exact Solutions

`benchmarks/bench_analytic.py` drives the shipped loops on the models above. The
model is split `H_slow(t) = H_model(t) - H_mu(t)` with `H_mu` a small enveloped
coupling, so the sum is exactly the model and the closed form still applies,
while the interaction-picture Magnus propagator gets the large-diagonal plus
small-coupling structure it exists to exploit. Batch entries are identical, so
all of them share one known answer while the per-scan-point cost still scales.

**Clean regime, the correctness gate.** `n=5`, `delta*dt = 1e-03`:

| variant | order per doubling | error at `N=2000` |
| --- | ---: | ---: |
| `left` | `1.00` | `2.202e-05` |
| `mid` | `2.00` | `1.208e-09` |
| `magnus` | `2.00` | `1.068e-09` |

All three loops attain their formal order to two decimals on both `rotating` and
`scalar`. This is the first time any of that has been verified against an answer
rather than against another run of the same code.

**Realistic regime, `delta*dt = 7.65e+04`, `n=64`, batch `25`.** Two results:

| model | `N_steps` | `left` | `mid` | `magnus` |
| --- | ---: | ---: | ---: | ---: |
| `scalar` (commuting) | `4000` | `4.050e-01` | `9.033e-03` | `9.033e-03` |
| `rotating` (non-commuting) | `4000` | `4.244e-03` | `4.216e-03` | `4.217e-03` |

- **Midpoint is vindicated where the error is phase-dominated.** On the commuting
  model `left` saturates near `0.4` with no convergence at all, while `mid` holds
  order `1.96`-`2.00` and reaches `9.0e-03` -- a `45x` gap. When the error comes
  from quadrature of a large diagonal phase, which is what the Stark-shifted
  spectrum produces, midpoint is the difference between converging and not.
- **On the non-commuting model nothing helps**: all three sit near `4e-03` with
  the same order. If the real problem's error is time-ordering dominated there
  may be a floor no propagator in this family removes.

**Magnus wall clock at production shape**: `13.12 s` against `49.46 s`, i.e.
**`3.8x` faster**, better than the `2.49x` on record, with accuracy identical to
`mid` to four significant figures.

### Where Magnus Breaks, Measured

Sweeping the coupling as a fraction of the spectral spread, `rotating`, `n=64`,
batch `25`, `N_steps=4000`:

| `||H_mu||/spread` | `mid` | `magnus` | verdict |
| ---: | ---: | ---: | --- |
| `5e-07` (**the real problem**) | `4.216e-03` | `4.217e-03` | identical |
| `5e-06` | `4.216e-03` | `4.314e-03` | `2%` worse |
| `5e-05` | `4.216e-03` | `4.181e-02` | `10x` worse, **silently** |
| `5e-04` | `4.216e-03` | overflow to NaN | complete breakdown |

The cause was `TAYLOR_TERMS = 8` in `apply_expm_taylor`: `||A|| ~ ||H_mu||*dt` is
about `4.3e-02` rad at the operating point, where eight terms are ample, but
`~1.9` at `5e-05` and `~19` at `5e-04`, where the series diverges. **Fixed** by
scaling and squaring on the vectors, chosen from the measured `||A||`, which
costs nothing at `k=0` and keeps the `n^2*S` scaling. After the fix `5e-04`
degrades gracefully to `5.6e-01` -- a genuine first-order-Magnus truncation
error, since the neglected second term grows with the coupling -- rather than
producing NaN.

Worth stating plainly: the failure at `5e-05` produced no warning of any kind.
The operating point is two orders of magnitude away from it, so this was latent
rather than active, but a fixed term count is the wrong shape for a guard.

### The Four SPA2 Regressions Remain Unexplained

> **Superseded.** They are explained, and they are not a Magnus defect: see
> the C1 rows in Verdicts and *Phase C, closed*. The comparison this
> subsection rests on scored `N=4000` against `magnus@128000`, a cross-`N`
> comparison of tracked-label columns, which is invalid here. At matched
> `N_steps` the two schemes converge together in every cell.

Two mechanisms were proposed and both are ruled out at the real operating point:

- **Cancellation in `magnus_integral`.** Its outer-product form loses the
  `expm1` cancellation benefit, and its small-argument guard is a hard cutoff at
  `|x*dt| < 1e-8`. Measured relative error against the `expm1` form peaks at
  `5e-09` right at the cutoff and is `<= 4.5e-11` elsewhere. Too small by orders
  of magnitude.
- **Near-degenerate spectra.** Real TlF has near-degenerate hyperfine levels;
  random Hermitian matrices do not, so the degeneracy branch was never exercised.
  Rebuilding the model with eigenvalues in near-degenerate pairs and sweeping the
  gap from `1e-01` to `1e-09` of the spread gives `magnus/mid` error ratios of
  `1.00, 0.99, 1.15, 1.00, 1.00`. No degradation.

At the real coupling ratio, with near-degeneracies present, Magnus matches `mid`.
The regressions are not reproduced by anything constructible here.

**The one structural element still unmodelled is `D_mu`**, the per-scan-point
detuning diagonal, which was set to zero throughout because a nonzero `D_mu` is
diagonal in the *slow eigenbasis* and therefore time-dependent through `V`,
which breaks the closed form. Magnus computes its oscillatory integral from
`delta = D + D_mu`, so a large `D_mu` changes the integrand's structure in a way
no model here reaches. That, or the regressions are an artefact of scoring
against `magnus@128000` -- Magnus's own fine run. Both remain open.

### Reopened And Overturned: The Magnus Propagator Is Faster At Matched Accuracy

**Measured 2026-08-28. `NATIVE_PORT_INVESTIGATION.md:85-173` rejects this
propagator; that rejection does not survive a conditioning-robust metric.**

Two independent defects in the original verdict:

1. It scored `max|delta probabilities_final|` over `np.linspace(-2e6, 1e6, 5)`.
   Measured local slopes at those five detunings are `9.6e-05`, `2.9e-04`,
   `2.5e-05`, **`1.0e-01`** and `6.8e-06` per kHz. The `+0.25 MHz` point turns a
   `0.2 kHz` effective error into `2e-02` of population on its own.
2. It applied a `1e-6` acceptance target to `|magnus - shipped|` at matched
   `N_steps`, treating the shipped scheme as exact. The shipped scheme's own
   error at those step counts is about `1e-4`, so the target asked Magnus to
   reproduce the shipped scheme's errors. The "`N_steps ~ 5e5`, `26x` slower"
   economics is derived entirely from that number.

Re-measured at **matched wall clock** -- the right question, since Magnus is
about `2.5x` cheaper per step -- on a lineshape window, batch `25`, with the
reference built from **two structurally different propagators** at `N=64000`
(exact eigensolve and Magnus) agreeing to `4.234e-05`:

| budget | shipped | residual | magnus | residual | verdict |
| ---: | --- | ---: | --- | ---: | --- |
| `14.9 s` | `N=2000` | `2.947e-03` | `N=5000` | `3.062e-03` | tie |
| `29.8 s` | `N=4000` | `2.554e-03` | `N=10000` | `8.138e-04` | **`3.14x` better** |
| `59.5 s` | `N=8000` | `1.463e-03` | `N=20000` | `6.542e-04` | **`2.24x` better** |

Wall clock matched to within `1%` in every pair. Read the other way,
`magnus@10000` beats `shipped@8000` on accuracy in half the time, so the speedup
at matched accuracy is at least `2x` in the production range.

### Where Magnus Is Worse, And Whether It Matters

> **Superseded.** scored against `magnus@128000`, i.e. Magnus's own fine run. Superseded by *Magnus At Production `delta*dt`, Against Closed Forms*, where Magnus matches midpoint to four significant figures against exact truth.

Full convergence grid, `5` detunings x `3` prefactors, at matched wall clock
(`shipped@8000` against `magnus@20000`), against the same reference:

| cell | shipped | magnus | ratio |
| --- | ---: | ---: | ---: |
| `det=-2.0 pref=1` | `6.684e-05` | `8.735e-05` | `0.77` |
| `det=-2.0 pref=4` | `1.131e-04` | `2.133e-04` | `0.53` |
| `det=-1.0 pref=1` | `1.088e-03` | `7.407e-04` | `1.47` |
| `det=-1.0 pref=16` | `4.231e-03` | `2.814e-03` | `1.50` |
| `det=+0.0 pref=1` | `5.954e-05` | `1.371e-05` | `4.34` |
| `det=+0.0 pref=16` | `2.641e-05` | `2.153e-04` | `0.12` |
| `det=+0.5 pref=4` | `3.178e-03` | `9.620e-04` | `3.30` |
| `det=+0.5 pref=16` | `4.865e-04` | `9.592e-05` | `5.07` |
| `det=+1.5 pref=16` | `1.951e-05` | `5.241e-05` | `0.37` |

Magnus wins `11` of `15`. **Worst cell, which is what certifies a scan:
`4.231e-03` against `2.814e-03`, a `1.50x` improvement.** Three of the four
regressions are at prefactor `4` or `16`, which is the expected direction --
`||A||` grows with the drive and so does the neglected second Magnus term.

Re-scored 2026-08-28 against `magnus@128000` with a per-cell floor from
cross-method disagreement and Magnus self-convergence, the four losses **survive**
-- floors are `10-50x` below them, so they are resolvable rather than noise.
Magnus still wins `11/15` and improves the binding cell from `4.507e-03` to
`2.538e-03` (`1.78x`). Note that reference is Magnus's own fine run and so
flatters Magnus; its wins may be overstated and its losses understated. Every
regressing cell has absolute error `<= 2.2e-04`, against `2.8e-03` in the
binding cell. A strict "no cell may regress" gate rejects this; a gate on
worst-case accuracy accepts it comfortably. Recording both readings rather than
silently picking one: the strict gate was written before the error distribution
across the grid was known, and rejecting a method for degrading cells already
`20x` better than the binding constraint repeats the error of scoring on a
quantity that does not bound the result.

### Not Yet Established

> **Partly superseded.** The first bullet is done: the multitone path is
> implemented, integrates the beat with per-component shifted kernels, and is
> tested against a closed form (`rotating_coupling`). It is now the *more*
> accurate multitone path, not a hazard to fence off. Read the remaining
> bullets with that in mind.

- **The multitone path is untested.** `make_magnus_loop` replaces only
  `_time_evolve_mu_batched_shared_slow`. The multitone loop carries a beat phase
  `exp(-i * delta_omega * t_sample)` inside `H_rot`; a Magnus step that freezes
  it rather than integrating it across the step would be wrong in a way no
  single-tone test can reveal.
- **The `n^3` second-Magnus-term objection stands** and is metric-independent.
  It bounds any attempt to push this to higher order, though it does not affect
  the first-order-Magnus result measured here.
- Only one lineshape window was used for the wall-clock table; the grid table
  above is single-shot per cell, with no repeats.

### Corrected: Midpoint Is The Stronger Lever, Measured On Lineshapes

> **Superseded.** the direction is right but the reference is a two-discretisation average with `3.991e-04` of its own uncertainty. Superseded by the closed-form measurement in *Magnus At Production `delta*dt`*, which gives `45x` for midpoint over left-endpoint on a commuting model.

The rejection of midpoint sampling recorded below used `max|dP|` at fixed
detunings, which the conditioning section above shows cannot measure convergence
here. Re-tested on the lineshape residual, over `det` from `-1.10` to
`-0.90 MHz`, against a reference built from **two different discretisations**
(`left-uniform` and `mid-graded`, both at `N=64000`) whose mutual disagreement is
`3.991e-04`:

| `N_steps` | `left-uniform` | `mid-uniform` | `left-graded` | `mid-graded` |
| ---: | ---: | ---: | ---: | ---: |
| `1000` | `9.165e-03` | `2.937e-03` | `3.380e-03` | `3.248e-03` |
| `2000` | `4.687e-03` | `1.830e-03` | `4.817e-03` | `1.175e-03` |
| `4000` | `2.942e-03` | `2.416e-03` | `2.942e-03` | `2.214e-03` |
| `8000` | `1.526e-03` | `1.444e-03` | `1.544e-03` | `1.106e-03` |
| `16000` | `7.144e-04` | `6.748e-04` | `5.984e-04` | `5.157e-04` |

Step-count saving at matched residual, against `left-uniform`:

| variant | `1000` | `2000` | `4000` | `8000` | `16000` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mid-uniform` | `4.01x` | `3.30x` | `1.23x` | `1.05x` | `1.00x` |
| `left-graded` | `3.25x` | `0.97x` | `1.00x` | `0.99x` | `1.00x` |
| `mid-graded` | `3.45x` | `5.08x` | `1.35x` | `1.34x` | `1.00x` |

- **Midpoint alone beats grading alone** (`4.01x` against `3.25x` at `1000`,
  `3.30x` against `0.97x` at `2000`), costs nothing per step, and needs no
  density pre-pass.
- **They compose sub-multiplicatively.** At `2000`, midpoint gives `3.30x`,
  grading alone `0.97x`, together `5.08x`.
- **The gain is concentrated at coarse step counts**: `3.4-5x` below `2000`,
  about `1.35x` at `8000`, unmeasurable at `16000`.
- Midpoint is still **not second order** -- residual ratios per doubling are
  `1.5-1.8`, not `4`. It lowers the error constant by about `3x` and leaves the
  order at `1`. Why the scheme stays first order is unresolved; the obvious
  suspect, a low-order field interpolant, is ruled out because `Ez_from_csv`
  uses cubic interpolation, so `H` is C2 in time.

The `16000` row is at the noise floor: residuals of `5-7e-04` against a
reference good to `3.991e-04`. Read it as unmeasurable rather than as no gain.

**Implication for the `1.58x` ceiling.** That ceiling bounds step *placement*
only. Midpoint moves the error constant, a different axis, which is why the
combination reaches `5.08x` at `N=2000`. Neither changes the convergence order,
so both gains shrink as the accuracy target tightens.

### Superseded: Midpoint Sampling Measured With The Broken Metric

> **Superseded.** This subsection calls midpoint "rejected". It is the shipped
> default, is second order against exact solutions where left-endpoint is first,
> and is `45x` more accurate at production `delta*dt` on a commuting model. The
> text is kept for the reasoning that led to the wrong call.

All four loops evaluate `H` at the left endpoint (`simulator.py:1505`, `:1641`,
`:1795`, `:1939`). The step is exact for a frozen `H`, so the only time-sampling
error is the time-ordering term: `O(dt^2)` per step, `O(dt)` globally, and
`H(t + dt/2)` costs the same. Measured on the `det=-1.0` cells, midpoint is
better at some step counts and worse at others with no systematic advantage;
both reach about `1e-03` by `4000`. Selected indices were `(14, 35)` throughout.

The prediction was that the left-endpoint diagonal phase error `delta'*dt^2/2`
dominates and midpoint cancels it exactly. It does cancel it -- but that term is
a per-eigenstate *phase*, and the observable is populations in that same
instantaneous eigenbasis, so cancelling it does not move the answer. Fourth
propagator change rejected here after Strang, Lie and Magnus, and for the same
reason: an error-term argument accepted before it was measured.

Caveat: the midpoint comparison was scored against a uniform reference at
`64000`, before the reference problem above was understood. Both schemes were
scored symmetrically against it, so the conclusion is unlikely to invert, but it
has **not** been re-scored against `G256`.

### Also A No-Op: An Endpoint Readout Basis

After the loop `psis` is at `T` but `last_evecs`/`V_fin` is the basis at the last
*sampled* time -- `T-dt` today. Re-diagonalising at exactly `t=T` and reordering
onto the tracking chain (with `V_fin` repointed at it, since `population()` reads
labels from `V_fin` and the probabilities must share that basis) moves `V_fin` by
`2.126e-04` and `probabilities_final` by `2.119e-08`. Quadratic, because the
readout state is already essentially an eigenstate, so the first-order term
vanishes by orthogonality.

### What This Actually Found

Not a speedup. Three things worth more than one:

1. **`N_steps=128000` uniform is not converged**, and any figure referenced to a
   finer run of the *same* scheme carries a bias that grows as the comparison
   tightens. Referenced runs in this repository should be re-read with that in
   mind.
2. **Worst-case convergence on this grid is set by one cell**, `det=-1.0`, which
   does not converge reliably under either grid anywhere from `500` to `24000`
   steps, while every other cell reaches `1e-05` or better. Production runs at
   `10000-20000` therefore carry roughly `1e-03` in that cell.
3. **The graded grid is the more accurate discretisation**, by the reference-free
   test and by `10-100x` in error on the cells that behave. It is available if
   the certifying cell is ever brought under control.

The time grid is bounded and now implemented. Its ceiling is `1.58x`, grading
reaches for it correctly once the step cap is removed, and `||dH/dt||` is the
right monitor of the four tried. Grading is worth `10-100x` in error on
well-conditioned cells and `5.6x` in effective detuning at `N_steps=1000`; it is
a wash in the steep partial-transfer regime. Whether to cut `N_steps` is a
per-observable judgement, and `benchmarks/bench_step_grid.py` is the tool for it.

What is left open is not numerics:

**Is the `det=-1.0` regime a useful place to take data?** The observable there
swings `0.177` across `100 kHz` and is exponentially sensitive to the Rabi rate
at a crossing `2.75 sigma` out in the beam wing. That is a statement about the
experiment, not the simulation, and it bounds how precisely any prediction in
that regime can be compared to a measurement.

Raw values: `scratch_sweep.json`, `scratch_refs.json` in the repository root,
written by ad-hoc scripts during the session and untracked; delete them freely.
`benchmarks/bench_step_grid.py` reproduces the measurements but does not write
those two files.

## Re-Prioritization Of The Existing Backlog

| Existing item | Status after this review |
| --- | --- |
| Priority 1: standardize shared-slow scans | Keep. Still the correct default route. |
| Priority 2: final-only / monitor-only storage | Keep. Large memory win, modest runtime win. |
| Priority 3: parameterized benchmark scripts | Done. `benchmarks/` now exists. |
| Priority 4: real-workload GPU bottleneck | Superseded by Priority A, which proposes a cause. |
| Priority 5: Windows CuPy DLL setup | Keep, and de-duplicate. See Ease Of Use C. |
| Priority 6: vectorize CPU inner loop | Demote. Measured `2-5%`. |
| Priority 7: monitor-only GPU probabilities | Keep as a memory cleanup, not a runtime item. |
| Priority 8: precompute field time series | Demote. Lives inside the `22%`. |
| Priority 9: cache coupling matrices | Fold into Priority D. |
| Priority 10: GPU buffer reuse | Demote. Never reproducibly measured. |

## Ease Of Use A: Empty Package `__init__.py` — DONE

`src/state_prep/__init__.py` is `0` bytes. Every notebook and example therefore
opens with eight separate deep imports:

```python
from state_prep.simulator import Simulator
from state_prep.trajectory import Trajectory
from state_prep.hamiltonians import SlowHamiltonian
from state_prep.electric_fields import ElectricField
from state_prep.magnetic_fields import MagneticField
from state_prep.microwaves import MicrowaveField, Polarization
from state_prep.intensity_profiles import GaussianBeam, BackgroundField
from state_prep.approximate_states import ...
```

Recommendation:

- Re-export the public API from `state_prep/__init__.py`, with `__all__`.
- Add `__version__`, sourced from the installed distribution metadata so it does
  not drift from `pyproject.toml`.

This is the largest reduction in friction per line changed available in the
repository.

## Ease Of Use B: No Tests — DONE

Implemented 2026-08-15: `tests/`, 20 passing plus 1 skipped (GPU, needs CuPy),
runs in about 11 seconds. `pytest` is a dev dependency and `pyproject.toml`
carries the config.

- `test_scan_agreement.py`: shared-slow versus repeated `run()`, probability
  normalisation, BLAS pinning being bit-for-bit inert, loky workers matching
  serial exactly, and the two eigensolver backends agreeing within the
  cross-backend tolerance.
- `test_api.py`: every name in `__all__` importable, `scan_grid` semantics
  including the fixed-source case, storage defaults, result accessors, pickle
  round trip, units and shapes.
- `test_rotating_frame.py`: the shift is diagonal and touches only the addressed
  manifold, detuning enters linearly, same-carrier fields must share a detuning,
  all three call sites produce identical arrays, and the multitone path is
  rejected unless explicitly enabled.
- `test_gpu.py`: skipped without CuPy.

Tolerances are named in `tests/conftest.py` and set from measurements rather
than guesses: exact for reordered arithmetic, tight for same-backend paths, and
about `1e-5` when the eigensolver, device, or ordering differs.

### Original Note

There is no test suite and no linter configuration. This matters more than usual
right now, because the active work rewrites propagation internals.

The existing backlog already asks for "a regression test comparing a small
shared-slow scan against repeated single runs". That test is also the oracle
required to validate Priority A and Priority B above.

Recommendation:

- Add a small regression test comparing `run_microwave_scan(...)` against
  repeated `run(...)` at a tolerance near the measured `7.314e-10`.
- Add a shape and units test for `SimulationResult` and `MicrowaveScanResult`.
- Add a CPU/GPU agreement test, skipped when CuPy is unavailable.

This should land before any propagation change, not after.

## Ease Of Use C: Duplicated Setup Logic — DONE

Implemented 2026-08-15. `build_rotating_frame_shift` now lives in
`state_prep/microwaves.py` and is the single implementation; `simulator.py`,
`benchmarks/common.py` and `state_prep_gpu/benchmark_spa2_real.py` all delegate
to it. CUDA DLL discovery is consolidated into
`state_prep_gpu._cupy.add_cuda_dll_directories`, which now also covers the glob
based discovery the benchmarks needed, with the `PATH` prepend behind a flag so
importing CuPy does not mutate the environment.

Verified: simulator output bitwise unchanged (`0.000e+00` across both batched
paths and all outputs), and all three call sites produce identical arrays. A
test pins that last property so the copies cannot silently diverge again.

### Original Note

The rotating-frame `D_mu` construction exists in three independent copies:

- `src/state_prep/simulator.py:701`
- `src/state_prep_gpu/benchmark_spa2_real.py:113`
- `benchmarks/common.py:440`

The CUDA DLL path setup exists in two:

- `src/state_prep_gpu/_cupy.py:57`
- `benchmarks/common.py:27`

Root cause:

`state_prep_gpu` accepts only precomputed arrays and provides no bridge from
`state_prep` objects, so every caller rebuilds the scan setup by hand.

Recommendation:

- Add a shared setup helper that converts a `Simulator` plus scan parameters
  into the array inputs the GPU path expects, and use it from all three sites.
- Have `benchmarks/common.py` import the DLL setup from `state_prep_gpu._cupy`
  rather than reimplementing it.

Beyond the maintenance cost, three copies of the rotating-frame convention is a
correctness risk: they can drift silently, and only the CPU copy is exercised by
the notebooks.

## Ease Of Use D: `MicrowaveScanResult` Is A Bare Container — DONE

Implemented 2026-08-15: `batch_size`, `get_state_probability`,
`get_monitor_probability`, `plot_state_probability`, `to_polars` and
`save_to_pickle`, with initial-state resolution matching `SimulationResult`
(accepting either the approximate state or the tracked eigenstate).

### Original Note

`SimulationResult` provides about ten accessor and plotting methods
(`src/state_prep/simulator.py:57` onward). `MicrowaveScanResult`
(`src/state_prep/simulator.py:325`) provides none: no
`get_monitor_probability(state)`, no plotting, no `save_to_pickle`.

This is visible in the repository as `29` hand-rolled CSV files under
`examples/SPA/Experimental verification/results`.

Recommendation:

- Add state-resolved accessors mirroring `SimulationResult`.
- Add `save_to_pickle`, consistent with the existing `dill` usage.
- Add a `to_polars()` export, since `polars` is already a dependency, to replace
  the ad hoc CSV writing in notebooks.

## Ease Of Use E: Scan Construction Is Still Manual — DONE

Implemented 2026-08-15: `state_prep.scan_grid` builds the `(B, M)` arrays,
taking the outer product of a detuning axis and a prefactor axis, and supports
holding a physically fixed source at zero detuning while the scanned fields
move. `state_prep.SCAN_STORAGE_DEFAULTS` carries the final-only storage settings
for scans built on repeated `run()` calls.

### Original Note

Priority 1 and 2 of the existing backlog both list a scan convenience helper as
a next target; it has not been added. Callers must still:

- build `(B, M)`-shaped `detunings_hz` and `intensity_prefactors` by hand and
  satisfy the broadcasting rules documented in the docstring,
- remember that `intensity_prefactors` scales *intensity*, with couplings
  scaling as its square root,
- set six storage flags correctly to avoid the memory blowup described in
  Priority 2.

`benchmarks/common.py:256` already grew a private `make_detunings` helper, which
indicates the public API is missing this.

Recommendation:

- Add a scan-parameter builder for the common one-dimensional and
  two-dimensional sweep shapes.
- Add documented final-only scan defaults, so the storage flags do not need to
  be set individually.

## Ease Of Use F: Packaging And Documentation Drift — DONE

Implemented 2026-08-15: `README.md` rewritten around `uv sync` /
`uv pip install -e .` with a minimal example and scan usage; `setup.py` and
`state_prep.yml` removed (both superseded and actively wrong: the former
declared a different distribution name, the latter pinned py39 packages against
a project requiring 3.11+); `copilot-instructions.md` reduced to a pointer at
`AGENTS.md`.

### Original Note

- `README.md` instructs users to create a conda environment and run
  `python setup.py install`. The project now uses `uv` and `pyproject.toml`.
- `setup.py` and `state_prep.yml` are leftovers from the pre-`uv` layout and
  contradict the current metadata.
- `copilot-instructions.md` is now a subset of `AGENTS.md`.

Recommendation:

- Rewrite the `README.md` getting-started section around `uv sync` and
  `uv pip install -e .`.
- Remove or clearly mark `setup.py` and `state_prep.yml` as legacy.
- Reduce `copilot-instructions.md` to a pointer at `AGENTS.md`.

## Suggested Order Of Work

0. ~~The paired before/after benchmark runner.~~ Done: `benchmarks/paired.py`.
1. Ease Of Use B, the regression test. Unblocks everything else safely. Set its
   tolerance against the `3.835e-07` backend-agreement floor measured above, not
   against zero.
2. ~~Pin BLAS to one thread.~~ **Done**: `limit_blas_threads`, `blas_threads=1`
   by default on `run()` and `run_microwave_scan()`. `1.572x` end to end,
   bitwise identical.
3. ~~Hoist `V @ V_rot`.~~ **Done**: applied to both batched paths. `1.060x`,
   exact.
4. Ease Of Use A, the `__init__.py` re-exports. Small and immediate.
5. Ease Of Use C through F as maintenance capacity allows.

Removed from the order, all measured and rejected:

- Performance Priority A, the split-step propagator.
- Performance Priority C, threading the batch loop, and the `eig_backend`
  default switch that only existed to enable it. The loky path already present
  wins at matched worker counts.
- Old Priorities 8 and 9: they optimize an intercept worth `4-7%` at real batch
  sizes.

Available with no code change at all, in rough order of value:

1. Route notebook scans through `run_microwave_scan` rather than looping
   `run()`: `2.38x` at production shape, and currently used in exactly one
   notebook against 36 uses of `.run(`.
2. Use `workers=8` on production-sized scans: a further `3.29x`, bitwise
   identical, now that BLAS pinning has removed the oversubscription that made
   loky look useless.

Steps `0` and `1` together give a correctness oracle and a timing oracle. Every
later step is then a matter of running both and reading the result, rather than
arguing from the code.
