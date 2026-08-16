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

#### Not yet checked — worth doing before relying on existing figures

The diagnosis above rests on **one grid cell** (`det = 0.0 MHz`, prefactor `1`),
chosen because it was the worst in the grid. It explains that cell cleanly, but
three things are unverified:

1. **Do the other 24 grid cells swap too, and at which step counts?** The swap
   partners seen here were `29`/`32` and `14`/`15`. Other detunings may have
   different near-degenerate pairs, or none.
2. **Are the published SPA2 background figures affected?** `depletion` only
   breaks if the run happens to land on the wrong side of a swap. Whether any
   figure actually did is unknown; nothing here shows one is wrong, only that
   the observable is capable of being wrong.
3. **Is `transferred` as safe as it looked?** It was stable to `5.3e-03` here,
   but it reads index `35` directly and would break if `34`/`35` ever swapped.
   That pair happened not to swap in this cell.

How to check: rerun the grid and, instead of comparing `probabilities_final`
elementwise, compare the two reduced observables and the sums over each
near-degenerate group. A swap shows up as a large elementwise difference with a
stable group sum, which is the signature already seen. The saved arrays from
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
