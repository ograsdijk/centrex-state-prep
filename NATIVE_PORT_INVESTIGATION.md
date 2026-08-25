# Native Port Investigation

Generated: 2026-08-24

**Recommendation: do not port the inner loop to Rust.** Both independent
justifications for a port were measured and both fail. The serial ceiling is
`1.04x` to `1.11x`, and the threading gain a native port unlocks lands on top of
what the `loky` path in the repository already delivers, not above it.

A working Rust implementation of the batch loop was built and measured
(**Stage 1**, below). It is genuinely faster than Python at the loop body -
`1.79x` at `n=64`, `1.39x` at `n=100` - and it still does not move the
end-to-end number, because at that point the serial shared work dominates.

This is a decision document. No code under `src/` was changed and no packaging
was touched. The Rust spike lives in `rust/inner_loop_spike/`, untracked.

Relationship to the existing documents:

- `IMPROVEMENTS.md` holds the measured performance backlog and the benchmarking
  protocol every number here follows.
- `PERFORMANCE_BENCHMARK_RESULTS.md` records the earlier measured numbers this
  work re-tests rather than reuses.

Three benchmarks were added and one extended; they are listed under **Artifacts**
at the end.

## Summary Of Findings

| Question | Measured answer | Consequence |
| --- | --- | --- |
| What is the non-LAPACK share of the inner loop? | `9.7%` at `n=64`, `4.1%` at `n=100` | A serial port is worth at most `1.11x` / `1.04x` |
| Can the per-point eigensolve be removed by a first-Magnus propagator? | No. Error floors at `1.574e-04`, needs `N_steps ~ 5e5` to reach `1e-6` | The eigensolve stays; `2.49x` is available only at unacceptable accuracy |
| Does a GIL-free batch loop beat the current best? | Rust: `1.79x` on the loop body, but `8.38 s` end to end against `loky8`'s measured `8.11-9.28 s` | A wash |
| Does free-threaded CPython help today? | No. `1.42x` *slower* in absolute terms than the standard build | Not a shortcut, and not a replacement for the port either |

## The Measurement That Decides It

`IMPROVEMENTS.md:34` records the eigensolve at `75.6%` of shared-slow scan
runtime, measured by profiling at a different shape. Re-measured directly on the
batch-loop body, with inputs captured from a real timestep of a real SPA2 scan,
it is substantially higher than that:

`benchmarks/bench_inner_loop_ceiling.py`, batch `25`, `zheevd`, BLAS pinned to
`1`, `repeat=5`, variants interleaved.

`n=64` (`Js=[0,1,2,3]`), `500` loop bodies per timed call:

| Variant | Mean | Std | us/body |
| --- | ---: | ---: | ---: |
| `full` | `0.1338 s` | `0.0006 s` | `267.5` |
| `eig_only` | `0.1208 s` | `0.0004 s` | `241.5` |
| `no_eig` | `0.0102 s` | `0.0001 s` | `20.5` |
| `assemble_only` | `0.0037 s` | `0.0000 s` | `7.4` |

`n=100` (`Js=[0,1,2,3,4]`), `250` loop bodies per timed call:

| Variant | Mean | Std | us/body |
| --- | ---: | ---: | ---: |
| `full` | `0.1798 s` | `0.0009 s` | `719.2` |
| `eig_only` | `0.1724 s` | `0.0005 s` | `689.6` |
| `no_eig` | `0.0094 s` | `0.0001 s` | `37.6` |
| `assemble_only` | `0.0031 s` | `0.0000 s` | `12.3` |

| | `n=64` | `n=100` |
| --- | ---: | ---: |
| LAPACK share (`eig_only/full`) | `90.3%` | `95.9%` |
| Python-side share (`1 - eig_only/full`) | `9.7%` | `4.1%` |
| Python-side share (`no_eig/full`, independent) | `7.7%` | `5.2%` |
| of which `H_rot` assembly | `2.8%` | `1.7%` |
| **Serial native-port ceiling** | **`1.108x`** | **`1.043x`** |

The two independent estimates of the Python-side share agree to within `2%`,
so the split is not an artifact of how it was measured.

**A Rust port calls the same LAPACK.** Whatever it does to the surrounding
arithmetic, `90-96%` of the loop body is untouched. If every non-LAPACK
operation in the batch-loop body became free, the loop would speed up by `1.11x`
at `n=64` and `1.04x` at `n=100` — and the share shrinks as `n` grows, so the
larger notebooks benefit least. This alone retires the serial case.

The entire remaining case for a port is therefore threading, and that is the
only thing the rest of this document tests.

## Removing The Eigensolve: The Magnus Propagator — REJECTED

If LAPACK is `90-96%` of the loop, the only large lever is not calling it.
`IMPROVEMENTS.md:598` identifies the candidate and explicitly leaves it
unimplemented pending an accuracy check. That check has now been run, twice, by
two different methods.

The propagator moves to the interaction picture with respect to the diagonal,
which is where the large `GHz` energies live, and keeps only the first Magnus
term:

```
delta_b = D + D_mu_diag_batch[b]
M_ij    = (exp(i*delta_i*dt) * conj(exp(i*delta_j*dt)) - 1) / (i*(delta_i - delta_j))
A       = H_mu_assembled . M                  (elementwise; A is Hermitian)
psi    <- (psi @ expm(-i*A)^T) * exp(-i*delta_b*dt)
```

Two properties make it cheap, and both hold as measured. `M` needs only `n`
complex exponentials rather than `n^2`, because `exp(i*(delta_i - delta_j)*dt)`
factorises into an outer product. And `expm(-i*A)` is never formed: measured
`||A||_2 = 4.3146e-02` rad at production `dt`, so an `8`-term Taylor series
applied directly to the `S=4` state vectors truncates near `1e-18` at `n^2*S`
cost instead of `n^3`.

The small-argument branch in `M` is load-bearing rather than defensive: `delta`
is nearly degenerate within a rotational manifold, so the quotient divides by
approximately zero there.

### Per-Step Error

`benchmarks/bench_trotter_conditioning.py`, extended with the Magnus variant
alongside the existing Strang and Lie ones so all three sit in one table. Real
rotating-frame Hamiltonian, `9` sample times along the trajectory, spectral norm
of `U_exact - U_approx`:

| `N_steps` | `dt` | Strang | Lie | **Magnus** |
| ---: | ---: | ---: | ---: | ---: |
| `10000` | `1.522e-07 s` | `1.392e-03` | `1.700e-02` | **`2.072e-04`** |
| `20000` | `7.609e-08 s` | `2.188e-04` | `4.332e-03` | **`2.679e-05`** |
| `40000` | `3.804e-08 s` | `1.032e-04` | `1.088e-03` | **`3.376e-06`** |

Magnus is `67x` better per step than Strang at production `N_steps`, and its
per-step error falls `7.74x` and `7.93x` as `dt` halves — third-order local
error, cleanly converging, where Strang stalls at `2.12x`. The diagnosis in
`IMPROVEMENTS.md` was right: handling the diagonal exactly instead of splitting
it is a genuinely better scheme.

It is still not good enough. Summed over all steps the error is `2.072` at
`N_steps=10000`, which exceeds the trivial bound of `2` for a difference of
unitaries and is therefore vacuous — the same signature that condemned Strang.

### End-To-End Error, Because The Bound Is Vacuous

Rejecting a propagator on a bound that returns a vacuous number would repeat
the reasoning failure this repository has documented twice. So the propagator
was implemented against the real simulator and the observable was compared
directly.

`benchmarks/bench_magnus_propagator.py`, SPA2 setup, batch `5`, `zheevd`,
`repeat=2`, variants interleaved, `max|probabilities_final - reference|` against
the shipped propagator at the *same* `N_steps` so propagator error is isolated
from discretisation error:

| `N_steps` | Exact | Magnus | Speedup | Max abs diff |
| ---: | ---: | ---: | ---: | ---: |
| `10000` | `19.782 s +/- 0.065` | `10.554 s +/- 0.002` | `1.87x` | `2.444e-03` |
| `20000` | `47.643 s +/- 2.023` | `22.121 s +/- 0.463` | `2.15x` | `6.260e-04` |
| `40000` | `79.519 s +/- 0.714` | `42.041 s +/- 0.008` | `1.89x` | `1.574e-04` |

And at production batch size, where the per-scan-point loop is `93%` of runtime:

| `N_steps` | Batch | Exact | Magnus | Speedup | Max abs diff |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `4000` | `25` | `29.925 s +/- 0.001` | `12.031 s +/- 0.032` | **`2.49x`** | `1.292e-02` |

The observable error falls `3.90x` then `3.98x` as `dt` halves: second order
globally, consistent with the third-order local error. Extrapolating from
`1.574e-04` at `N_steps=40000`, reaching the `1e-6` acceptance target needs a
`157x` error reduction, hence `sqrt(157) = 12.5x` more steps, i.e.
`N_steps ~ 5e5`. At the measured `42.04 s` per `40000` steps that is about
`525 s`, against `19.8 s` for the current propagator at production
`N_steps=10000` — roughly `26x` *slower* for matched accuracy.

**Verdict: rejected.** The speedup is real (`2.49x` at production batch size)
but it is not available at an acceptable tolerance, and it cannot be bought back
by refining `dt`. Adding the second Magnus term would restore the accuracy and
destroy the economics in the same stroke: the term is a commutator, which is
`n^3`, which is the cost the scheme exists to avoid.

This is the third propagator measured and rejected for this loop, after Strang
and Lie. The per-scan-point eigensolve is looking less like an implementation
choice and more like a requirement of resolving `GHz` splittings over `150 ns`
steps.

## Threading: What A Port Would Actually Buy

With the serial case dead and the algorithmic case dead, the port stands or
falls on threading the batch loop without a GIL.

That quantity can be measured without writing any Rust. A free-threaded CPython
build runs the *same* NumPy and LAPACK calls on the *same* machine with the GIL
out of the way, which is the closest available proxy for what a native threaded
loop would experience.

`benchmarks/bench_gil_thread_scaling.py` replays the captured batch-loop body
under `1`, `2`, `4` and `8` threads. It imports nothing from `state_prep` or
`centrex_tlf` — those ship compiled extensions for `cp311`/`cp313` only and
cannot be imported by a free-threaded interpreter — so it reads its inputs from
an `.npz` written by `bench_inner_loop_ceiling.py --dump-npz`.

`n=64`, batch `25`, BLAS pinned to `1` thread and confirmed via `threadpoolctl`,
`repeat=5`:

| Interpreter | numpy / scipy | Backend | `1` thread | `8` threads | Scaling |
| --- | --- | --- | ---: | ---: | ---: |
| `3.11.13` GIL (project venv) | `2.3.2` / `1.17.1` | `numpy` | `0.1352 s` | `0.0464 s` | `2.92x` |
| `3.11.13` GIL (project venv) | `2.3.2` / `1.17.1` | `zheevd` | `0.1338 s` | `0.1556 s` | `0.86x` |
| `3.13.9` GIL (matched) | `2.5.2` / `1.16.3` | `numpy` | `0.1400 s` | `0.0488 s` | `2.87x` |
| `3.13.9` GIL (matched) | `2.5.2` / `1.16.3` | `zheevd` | `0.1425 s` | `0.1601 s` | `0.89x` |
| `3.13.15t` free-threaded | `2.5.2` / `1.16.3` | `zheevd` | `0.3077 s` | **`0.0692 s`** | **`4.45x`** |
| `3.13.15t` free-threaded | `2.5.2` / `1.16.3` | `numpy` | `0.6221 s` | `0.4741 s` | `1.31x` |

The matched-version `3.13.9` row exists to separate free-threading from the
library bump, and it settles that question: at identical `numpy` and `scipy`
versions the standard build reproduces the project venv to within `5%`, so the
free-threaded interpreter's slower single-thread time is a property of
free-threading, not of the wheels.

Reading the table:

- **Free-threading does unlock `zheevd`.** It goes from `0.86x` (the documented
  GIL-holding f2py wrapper) to `4.45x` on `8` threads, `83%` efficient at `4`.
  That is the highest scaling measured anywhere in this repository for the batch
  loop.
- **And it loses anyway.** Single-threaded it costs `0.3077 s` against
  `0.1338 s`, a `2.16x` free-threading penalty, so the best absolute time it
  reaches is `0.0692 s` against the standard build's `0.0464 s`. Free-threaded
  CPython is `1.42x` slower here in the only terms that matter.
- The free-threaded `numpy` row is confounded and should not be read as a
  result: `threadpoolctl` detected no BLAS at all in that wheel, which is the
  likely explanation for the `4.6x` single-thread anomaly.
- **Threading is not GIL-bound as previously believed.** At `n=64` only `9.7%`
  of the loop body holds the GIL, which predicts a `4.7x` ceiling at `8`
  threads, yet the standard build reaches `2.92x` on the isolated loop body with
  efficiency collapsing from `88%` at `2` threads to `36%` at `8`. Even the
  GIL-free build tails off from `83%` to `56%`. Something shared — memory
  bandwidth, LAPACK workspace allocation, or clock throttling under load —
  binds before the GIL does, and a native port inherits that limit unchanged.

### Stage 1: The Rust Implementation, Measured

The projection above was replaced by a measurement. `rust/inner_loop_spike/` is
a standalone Rust binary - no PyO3, no build matrix, no change to the Python
package - that replays the same captured inputs, calls the same LAPACK `evd`
driver out of the same OpenBLAS DLL, and reports the same table.

It resolves `scipy_zheevd_` at runtime with `LoadLibraryW`/`GetProcAddress`,
following `../CeNTREX-TlF/rust/src/lindblad/blas.rs`, which does exactly this
for `zher2k` in `135` lines. Two properties are inherited from that DLL and were
confirmed rather than assumed: symbols are prefixed (`scipy_zheevd_`, not
`zheevd_`), and the build is ILP64, so every Fortran INTEGER is `i64`.

Threading is `rayon` work-stealing over individual scan points, with LAPACK
workspace allocated once per worker rather than per point.

`n=64`, batch `25`, BLAS pinned to `1`, `20` inner reps, `repeat=5`,
interleaved:

| Threads | Rust | Std | Speedup | Efficiency | Python best (`numpy`, threaded) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `0.1233 s` | `0.0008 s` | `1.00x` | `100%` | `0.1352 s` |
| `2` | `0.0690 s` | `0.0012 s` | `1.79x` | `89%` | `0.0766 s` |
| `4` | `0.0385 s` | `0.0007 s` | `3.20x` | `80%` | `0.0506 s` |
| `8` | **`0.0259 s`** | `0.0006 s` | **`4.76x`** | `60%` | **`0.0464 s`** |

`n=100`, `10` inner reps:

| Threads | Rust | Std | Speedup | Python best (`numpy`, threaded) |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0.1761 s` | `0.0009 s` | `1.00x` | `0.1757 s` |
| `8` | **`0.0332 s`** | `0.0011 s` | **`5.31x`** | **`0.0462 s`** |

Reading the result:

- **E0's prediction was accurate.** Rust's serial time is `0.1233 s` against
  Python's `0.1338 s` at the same backend: `1.09x`, against the `1.108x` ceiling
  E0 predicted from the LAPACK share. At `n=100` it is `0.1761 s` against
  `0.1757 s` — `1.00x`, against a predicted `1.043x`. Removing all Python
  dispatch buys what the split said it would buy, and no more.
- **Rust scales better than anything else measured**: `4.76x` at `n=64` and
  `5.31x` at `n=100`, beating free-threaded CPython's `4.45x` and Python
  threading's `2.92x`. `rayon` work-stealing is the best available answer to
  the batch loop.
- **On the loop body it wins clearly**: `1.79x` at `n=64` (`0.0259 s` against
  `0.0464 s`) and `1.39x` at `n=100`.
- **Efficiency still falls off at `8` threads** (`60%`, `66%`), reproducing the
  same tail-off seen in both Python builds. That confirms the non-GIL shared
  limit diagnosed earlier: it is not an interpreter artifact, and Rust does not
  escape it.

### Why A Clear Loop-Body Win Is Still A Wash

The end-to-end consequence, using the `10%` shared / `90%` per-point split
measured at this shape in `IMPROVEMENTS.md:818` and the `30.56 s` serial
baseline recorded there. The shared portion stays in Python and stays serial,
since `H_slow_t` is a Python callable into `centrex_tlf`:

```
3.06 s  shared, unchanged, serial
5.32 s  27.5 s * (0.0259 / 0.1338)
-----
8.38 s  end to end, batch 25, N_steps 4000
```

Against `loky8`, measured at `9.28 s` and `8.11 s` in two separate sessions.
`8.38 s` is **inside that spread**. Expressed against the serial baseline, Rust
gives `3.65x` where `loky8` gives `3.29-3.77x`.

A `1.79x` win on the loop body evaporates into a tie because Amdahl takes it:
once the per-point work is `5.32 s`, the untouched `3.06 s` of shared work is
`37%` of the total. The remaining headroom is in the part Rust did not touch,
and touching it means porting `H_slow_t` and the slow eigensolve too — i.e.
porting the simulator, not the loop.

And the reason `loky` matches this without any of it is the inversion recorded
at `IMPROVEMENTS.md:845`: `loky` parallelises the shared work as well, by
repeating it redundantly in each process. That converts the serial bottleneck
into parallel redundant work. A threaded port cannot, which is precisely the
`3.06 s` that costs Rust its lead.

## Recommendation

**Do not port.** Specifically:

1. **No Rust extension.** The serial case is worth `1.04-1.11x` (predicted, then
   confirmed at `1.09x` and `1.00x` by a working implementation), and the
   threading case measures as a wash against the `loky` path already shipped:
   `8.38 s` against `8.11-9.28 s`. The Rust loop body is genuinely `1.79x`
   faster and it does not matter, because the shared work it cannot reach then
   accounts for `37%` of the runtime.
2. **No free-threaded interpreter.** It is `1.42x` slower in absolute terms
   today. Worth re-testing when the free-threading single-thread penalty closes,
   since its `4.45x` scaling is the best measured — but that is a "watch", not a
   task.
3. **The Magnus propagator is closed**, alongside Strang and Lie. Reopen it only
   with a scheme that restores accuracy without an `n^3` term, which the second
   Magnus term is not.
4. **The actionable performance work remains what `IMPROVEMENTS.md` already
   says**, and it needs no new toolchain: route notebook scans through
   `run_microwave_scan` rather than looping `run()` (`2.38x`, and still used in
   one notebook against 36 uses of `.run(`), and use `workers=8` on
   production-sized scans (`3.29x`, bitwise identical).

The cross-platform distribution question (`faer` versus vendored LAPACK versus
`dlopen`) and the CI cost were scoped as experiment E3/E4 in the plan. They were
**not run**, because they only price a port that is not recommended. If the
recommendation is ever revisited, they are the next things to measure. Note for
that eventuality: `../CeNTREX-TlF` sidesteps the whole problem by committing
`centrex_tlf_rust.cp311-win_amd64.pyd` and loading BLAS from the installed
`scipy-openblas` at runtime via `LoadLibraryW` — a Windows-only route that the
cross-platform-wheels requirement rules out.

## What Is Measured Versus What Is Inferred

Stated plainly, because the recommendation now rests on one composed number
rather than a projected one:

**Measured**: every figure in the serial-ceiling table; the Magnus per-step and
end-to-end errors and speedups; the whole thread-scaling matrix; and the Rust
implementation's serial and threaded times at `n=64` and `n=100`, validated
against the NumPy reference to `3.7e-09` / `9.1e-09`.

**Inferred**: the `8.38 s` end-to-end figure. It composes a *measured* Rust loop
body with a *measured* `10%`/`90%` shared/per-point split at the same shape. It
is not an end-to-end run of a Rust-integrated simulator, because building that
means the PyO3 boundary, which Stage 1 deliberately skipped.

What would change the conclusion: only a large reduction in the shared `3.06 s`,
since that is `37%` of the projected total and Rust cannot reach it from inside
the batch loop. Nothing in the current design offers that. The residual
uncertainty in the PyO3 boundary itself runs the wrong way for the port — it can
only add overhead to the `8.38 s`, never subtract from it.

The one honest caveat in Rust's favour: the loop body's non-LAPACK arithmetic is
hand-written scalar Rust, not tuned BLAS. It could be somewhat faster still. But
E0 bounds that entire category at `9.7%` of the loop body, so the ceiling it
could add is already inside the `1.09x` serial figure.

## Benchmarking Protocol

Every timing here follows `IMPROVEMENTS.md:48`: variants interleaved round-robin
within a single session rather than blocked, `warmup >= 1`, `repeat >= 2` with
means and standard deviations reported together, BLAS pinned to one thread, and
two problem sizes wherever a conclusion could be shape-dependent. Numerical
agreement is reported beside every timing for anything that changes results.

`pytest` was run before and after: `23` passed, `1` skipped (GPU, no CuPy), no
change. (`CLAUDE.md` and `IMPROVEMENTS.md` both still say `20` tests; the suite
has grown since those were written.) Nothing under `src/` was modified, so this is a statement about the
benchmarks, not about the engine.

## Artifacts

Added:

- `benchmarks/bench_inner_loop_ceiling.py` — splits the batch-loop body into
  LAPACK and Python-side time on inputs captured from a real scan timestep. Also
  writes those inputs with `--dump-npz`.
- `benchmarks/bench_magnus_propagator.py` — the interaction-picture Magnus
  propagator run against the real simulator, for accuracy and speed.
- `benchmarks/bench_gil_thread_scaling.py` — thread scaling of the captured loop
  body, dependency-free enough to run on a free-threaded interpreter.

- `rust/inner_loop_spike/` — the Stage 1 Rust implementation: `lapack.rs` binds
  `scipy_zheevd_` at runtime, `main.rs` runs the batch loop under `rayon` and
  validates against the NumPy reference before timing. Untracked, via a `rust/`
  entry in `.gitignore`.

Extended:

- `benchmarks/bench_trotter_conditioning.py` — the Magnus propagator now sits
  beside the Strang and Lie variants in the same per-step error table.

Reproducing the four results, in order:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_inner_loop_ceiling.py --js 0,1,2,3 --batch 25
.\.venv\Scripts\python.exe benchmarks\bench_inner_loop_ceiling.py --js 0,1,2,3,4 --batch 25 --inner-reps 10
.\.venv\Scripts\python.exe benchmarks\bench_trotter_conditioning.py
.\.venv\Scripts\python.exe benchmarks\bench_magnus_propagator.py --batch 5 --n-steps 10000 20000 40000 --repeat 2
```

The thread-scaling matrix needs a free-threaded interpreter, which `uv` will
fetch:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_inner_loop_ceiling.py --batch 25 --dump-npz inner_loop_n64.npz
uv python install 3.13t
uv venv --python 3.13t ft
uv pip install --python ft\Scripts\python.exe numpy "scipy==1.16.*" threadpoolctl
$env:OPENBLAS_NUM_THREADS=1; $env:OMP_NUM_THREADS=1
.\.venv\Scripts\python.exe benchmarks\bench_gil_thread_scaling.py inner_loop_n64.npz --eig-backend zheevd
ft\Scripts\python.exe benchmarks\bench_gil_thread_scaling.py inner_loop_n64.npz --eig-backend zheevd
```

`scipy` is pinned below `1.17` because no free-threaded wheel is published for
`1.18` and it falls back to a source build that fails on Windows without MSVC.

The Rust spike, which needs the flat binary dump and the path to the OpenBLAS
DLL that SciPy ships:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_inner_loop_ceiling.py --batch 25 --dump-bin inner_loop_n64.bin
cargo build --release --manifest-path rust\inner_loop_spike\Cargo.toml
$dll = .\.venv\Scripts\python.exe -c "import glob,os,scipy;print(os.path.abspath(glob.glob(os.path.join(os.path.dirname(scipy.__file__),'..','scipy.libs','*openblas*.dll'))[0]))"
$env:OPENBLAS_NUM_THREADS=1
rust\inner_loop_spike\target\release\inner_loop_spike.exe inner_loop_n64.bin $dll
```
