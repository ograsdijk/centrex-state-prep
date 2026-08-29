"""Why a uniform grid beats a graded one on SPA2-like problems.

Grading *does* reduce the summed local error -- equidistribution works. It loses
anyway, and this benchmark measures the reason: on a uniform grid the leading
local error telescopes, so the errors largely cancel against each other instead
of accumulating. A graded grid forfeits most of that cancellation, and the loss
is bigger than the placement gain.

The measured quantity is the **cancellation ratio**

    sum_i ||U_exact_step_i - U_scheme_step_i||   /   ||U_scheme_total - U_exact(T)||

i.e. how much larger the errors would be if they simply added up. It needs a
closed form for the per-step exact propagator, which `analytic_models` supplies:
`U_exact(t_{i+1}) U_exact(t_i)^H`. No reference run and no fitting is involved.

`IMPROVEMENTS.md` reported this cancellation as `17.7x` uniform against
`2.5x`/`4.5x` graded, from a configuration that was never recorded. This gives
`47.5x` against `8.4x` on `spa_like` -- different absolutes, same mechanism, and
the ratio that decides the verdict agrees. Absolute cancellation is
case-dependent; the ratio is what to quote.

`--timing` measures the cost of building a graded grid, with repeats, since a
single pair of timings is not evidence.

    python benchmarks/bench_grid_cancellation.py
    python benchmarks/bench_grid_cancellation.py --timing --repeats 7
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import analytic_models as am
from state_prep import build_time_grid, field_variation_density


def step_propagators(model, t_array, sampling: str):
    """Exact and frozen-midpoint one-step propagators for each interval."""
    offset = 0.5 if sampling == "mid" else 0.0
    exact, scheme = [], []
    for i in range(t_array.size - 1):
        t0, t1 = float(t_array[i]), float(t_array[i + 1])
        dt = t1 - t0
        U0, U1 = model.U_exact(t0), model.U_exact(t1)
        exact.append(U1 @ U0.conj().T)
        scheme.append(am.herm_expm(model.H_t(t0 + offset * dt), dt))
    return exact, scheme


def cancellation(model, t_array, sampling: str = "mid"):
    exact, scheme = step_propagators(model, t_array, sampling)
    local = sum(np.linalg.norm(a - b, 2) for a, b in zip(exact, scheme))
    total = np.eye(model.n, dtype=complex)
    for b in scheme:
        total = b @ total
    global_err = np.linalg.norm(total - model.U_exact(model.T), 2)
    return local, global_err, (local / global_err if global_err > 0 else np.inf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["spa_like", "rotating", "scalar"])
    ap.add_argument("--n-steps", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    print("cancellation ratio = sum of per-step errors / actual accumulated error")
    print("Higher is better: it is how much the grid's errors cancel.\n")
    print(f"{'model':<12}{'N':>7}{'grid':>9}{'sum local':>13}{'global':>12}"
          f"{'cancel':>10}{'kept':>8}")
    summary = {}
    for name in args.models:
        model = getattr(am, name)()
        if model.U_exact is None:
            print(f"{name:<12}  (no closed form, skipped)")
            continue
        if model.H_slow is None:
            # Grading equidistributes ||d(H_slow + H_mu)/dt||; a model with no
            # slow part has nothing to grade on.
            print(f"{name:<12}  (no H_slow, nothing to grade on, skipped)")
            continue
        density = field_variation_density(model.H_slow, model.muw_hams or [])
        for n in args.n_steps:
            for grid in ("uniform", "graded"):
                t_array = (np.linspace(0.0, model.T, n) if grid == "uniform"
                           else build_time_grid(model.T, n, density=density,
                                                order=args.order))
                loc, glob, ratio = cancellation(model, t_array)
                kept = 1.0 - 1.0 / ratio if np.isfinite(ratio) else 1.0
                summary.setdefault((name, grid), []).append(ratio)
                print(f"{name:<12}{n:>7}{grid:>9}{loc:>13.3e}{glob:>12.3e}"
                      f"{ratio:>9.1f}x{kept:>7.0%}")
        print()

    print("uniform vs graded cancellation, by model:")
    for name in args.models:
        u = summary.get((name, "uniform")); g = summary.get((name, "graded"))
        if not u or not g:
            continue
        print(f"  {name:<12} uniform {np.mean(u):>7.1f}x   graded {np.mean(g):>7.1f}x"
              f"   uniform advantage {np.mean(u)/np.mean(g):>5.2f}x")

    if args.timing:
        print(f"\ngrid construction cost at matched N_steps, {args.repeats} repeats")
        model = getattr(am, args.models[0])()
        density = field_variation_density(model.H_slow, model.muw_hams or [])
        n = max(args.n_steps)
        times = {"uniform": [], "graded": []}
        for _ in range(args.repeats):
            for grid in ("uniform", "graded"):
                t0 = time.perf_counter()
                if grid == "uniform":
                    np.linspace(0.0, model.T, n)
                else:
                    build_time_grid(model.T, n, density=density, order=args.order)
                times[grid].append(time.perf_counter() - t0)
        for grid in ("uniform", "graded"):
            v = np.array(times[grid]) * 1e3
            print(f"  {grid:<9} {v.mean():8.3f} ms +- {v.std():.3f}")
        over = np.mean(times["graded"]) / np.mean(times["uniform"]) - 1.0
        print(f"  graded overhead: {over:+.1%} of grid construction alone "
              f"(not of the run)")


if __name__ == "__main__":
    main()
