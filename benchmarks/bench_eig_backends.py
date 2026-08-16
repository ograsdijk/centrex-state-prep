"""Which eigensolver is fastest in the configuration the simulator actually runs in?

Settles why zheevd was originally chosen over eigh. Candidate explanations:

1. The original comparison was against `scipy.linalg.eigh`, whose default driver
   for complex Hermitian input is `evr`. `zheevd` is `evd` (divide and conquer),
   which is normally faster when all eigenvectors are wanted. `np.linalg.eigh`
   already uses `evd` internally, which would explain why it ties with zheevd.
2. OpenBLAS thread oversubscription under loky workers, where each of N workers
   spawns its own BLAS thread pool.
3. A different numpy/scipy/BLAS build at the time.

BLAS thread count must be set before numpy is imported, so the parent re-execs
itself as a child per BLAS setting.

    python check_eig_grid.py            # run the whole grid
    python check_eig_grid.py --child N  # internal
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPEAT = 3
SIZES = {64: 1200, 100: 600}  # n -> matrices, roughly equal total work


# These must be module level with self-contained imports. Nesting them inside
# child_main made loky capture `zheevd` in the closure, and scipy's f2py
# `fortran` objects are not picklable ("cannot pickle 'fortran' object").
def make_mats(n: int, count: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        out.append(np.ascontiguousarray(a + a.conj().T))
    return out


def solve(mats, backend: str) -> float:
    import numpy as np
    import scipy.linalg as sla
    from scipy.linalg.lapack import zheevd

    acc = 0.0
    if backend == "zheevd":
        for m in mats:
            w, v, info = zheevd(m)
            acc += float(w[0])
    elif backend == "np_eigh":
        for m in mats:
            w, v = np.linalg.eigh(m)
            acc += float(w[0])
    elif backend == "sp_eigh_evr":
        for m in mats:
            w, v = sla.eigh(m, driver="evr")
            acc += float(w[0])
    elif backend == "sp_eigh_evd":
        for m in mats:
            w, v = sla.eigh(m, driver="evd")
            acc += float(w[0])
    elif backend == "sp_eigh_default":
        for m in mats:
            w, v = sla.eigh(m)
            acc += float(w[0])
    else:
        raise ValueError(backend)
    return acc


def proc_worker(seed: int, n: int, count: int, backend: str) -> float:
    # Generate inside the worker: shipping matrices would dominate the timing.
    return solve(make_mats(n, count, seed), backend)


def child_main(blas_threads: str) -> int:
    import time
    from concurrent.futures import ThreadPoolExecutor
    from statistics import mean, stdev

    import numpy as np
    from joblib import Parallel, delayed

    backends = [
        "zheevd",
        "np_eigh",
        "sp_eigh_default",
        "sp_eigh_evr",
        "sp_eigh_evd",
    ]
    rows = []

    for n, count in SIZES.items():
        mats = make_mats(n, count, 0)

        for backend in backends:
            solve(mats[: min(50, count)], backend)  # warmup

            for mode in ("serial", "thread4", "thread8", "proc4"):
                times = []
                for _ in range(REPEAT):
                    if mode == "serial":
                        t0 = time.perf_counter()
                        solve(mats, backend)
                        dt = time.perf_counter() - t0
                    elif mode.startswith("thread"):
                        k = int(mode.replace("thread", ""))
                        bounds = np.linspace(0, count, k + 1).astype(int)
                        chunks = [mats[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
                        t0 = time.perf_counter()
                        with ThreadPoolExecutor(max_workers=k) as ex:
                            list(ex.map(lambda c: solve(c, backend), chunks))
                        dt = time.perf_counter() - t0
                    else:
                        k = 4
                        per = count // k
                        t0 = time.perf_counter()
                        Parallel(n_jobs=k, backend="loky")(
                            delayed(proc_worker)(seed, n, per, backend)
                            for seed in range(k)
                        )
                        dt = time.perf_counter() - t0
                    times.append(dt)

                m = mean(times)
                rows.append(
                    {
                        "blas_threads": blas_threads,
                        "n": n,
                        "count": count,
                        "backend": backend,
                        "mode": mode,
                        "mean_s": m,
                        "std_s": stdev(times) if len(times) > 1 else 0.0,
                        "per_solve_us": m / count * 1e6,
                    }
                )
                print(
                    f"  blas={blas_threads:<7} n={n:<4} {backend:<16} {mode:<8} "
                    f"{m:7.3f}s  {m / count * 1e6:7.1f} us/solve",
                    file=sys.stderr,
                    flush=True,
                )

    print(json.dumps(rows))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child")
    args = parser.parse_args()

    if args.child:
        return child_main(args.child)

    all_rows = []
    for blas in ("1", "default"):
        env = dict(os.environ)
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            if blas == "1":
                env[var] = "1"
            else:
                env.pop(var, None)

        print(f"=== BLAS threads: {blas} ===", file=sys.stderr, flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", blas],
            env=env,
            capture_output=True,
            text=True,
        )
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"child failed: {proc.returncode}", file=sys.stderr)
            return proc.returncode
        all_rows.extend(json.loads(proc.stdout.strip().splitlines()[-1]))

    print()
    print("=" * 92)
    print("SUMMARY: microseconds per eigensolve (lower is better)")
    print("=" * 92)

    backends = sorted({r["backend"] for r in all_rows})
    for n in sorted({r["n"] for r in all_rows}):
        for blas in ("1", "default"):
            print()
            print(f"n={n}, BLAS threads={blas}")
            header = f"{'backend':<18}" + "".join(
                f"{m:>12}" for m in ("serial", "thread4", "thread8", "proc4")
            )
            print(header)
            for backend in backends:
                cells = []
                for mode in ("serial", "thread4", "thread8", "proc4"):
                    match = [
                        r
                        for r in all_rows
                        if r["n"] == n
                        and r["blas_threads"] == blas
                        and r["backend"] == backend
                        and r["mode"] == mode
                    ]
                    cells.append(f"{match[0]['per_solve_us']:>11.1f}" if match else f"{'-':>12}")
                print(f"{backend:<18}" + "".join(cells))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eig_grid.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print()
    print(f"raw results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
