"""How well does the batch loop thread, with and without the GIL?

The case for a Rust port rests entirely on threading: `bench_inner_loop_ceiling.py`
shows the non-LAPACK share of the batch-loop body is only 4-10%, so a serial
port is worth at most about 1.1x. What Rust would buy is a batch loop that
threads without a GIL.

A free-threaded CPython build buys the same thing for the price of an
interpreter download, so it is measured first. If the free-threaded interpreter
reaches the scaling a native port could reach, the port has no remaining
justification.

This script deliberately imports nothing from `state_prep` or `centrex_tlf`.
Those ship compiled extensions built for cp311/cp313 only, so a free-threaded
interpreter cannot import them. It replays the captured batch-loop body from an
`.npz` written by:

    python benchmarks/bench_inner_loop_ceiling.py --dump-npz inner_loop.npz

then runs it under 1, 2, 4 and 8 threads on whichever interpreter invoked it:

    .venv/Scripts/python.exe benchmarks/bench_gil_thread_scaling.py inner_loop.npz
    <free-threaded python> benchmarks/bench_gil_thread_scaling.py inner_loop.npz

Only `numpy` and `scipy` are required.
"""

from __future__ import annotations

import argparse
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def gil_status() -> str:
    """Describe whether this interpreter is running with the GIL enabled."""
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if not free_threaded:
        return "GIL enabled (standard build)"
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_enabled is None:
        return "free-threaded build, GIL state unknown"
    return (
        "free-threaded build, GIL currently ENABLED "
        "(an extension re-enabled it - the measurement below is not GIL-free)"
        if is_enabled()
        else "free-threaded build, GIL DISABLED"
    )


def make_eig(backend: str):
    if backend == "zheevd":
        from scipy.linalg.lapack import zheevd

        def eig(M):
            w, v, info = zheevd(M)
            if info != 0:
                w, v = np.linalg.eigh(M)
            return w, v

        return eig

    def eig(M):
        return np.linalg.eigh(M)

    return eig


def run_chunk(lo, hi, *, H_mu_rot, cs, D, D_mu, psis_slow, dt, diag_idx, eig):
    """The shipped batch-loop body, restricted to scan points [lo, hi)."""
    for b in range(lo, hi):
        H_rot = (cs[b, 0] * H_mu_rot[0]).copy()
        for j in range(1, len(H_mu_rot)):
            H_rot += cs[b, j] * H_mu_rot[j]
        H_rot.flat[diag_idx] += D
        H_rot.flat[diag_idx] += D_mu[b]
        D_rot, V_rot = eig(H_rot)
        ph = np.exp(-1j * D_rot * dt)
        tmp = psis_slow[b] @ V_rot.conj()
        tmp *= ph[np.newaxis, :]
        psis_slow[b] = tmp @ V_rot.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", help="inputs from bench_inner_loop_ceiling.py --dump-npz")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--inner-reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--eig-backend", default="numpy", choices=["zheevd", "numpy"])
    args = parser.parse_args()

    data = np.load(args.npz)
    H_mu_rot = [np.ascontiguousarray(M) for M in data["H_mu_rot"]]
    cs = data["coupling_scales"]
    D = data["D"]
    D_mu = data["D_mu_diag_batch"]
    psis_ref = data["psis_slow"]
    dt = float(data["dt"])
    batch, _, n = psis_ref.shape
    diag_idx = slice(None, None, n + 1)

    eig = make_eig(args.eig_backend)
    reps = args.inner_reps

    print(f"python  : {sys.version.split()[0]}")
    print(f"gil     : {gil_status()}")
    print(f"backend : {args.eig_backend}")
    print(f"shape   : n={n}, batch={batch}, S={psis_ref.shape[1]}")
    try:
        from threadpoolctl import threadpool_info

        for info in threadpool_info():
            print(f"blas    : {info['internal_api']} threads={info['num_threads']}")
    except ImportError:
        print("blas    : threadpoolctl unavailable; set OMP/OPENBLAS_NUM_THREADS=1")
    print()

    def workload(pool, nthreads):
        def run():
            for _ in range(reps):
                psis_slow = psis_ref.copy()
                if nthreads == 1:
                    run_chunk(
                        0, batch, H_mu_rot=H_mu_rot, cs=cs, D=D, D_mu=D_mu,
                        psis_slow=psis_slow, dt=dt, diag_idx=diag_idx, eig=eig,
                    )
                else:
                    edges = np.linspace(0, batch, nthreads + 1).astype(int)
                    futures = [
                        pool.submit(
                            run_chunk, int(lo), int(hi),
                            H_mu_rot=H_mu_rot, cs=cs, D=D, D_mu=D_mu,
                            psis_slow=psis_slow, dt=dt, diag_idx=diag_idx, eig=eig,
                        )
                        for lo, hi in zip(edges[:-1], edges[1:])
                        if hi > lo
                    ]
                    for f in futures:
                        f.result()
            return psis_slow

        return run

    results = {}
    pools = {t: ThreadPoolExecutor(max_workers=t) for t in args.threads if t > 1}
    try:
        runners = {t: workload(pools.get(t), t) for t in args.threads}
        for _ in range(args.warmup):
            for run in runners.values():
                run()
        samples = {t: [] for t in args.threads}
        for _ in range(args.repeat):
            for t, run in runners.items():
                start = time.perf_counter()
                run()
                samples[t].append(time.perf_counter() - start)
        for t, times in samples.items():
            results[t] = (float(np.mean(times)), float(np.std(times)))
    finally:
        for pool in pools.values():
            pool.shutdown()

    base = results[args.threads[0]][0]
    print(f"{'threads':>8}{'mean':>12}{'std':>10}{'speedup':>10}{'efficiency':>13}")
    for t in args.threads:
        m, s = results[t]
        speedup = base / m
        print(f"{t:>8}{m:>11.4f}s{s:>9.4f}s{speedup:>9.2f}x{speedup / t:>12.0%}")

    print()
    print("Reference points from IMPROVEMENTS.md at batch 25, N_steps 4000, whole")
    print("scan rather than loop body alone: loky at 8 workers reached 3.29-3.77x,")
    print("Python threading reached 2.65x. Those include the shared per-timestep")
    print("work, which this harness excludes, so this number is the optimistic")
    print("upper bound for a threaded batch loop - which is exactly what a native")
    print("port would be competing for.")


if __name__ == "__main__":
    main()
