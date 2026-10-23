from __future__ import annotations

import argparse

import numpy as np

from bench_gpu_kernels import cuda_event_timing
from common import (
    add_nvidia_dll_paths,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CuPy batched and per-matrix Hermitian eigh.")
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    add_nvidia_dll_paths()
    from cupy_legacy import cupy
    import cupyx.cusolver

    cp = cupy()
    rng = np.random.default_rng(717)
    raw = rng.normal(size=(args.batch, args.n, args.n)) + 1j * rng.normal(
        size=(args.batch, args.n, args.n)
    )
    matrices = cp.asarray((raw + raw.conj().transpose(0, 2, 1)) / 2, dtype=cp.complex128)

    batched = lambda: cp.linalg.eigh(matrices)
    sequential = lambda: tuple(
        cp.stack(values, axis=0)
        for values in zip(*(cp.linalg.eigh(matrices[i]) for i in range(args.batch)))
    )
    direct_jacobi = lambda: cupyx.cusolver.syevj(matrices, "L", True)

    outputs = {}
    results = []
    for name, fn in (
        ("batched_cp_eigh_jacobi", batched),
        ("direct_cusolver_syevj", direct_jacobi),
        ("looped_2d_cp_eigh", sequential),
    ):
        timing = cuda_event_timing(cp, fn, warmup=args.warmup, repeat=args.repeat)
        outputs[name] = fn()
        cp.cuda.Stream.null.synchronize()
        print(
            f"{name}: {timing['time_mean_s']:.4f} ± {timing['time_std_s']:.4f}s "
            f"({timing['repeat']} repeats)"
        )
        results.append(result_row(name, **timing, batch=args.batch, n=args.n))

    ref_w, ref_v = outputs["batched_cp_eigh_jacobi"]
    for name, (w, v) in outputs.items():
        eig_error = float(cp.max(cp.abs(w - ref_w)).get())
        residual = float(cp.max(cp.linalg.norm(matrices @ v - v * w[:, None, :], axis=(-2, -1))).get())
        orthogonality = float(
            cp.max(cp.linalg.norm(cp.matmul(cp.conj(cp.swapaxes(v, -2, -1)), v) - cp.eye(args.n), axis=(-2, -1))).get()
        )
        print(
            f"  {name}: max eigenvalue difference={eig_error:.3e}, "
            f"residual={residual:.3e}, orthogonality={orthogonality:.3e}"
        )
        next(row for row in results if row["case"] == name).update(
            max_eigenvalue_difference=eig_error,
            residual=residual,
            orthogonality=orthogonality,
        )

    payload = make_payload(
        benchmark="gpu_eig_solvers",
        config={
            "batch": args.batch,
            "n": args.n,
            "repeat": args.repeat,
            "warmup": args.warmup,
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_eig_solvers"),
    )


if __name__ == "__main__":
    main()
