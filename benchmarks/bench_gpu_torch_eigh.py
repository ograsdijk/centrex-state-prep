from __future__ import annotations

import argparse
import time

import numpy as np

from bench_gpu_kernels import hermitian
from common import (
    add_nvidia_dll_paths,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)


def synchronized_time(fn, sync, *, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeat):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - start)
    return {
        "time_best_s": float(min(samples)),
        "time_mean_s": float(np.mean(samples)),
        "time_std_s": float(np.std(samples)),
        "repeat": repeat,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch batched CUDA eigh with the CuPy GPU baselines."
    )
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    add_nvidia_dll_paths()
    import cupy as cp
    import cupyx.cusolver
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    if not cp.cuda.is_available():
        raise RuntimeError("CuPy CUDA is unavailable")

    rng = np.random.default_rng(717)
    raw = rng.normal(size=(args.batch, args.n, args.n)) + 1j * rng.normal(
        size=(args.batch, args.n, args.n)
    )
    matrices_np = ((raw + raw.conj().transpose(0, 2, 1)) / 2).astype(np.complex128)
    matrices_cp = cp.asarray(matrices_np)
    matrices_torch = torch.as_tensor(matrices_np, dtype=torch.complex128, device="cuda")

    solvers = {
        "cupy_eigh": lambda: cp.linalg.eigh(matrices_cp),
        "cupy_direct_syevj": lambda: cupyx.cusolver.syevj(
            matrices_cp, "L", True
        ),
        "torch_eigh": lambda: torch.linalg.eigh(matrices_torch),
    }
    syncs = {
        "cupy_eigh": cp.cuda.Stream.null.synchronize,
        "cupy_direct_syevj": cp.cuda.Stream.null.synchronize,
        "torch_eigh": torch.cuda.synchronize,
    }
    for fn in solvers.values():
        for _ in range(args.warmup):
            fn()
    cp.cuda.Stream.null.synchronize()
    torch.cuda.synchronize()

    # Rotate solver order per round so each backend sees comparable GPU state.
    names = list(solvers)
    samples = {name: [] for name in names}
    for round_idx in range(args.repeat):
        order = names[round_idx % len(names) :] + names[: round_idx % len(names)]
        for name in order:
            timing = synchronized_time(
                solvers[name], syncs[name], warmup=0, repeat=1
            )
            samples[name].append(timing["time_mean_s"])

    outputs = {name: fn() for name, fn in solvers.items()}
    cp.cuda.Stream.null.synchronize()
    torch.cuda.synchronize()
    reference_w = outputs["cupy_eigh"][0]
    results = []
    for name in names:
        times = samples[name]
        row = result_row(
            name,
            time_best_s=float(min(times)),
            time_mean_s=float(np.mean(times)),
            time_std_s=float(np.std(times)),
            repeat=args.repeat,
            warmup=args.warmup,
            batch=args.batch,
            n=args.n,
        )
        w, v = outputs[name]
        if name == "torch_eigh":
            w = cp.from_dlpack(w)
            v = cp.from_dlpack(v)
        eig_error = float(cp.max(cp.abs(w - reference_w)).get())
        residual = float(
            cp.max(
                cp.linalg.norm(
                    matrices_cp @ v - v * w[:, None, :], axis=(-2, -1)
                )
            ).get()
        )
        orthogonality = float(
            cp.max(
                cp.linalg.norm(
                    cp.matmul(cp.conj(cp.swapaxes(v, -2, -1)), v)
                    - cp.eye(args.n),
                    axis=(-2, -1),
                )
            ).get()
        )
        row.update(
            max_eigenvalue_difference=eig_error,
            residual=residual,
            orthogonality=orthogonality,
        )
        results.append(row)
        print(
            f"{name}: {row['time_mean_s']:.4f} ± {row['time_std_s']:.4f}s; "
            f"max |Δλ|={eig_error:.3e}, residual={residual:.3e}, "
            f"orthogonality={orthogonality:.3e}",
            flush=True,
        )

    payload = make_payload(
        benchmark="gpu_torch_eigh",
        config={
            "batch": args.batch,
            "n": args.n,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_torch_eigh"),
    )


if __name__ == "__main__":
    main()
