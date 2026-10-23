from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import (
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)


def hermitian(rng: np.random.Generator, n: int) -> np.ndarray:
    matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (matrix + matrix.conj().T) / 2


def cuda_event_timing(torch, fn, *, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return {
        "time_best_s": float(min(times)),
        "time_mean_s": float(sum(times) / len(times)),
        "time_std_s": float(np.std(times)),
        "repeat": repeat,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark low-level GPU kernels.")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gpu-timeout", type=float, default=30.0)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args._worker:
        run_parent(args)
        return

    run_worker(args)


def run_parent(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        worker_output = Path(tmp) / "gpu_kernel_worker.json"
        env = dict(os.environ)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--batch",
            str(args.batch),
            "--n",
            str(args.n),
            "--repeat",
            str(args.repeat),
            "--warmup",
            str(args.warmup),
            "--output",
            str(worker_output),
            "--_worker",
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(args.gpu_timeout),
            )
            if completed.returncode == 0 and worker_output.is_file():
                payload = json.loads(worker_output.read_text(encoding="utf-8"))
            else:
                payload = make_payload(
                    benchmark="gpu_kernels",
                    config=worker_config(args),
                    results=[
                        result_row(
                            "gpu_kernel_worker",
                            status="failed",
                            returncode=completed.returncode,
                            stdout=completed.stdout[-4000:],
                            stderr=completed.stderr[-4000:],
                        )
                    ],
                    include_gpu_env=True,
                )
        except subprocess.TimeoutExpired as exc:
            payload = make_payload(
                benchmark="gpu_kernels",
                config=worker_config(args),
                results=[
                    result_row(
                        "gpu_kernel_worker",
                        status="skipped",
                        error=f"GPU worker timed out after {args.gpu_timeout:.1f} s",
                        stdout=(exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                        stderr=(exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                    )
                ],
                include_gpu_env=True,
            )

    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_kernels"),
    )


def worker_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "batch": args.batch,
        "n": args.n,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "gpu_timeout": args.gpu_timeout,
    }


def run_worker(args: argparse.Namespace) -> None:
    results = []
    try:
        import torch
        from state_prep_gpu.slow_hamiltonian import (
            BatchedSlowHamiltonianTerms,
            slow_hamiltonian_batch,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch is unavailable")
        device = torch.device("cuda")
        rng = np.random.default_rng(123)
        terms_np = {
            key: hermitian(rng, args.n)
            for key in ("Hff", "HSx", "HSy", "HSz", "HZx", "HZy", "HZz")
        }
        terms = BatchedSlowHamiltonianTerms(
            **{
                key: torch.as_tensor(value, dtype=torch.complex128, device=device)
                for key, value in terms_np.items()
            }
        )
        e_field = torch.as_tensor(rng.normal(size=(args.batch, 3)), dtype=torch.float64, device=device)
        b_field = torch.as_tensor(rng.normal(size=(args.batch, 3)), dtype=torch.float64, device=device)
        out = torch.empty((args.batch, args.n, args.n), dtype=torch.complex128, device=device)

        expected = slow_hamiltonian_batch(terms, e_field, b_field)
        torch.testing.assert_close(
            slow_hamiltonian_batch(terms, e_field, b_field, out=out), expected
        )

        alloc_timing = cuda_event_timing(
            torch,
            lambda: slow_hamiltonian_batch(terms, e_field, b_field),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results.append(
            result_row(
                "slow_hamiltonian_allocate",
                **alloc_timing,
                batch=args.batch,
                n=args.n,
            )
        )

        out_timing = cuda_event_timing(
            torch,
            lambda: slow_hamiltonian_batch(terms, e_field, b_field, out=out),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results.append(
            result_row(
                "slow_hamiltonian_out",
                **out_timing,
                batch=args.batch,
                n=args.n,
                speedup_vs_allocate=alloc_timing["time_best_s"] / out_timing["time_best_s"]
                if out_timing["time_best_s"] > 0
                else None,
            )
        )
    except Exception as exc:
        results.append(result_row("gpu_kernel_setup", status="skipped", error=repr(exc)))

    payload = make_payload(
        benchmark="gpu_kernels",
        config=worker_config(args),
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_kernels"),
    )


if __name__ == "__main__":
    main()
