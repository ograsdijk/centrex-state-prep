from __future__ import annotations

import argparse

import numpy as np

from common import (
    add_nvidia_dll_paths,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)
from paired import compare_paired


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and CuPy for non-eigensolver GPU operations."
    )
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    add_nvidia_dll_paths()
    import cupy as cp
    import torch

    if not torch.cuda.is_available() or not cp.cuda.is_available():
        raise RuntimeError("Both PyTorch CUDA and CuPy CUDA must be available")

    n, batch, states, tones = (
        args.dimension,
        args.batch,
        args.initial_states,
        2,
    )
    rng = np.random.default_rng(1203)

    def random_complex(shape):
        values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return values.astype(np.complex128)

    def hermitian_batch(shape):
        values = random_complex(shape)
        return (values + values.conj().swapaxes(-1, -2)) / 2

    terms_np = [hermitian_batch((n, n)) for _ in range(7)]
    fields_np = rng.normal(scale=1e-3, size=(6,))
    terms_cp = [cp.asarray(value) for value in terms_np]
    terms_torch = [torch.as_tensor(value, device="cuda") for value in terms_np]
    fields_cp = cp.asarray(fields_np)
    fields_torch = torch.as_tensor(fields_np, device="cuda")

    V_np, _ = np.linalg.qr(random_complex((n, n)))
    V_rot_np, _ = np.linalg.qr(random_complex((batch, n, n)))
    eigs_np = np.broadcast_to(np.linspace(-1.0, 1.0, n), (batch, n)).copy()
    psi_np = random_complex((batch, states, n))
    H_mu_np = hermitian_batch((tones, n, n))
    weights_np = random_complex((batch, tones))
    diag_np = rng.normal(size=(batch, n))

    V_cp, V_torch = cp.asarray(V_np), torch.as_tensor(V_np, device="cuda")
    V_track_cp = cp.asarray(V_np[None, :, :])
    V_track_torch = torch.as_tensor(V_np[None, :, :], device="cuda")
    V_rot_cp, V_rot_torch = cp.asarray(V_rot_np), torch.as_tensor(V_rot_np, device="cuda")
    eigs_cp, eigs_torch = cp.asarray(eigs_np), torch.as_tensor(eigs_np, device="cuda")
    psi_cp, psi_torch = cp.asarray(psi_np), torch.as_tensor(psi_np, device="cuda")
    H_mu_cp, H_mu_torch = cp.asarray(H_mu_np), torch.as_tensor(H_mu_np, device="cuda")
    weights_cp, weights_torch = cp.asarray(weights_np), torch.as_tensor(weights_np, device="cuda")
    diag_cp, diag_torch = cp.asarray(diag_np), torch.as_tensor(diag_np, device="cuda")
    eig_ref_cp = cp.zeros((1, n), dtype=cp.float64)
    eig_ref_torch = torch.zeros((1, n), dtype=torch.float64, device="cuda")

    def slow_cupy():
        hff, hsx, hsy, hsz, hzx, hzy, hzz = terms_cp
        ex, ey, ez, bx, by, bz = fields_cp
        return 2 * np.pi * (
            hff + ex * hsx + ey * hsy + ez * hsz + bx * hzx + by * hzy + bz * hzz
        )

    def slow_torch():
        hff, hsx, hsy, hsz, hzx, hzy, hzz = terms_torch
        ex, ey, ez, bx, by, bz = fields_torch
        return 2 * np.pi * (
            hff + ex * hsx + ey * hsy + ez * hsz + bx * hzx + by * hzy + bz * hzz
        )

    def reorder_cupy():
        overlap = cp.abs(cp.matmul(cp.conj(V_track_cp.swapaxes(-2, -1)), V_track_cp))
        order = cp.argsort(cp.argmax(overlap, axis=-1), axis=-1)
        return cp.take_along_axis(eig_ref_cp, order, axis=-1), cp.take_along_axis(
            V_track_cp, order[..., None, :], axis=-1
        )

    def reorder_torch():
        overlap = torch.abs(torch.matmul(V_track_torch.mH, V_track_torch))
        order = torch.argsort(torch.argmax(overlap, dim=-1), dim=-1, stable=True)
        return torch.gather(eig_ref_torch, -1, order), torch.gather(
            V_track_torch, -1, order.unsqueeze(-2).expand(-1, n, -1)
        )

    diag_idx_cp = cp.arange(n)
    diag_idx_torch = torch.arange(n, device="cuda")

    def microwave_cupy():
        rotated = cp.matmul(cp.conj(V_cp.T), cp.matmul(H_mu_cp, V_cp))
        combined = cp.tensordot(weights_cp, rotated, axes=(1, 0))
        combined[:, diag_idx_cp, diag_idx_cp] += diag_cp
        return combined

    def microwave_torch():
        rotated = torch.matmul(V_torch.mH, torch.matmul(H_mu_torch, V_torch))
        combined = torch.tensordot(weights_torch, rotated, dims=([1], [0]))
        combined[:, diag_idx_torch, diag_idx_torch] += diag_torch
        return combined

    dt = 1e-9

    def update_cupy():
        psi_slow = cp.matmul(psi_cp, cp.conj(V_cp))
        phases = cp.exp(-1j * eigs_cp * dt)
        temp = cp.matmul(psi_slow, cp.conj(V_rot_cp))
        temp *= phases[:, None, :]
        psi_slow = cp.matmul(temp, cp.swapaxes(V_rot_cp, -2, -1))
        return cp.matmul(psi_slow, V_cp.T)

    def update_torch():
        psi_slow = torch.matmul(psi_torch, V_torch.conj())
        phases = torch.exp(-1j * eigs_torch * dt)
        temp = torch.matmul(psi_slow, V_rot_torch.conj())
        temp = temp * phases[:, None, :]
        psi_slow = torch.matmul(temp, V_rot_torch.transpose(-2, -1))
        return torch.matmul(psi_slow, V_torch.T)

    operations = {
        "slow_hamiltonian_assembly": (slow_cupy, slow_torch),
        # Historical operation comparison only. Production tracking now uses
        # optimal assignment and phase alignment, measured by bench_gpu_assignment.
        "legacy_greedy_eigenvector_reordering": (reorder_cupy, reorder_torch),
        "microwave_rotation_and_assembly": (microwave_cupy, microwave_torch),
        "state_update": (update_cupy, update_torch),
    }
    results = []
    for name, (cupy_fn, torch_fn) in operations.items():
        def repeated(fn):
            value = None
            for _ in range(args.iterations):
                value = fn()
            return value

        paired = compare_paired(
            {"cupy": lambda: repeated(cupy_fn), "torch": lambda: repeated(torch_fn)},
            repeat=args.repeat,
            warmup=args.warmup,
            baseline="cupy",
            sync=torch.cuda.synchronize,
            progress=False,
        )
        t_cp = paired.timings["cupy"]
        t_torch = paired.timings["torch"]
        ratio = t_cp.mean_s / t_torch.mean_s
        comparison = paired.comparisons[0]
        cupy_output = cupy_fn()
        torch_output = torch_fn()
        torch.cuda.synchronize()

        def output_parts(value):
            return value if isinstance(value, tuple) else (value,)

        max_difference = max(
            float(
                cp.max(
                    cp.abs(cupy_value - cp.from_dlpack(torch_value))
                ).get()
            )
            for cupy_value, torch_value in zip(
                output_parts(cupy_output), output_parts(torch_output)
            )
        )
        results.append(
            result_row(
                name,
                cupy_mean_s=t_cp.mean_s / args.iterations,
                cupy_std_s=t_cp.std_s / args.iterations,
                torch_mean_s=t_torch.mean_s / args.iterations,
                torch_std_s=t_torch.std_s / args.iterations,
                torch_speedup=ratio,
                paired_delta_mean_s=comparison.delta_mean_s / args.iterations,
                paired_ci95_low_s=comparison.ci95_s[0] / args.iterations,
                paired_ci95_high_s=comparison.ci95_s[1] / args.iterations,
                paired_verdict=comparison.verdict,
                max_abs_output_difference=max_difference,
                repeat=args.repeat,
                warmup=args.warmup,
                iterations_per_sample=args.iterations,
                batch=batch,
                n=n,
                initial_states=states,
            )
        )
        print(
            f"{name}: CuPy {t_cp.mean_s / args.iterations * 1e3:.3f} ms; "
            f"PyTorch {t_torch.mean_s / args.iterations * 1e3:.3f} ms; "
            f"PyTorch speedup {ratio:.3f}x ({comparison.verdict}); "
            f"max output diff={max_difference:.3e}",
            flush=True,
        )

    payload = make_payload(
        benchmark="gpu_torch_ops",
        config={
            "batch": batch,
            "dimension": n,
            "initial_states": states,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "iterations_per_sample": args.iterations,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_torch_ops"),
    )


if __name__ == "__main__":
    main()
