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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep cuSOLVER batched Jacobi tolerance.")
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    add_nvidia_dll_paths()
    from cupy_legacy import cupy
    import cupy_backends.cuda.libs.cublas as cublas
    import cupy_backends.cuda.libs.cusolver as cusolver

    cp = cupy()
    rng = np.random.default_rng(88)
    raw = rng.normal(size=(args.batch, args.n, args.n)) + 1j * rng.normal(
        size=(args.batch, args.n, args.n)
    )
    h = cp.asarray((raw + raw.conj().transpose(0, 2, 1)) / 2, dtype=cp.complex128)
    ref_w, ref_v = cp.linalg.eigh(h)
    cp.cuda.Stream.null.synchronize()
    handle = cp.cuda.device.get_cusolver_handle()
    results = []

    for tolerance in (1e-7, 1e-6, 1e-5, 1e-4):
        params = cusolver.createSyevjInfo()
        cusolver.xsyevjSetTolerance(params, tolerance)
        cusolver.xsyevjSetMaxSweeps(params, 100)

        # Each matrix is packed as column-major Hermitian data for cuSOLVER.
        a = cp.ascontiguousarray(cp.swapaxes(h, -2, -1))
        w = cp.empty((args.batch, args.n), dtype=cp.float64)
        dev_info = cp.empty((args.batch,), dtype=cp.int32)
        lwork = cusolver.zheevjBatched_bufferSize(
            handle,
            cusolver.CUSOLVER_EIG_MODE_VECTOR,
            cublas.CUBLAS_FILL_MODE_LOWER,
            args.n,
            a.data.ptr,
            args.n,
            w.data.ptr,
            params,
            args.batch,
        )
        work = cp.empty((lwork,), dtype=cp.complex128)

        def solve() -> int:
            matrix_arg = cp.ascontiguousarray(cp.swapaxes(h, -2, -1))
            return cusolver.zheevjBatched(
                handle,
                cusolver.CUSOLVER_EIG_MODE_VECTOR,
                cublas.CUBLAS_FILL_MODE_LOWER,
                args.n,
                matrix_arg.data.ptr,
                args.n,
                w.data.ptr,
                work.data.ptr,
                lwork,
                dev_info.data.ptr,
                params,
                args.batch,
            )

        for _ in range(args.warmup):
            status = solve()
        cp.cuda.Stream.null.synchronize()
        times = []
        for _ in range(args.repeat):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            status = solve()
            end.record()
            end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end) / 1000.0)

        # cuSOLVER writes column-major eigenvectors into row-major storage.
        # `a` above is the pre-solve workspace query array; solve uses a new
        # matrix_arg. Recompute one result for its eigenvectors and diagnostics.
        matrix_arg = cp.ascontiguousarray(cp.swapaxes(h, -2, -1))
        status = cusolver.zheevjBatched(
            handle,
            cusolver.CUSOLVER_EIG_MODE_VECTOR,
            cublas.CUBLAS_FILL_MODE_LOWER,
            args.n,
            matrix_arg.data.ptr,
            args.n,
            w.data.ptr,
            work.data.ptr,
            lwork,
            dev_info.data.ptr,
            params,
            args.batch,
        )
        cp.cuda.Stream.null.synchronize()
        vectors = cp.swapaxes(matrix_arg, -2, -1)
        residual = cp.max(cp.linalg.norm(h @ vectors - vectors * w[:, None, :], axis=(-2, -1)))
        orthogonality = cp.max(
            cp.linalg.norm(cp.conj(cp.swapaxes(vectors, -2, -1)) @ vectors - cp.eye(args.n), axis=(-2, -1))
        )
        eig_error = cp.max(cp.abs(w - ref_w))
        try:
            sweeps = str(cusolver.xsyevjGetSweeps(handle, params))
        except Exception as exc:
            sweeps = f"unavailable ({type(exc).__name__})"
        try:
            solver_residual = f"{cusolver.xsyevjGetResidual(handle, params):.3e}"
        except Exception as exc:
            solver_residual = f"unavailable ({type(exc).__name__})"
        mean_s = float(np.mean(times))
        std_s = float(np.std(times, ddof=1))
        max_eig_error = float(eig_error.get())
        residual_value = float(residual.get())
        orthogonality_value = float(orthogonality.get())
        print(
            f"tol={tolerance:.0e}: {mean_s:.4f} ± {std_s:.4f}s, "
            f"status={status}, sweeps={sweeps}, solver residual={solver_residual}, "
            f"max eig diff={max_eig_error:.3e}, "
            f"residual={residual_value:.3e}, orthogonality={orthogonality_value:.3e}"
        )
        results.append(
            result_row(
                "cusolver_batched_jacobi",
                tolerance=tolerance,
                time_mean_s=mean_s,
                time_std_s=std_s,
                batch=args.batch,
                n=args.n,
                max_eigenvalue_difference=max_eig_error,
                residual=residual_value,
                orthogonality=orthogonality_value,
                status_code=str(status),
                sweeps=sweeps,
                solver_residual=solver_residual,
            )
        )
        cusolver.destroySyevjInfo(params)

    payload = make_payload(
        benchmark="gpu_eig_tolerance",
        config={
            "batch": args.batch,
            "n": args.n,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "matrix": "random hermitian complex128",
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_eig_tolerance"),
    )


if __name__ == "__main__":
    main()
