"""Retired real-scan runner; the standalone CuPy solver adapter is retained."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

RETIREMENT_MESSAGE = (
    "This real-scan CuPy Jacobi tolerance benchmark is retired after the "
    "PyTorch migration. PyTorch does not expose Jacobi tolerance controls. "
    "Use benchmarks/bench_gpu_spa2.py for the supported CPU/PyTorch comparison, "
    "or benchmarks/bench_gpu_eig_tolerance.py for the standalone legacy "
    "cuSOLVER experiment (requires CuPy)."
)


class BatchedJacobi:
    """Thin benchmark adapter for cuSOLVER's exposed batched Jacobi controls."""

    def __init__(self, cp: Any, *, batch: int, n: int, tolerance: float):
        import cupy_backends.cuda.libs.cublas as cublas
        import cupy_backends.cuda.libs.cusolver as cusolver

        self.cp = cp
        self.cublas = cublas
        self.cusolver = cusolver
        self.handle = cp.cuda.device.get_cusolver_handle()
        self.params = cusolver.createSyevjInfo()
        cusolver.xsyevjSetTolerance(self.params, tolerance)
        cusolver.xsyevjSetMaxSweeps(self.params, 100)
        self.batch = batch
        self.n = n
        self.w = cp.empty((batch, n), dtype=cp.float64)
        self.dev_info = cp.empty((batch,), dtype=cp.int32)
        self.query_a = cp.empty((batch, n, n), dtype=cp.complex128)
        self.lwork = cusolver.zheevjBatched_bufferSize(
            self.handle,
            cusolver.CUSOLVER_EIG_MODE_VECTOR,
            cublas.CUBLAS_FILL_MODE_LOWER,
            n,
            self.query_a.data.ptr,
            n,
            self.w.data.ptr,
            self.params,
            batch,
        )
        self.work = cp.empty((self.lwork,), dtype=cp.complex128)

    def __call__(self, matrix: Any, UPLO: str = "L") -> tuple[Any, Any]:
        if matrix.ndim != 3 or matrix.shape != (self.batch, self.n, self.n):
            raise ValueError("Jacobi adapter expects (batch,n,n) matrices")
        if UPLO != "L":
            raise ValueError("Jacobi adapter currently supports UPLO='L' only")
        cp = self.cp
        packed = cp.ascontiguousarray(cp.swapaxes(matrix, -2, -1))
        self.cusolver.zheevjBatched(
            self.handle,
            self.cusolver.CUSOLVER_EIG_MODE_VECTOR,
            self.cublas.CUBLAS_FILL_MODE_LOWER,
            self.n,
            packed.data.ptr,
            self.n,
            self.w.data.ptr,
            self.work.data.ptr,
            self.lwork,
            self.dev_info.data.ptr,
            self.params,
            self.batch,
        )
        return self.w, cp.swapaxes(packed, -2, -1)

    def close(self) -> None:
        self.cusolver.destroySyevjInfo(self.params)


def compare_candidate(
    *, setup: dict[str, Any], detunings: np.ndarray, powers: np.ndarray,
    n_steps: int, repeat: int, warmup: int, tolerance: float,
) -> dict[str, Any]:
    raise RuntimeError(RETIREMENT_MESSAGE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test faster Jacobi tolerances on real SPA2 propagation.")
    parser.add_argument("--n-steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-5, 1e-4])
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    parser.parse_args()

    parser.error(RETIREMENT_MESSAGE)


if __name__ == "__main__":
    main()
