"""Benchmark-only native solver adapters; not production APIs or dependencies."""
from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np


class ReusableLapack:
    """Call SciPy's installed Cython LAPACK capsule with persistent scratch.

    Serial use only. Each solve still owns its output arrays; only LAPACK's
    scratch arrays are reused. The input is copied in Fortran layout.
    """
    def __init__(self):
        from scipy.linalg import cython_lapack
        get_name = ct.pythonapi.PyCapsule_GetName
        get_name.argtypes, get_name.restype = [ct.py_object], ct.c_char_p
        get_pointer = ct.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes, get_pointer.restype = [ct.py_object, ct.c_char_p], ct.c_void_p
        capsule = cython_lapack.__pyx_capi__["zheevd"]
        name = get_name(capsule)
        # Refuse an incompatible installed wrapper ABI rather than guessing.
        if name.count(b"*") != 13 or not name.startswith(b"void ("):
            raise RuntimeError(f"Unsupported LAPACK capsule: {name!r}")
        self.solve = ct.CFUNCTYPE(None, *([ct.c_void_p]*13))(get_pointer(capsule, name))
        self.workspaces = {}

    def __call__(self, matrix):
        from scipy.linalg.lapack import zheevd_lwork
        if matrix.dtype != np.complex128 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Prototype requires a complex128 matrix")
        n = matrix.shape[0]
        if n not in self.workspaces:
            lw, liw, lrw, info = zheevd_lwork(n)
            if info:
                raise RuntimeError(info)
            self.workspaces[n] = (np.empty(int(lw.real), complex),
                                  np.empty(int(lrw), float), np.empty(int(liw), np.int32))
        work, rwork, iwork = self.workspaces[n]
        a = np.array(matrix, order="F", copy=True)
        w = np.empty(n)
        ni, lda, lw, lrw, liw = map(ct.c_int, (n, n, len(work), len(rwork), len(iwork)))
        job, triangle, info = ct.c_char(b"V"), ct.c_char(b"U"), ct.c_int()
        self.solve(ct.byref(job), ct.byref(triangle), ct.byref(ni), a.ctypes.data,
                   ct.byref(lda), w.ctypes.data, work.ctypes.data, ct.byref(lw),
                   rwork.ctypes.data, ct.byref(lrw), iwork.ctypes.data, ct.byref(liw), ct.byref(info))
        return w, a, info.value


class ReusableCusolver:
    """Installed CUDA Hermitian drivers with cached native state/scratch.

    Complex128 only. ``flush`` must be called before accepting results when
    deferred_info=True. Each stream owns separate native state and scratch.
    CUDA calls are synchronous Python ctypes calls, not a custom CUDA kernel.
    """
    def __init__(self, deferred_info=False, driver='dc'):
        import torch
        self.torch = torch
        self.library = ct.CDLL(str(Path(torch.__file__).parent / "lib" / "cusolver64_12.dll"))
        signatures = {
            "cusolverDnCreate": [ct.POINTER(ct.c_void_p)],
            "cusolverDnCreateParams": [ct.POINTER(ct.c_void_p)],
            "cusolverDnDestroy": [ct.c_void_p],
            "cusolverDnDestroyParams": [ct.c_void_p],
            "cusolverDnSetStream": [ct.c_void_p, ct.c_void_p],
            "cusolverDnXsyevBatched_bufferSize": [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int,
                ct.c_int64, ct.c_int, ct.c_void_p, ct.c_int64, ct.c_int, ct.c_void_p,
                ct.c_int, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t), ct.c_int64],
            "cusolverDnXsyevBatched": [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int,
                ct.c_int64, ct.c_int, ct.c_void_p, ct.c_int64, ct.c_int, ct.c_void_p,
                ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_int64],
            "cusolverDnCreateSyevjInfo": [ct.POINTER(ct.c_void_p)],
            "cusolverDnDestroySyevjInfo": [ct.c_void_p],
            "cusolverDnXsyevjSetSortEig": [ct.c_void_p,ct.c_int],
            "cusolverDnZheevjBatched_bufferSize": [ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,
                ct.c_void_p,ct.c_int,ct.c_void_p,ct.POINTER(ct.c_int),ct.c_void_p,ct.c_int],
            "cusolverDnZheevjBatched": [ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,
                ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int],
        }
        for name, signature in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = signature, ct.c_int
        self.cache, self.pending = {}, []
        self.deferred_info = deferred_info
        if driver not in ('dc','jacobi'):
            raise ValueError(driver)
        self.driver = driver

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"cuSOLVER status {status}")

    def __call__(self, matrix, UPLO="L", **kwargs):
        torch = self.torch
        if (kwargs or matrix.dtype != torch.complex128 or matrix.device.type != "cuda"
                or matrix.ndim not in (2, 3) or matrix.shape[-2] != matrix.shape[-1]
                or matrix.requires_grad or UPLO not in ('L','U')):
            raise ValueError("Prototype requires CUDA complex128, rank 2 or 3, no out/grad options")
        n = matrix.shape[-1]
        batch = matrix.shape[0] if matrix.ndim == 3 else 1
        stream = torch.cuda.current_stream(matrix.device)
        triangle = 0 if UPLO == "L" else 1
        key = (matrix.device.index, stream.cuda_stream, n, batch, triangle)
        # Each matrix is column-major, while the batch dimension is contiguous.
        vectors = matrix.mT.contiguous().mT
        values = torch.empty(matrix.shape[:-1], device=matrix.device, dtype=torch.float64)
        info = torch.empty(batch, device=matrix.device, dtype=torch.int32)
        lib = self.library
        if key not in self.cache:
            handle, params = ct.c_void_p(), ct.c_void_p()
            self.check(lib.cusolverDnCreate(ct.byref(handle)))
            if self.driver == 'jacobi':
                self.check(lib.cusolverDnCreateSyevjInfo(ct.byref(params)))
                self.check(lib.cusolverDnXsyevjSetSortEig(params,1))
            else:
                self.check(lib.cusolverDnCreateParams(ct.byref(params)))
            self.check(lib.cusolverDnSetStream(handle, stream.cuda_stream))
            device_size, host_size = ct.c_size_t(), ct.c_size_t()
            if self.driver == 'jacobi':
                count = ct.c_int()
                self.check(lib.cusolverDnZheevjBatched_bufferSize(handle,1,triangle,n,vectors.data_ptr(),
                    n,values.data_ptr(),ct.byref(count),params,batch))
                device_work = torch.empty(count.value,device=matrix.device,dtype=torch.complex128)
            else:
                self.check(lib.cusolverDnXsyevBatched_bufferSize(handle, params, 1, triangle, n, 5,
                    vectors.data_ptr(), n, 1, values.data_ptr(), 5, ct.byref(device_size), ct.byref(host_size), batch))
                device_work = torch.empty(device_size.value, device=matrix.device, dtype=torch.uint8)
            host_work = ct.create_string_buffer(max(1, host_size.value))
            self.cache[key] = handle, params, device_work, host_work, host_size.value
        handle, params, device_work, host_work, host_size = self.cache[key]
        if self.driver == 'jacobi':
            self.check(lib.cusolverDnZheevjBatched(handle,1,triangle,n,vectors.data_ptr(),n,
                values.data_ptr(),device_work.data_ptr(),device_work.numel(),info.data_ptr(),params,batch))
        else:
            self.check(lib.cusolverDnXsyevBatched(handle, params, 1, triangle, n, 5,
                vectors.data_ptr(), n, 1, values.data_ptr(), 5, device_work.data_ptr(),
                device_work.numel(), host_work, host_size, info.data_ptr(), batch))
        if self.deferred_info:
            self.pending.append(info)
        elif bool(info.any().item()):
            raise RuntimeError(f"cuSOLVER convergence failure: {info.cpu().tolist()}")
        return values, vectors

    def flush(self):
        if self.pending:
            failed = self.torch.cat(self.pending).ne(0).any().item()
            self.pending.clear()
            if failed:
                raise RuntimeError("Deferred cuSOLVER convergence failure; discard run")

    def close(self):
        self.flush()
        self.torch.cuda.synchronize()
        for handle, params, *_ in self.cache.values():
            destroy = (self.library.cusolverDnDestroySyevjInfo if self.driver == 'jacobi'
                       else self.library.cusolverDnDestroyParams)
            self.check(destroy(params))
            self.check(self.library.cusolverDnDestroy(handle))
        self.cache.clear()
