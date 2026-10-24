"""Square optimal assignment. CUDA costs and results stay on-device.

Uses the shortest augmenting path form of the Hungarian algorithm. Distinct
row minima are already globally optimal; conflicting matches use the full
solver. NVRTC comes with PyTorch, avoiding an nvcc/MSVC build dependency.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
from functools import lru_cache
from pathlib import Path
import sys


_SOURCE = r'''
#define INFINITY __int_as_float(0x7f800000)
typedef REAL real;
extern "C" __global__ void assignment(const real* costs, long long* orders) {
    const int n=SIZE, t=threadIdx.x;
    const real* a=costs+(long long)blockIdx.x*n*n;
    long long* out=orders+(long long)blockIdx.x*n;
    __shared__ real u[SIZE+1], v[SIZE+1], dist[SIZE+1], best[THREADS];
    __shared__ int p[SIZE+1], way[SIZE+1], used[SIZE+1], choice[THREADS];
    __shared__ int counts[SIZE], rowmin[SIZE], ambiguous, invalid, j0, finished;
    if (t==0) { ambiguous=0; invalid=0; }
    for (int j=t; j<n; j+=blockDim.x) counts[j]=0;
    __syncthreads();
    for (int i=t; i<n; i+=blockDim.x) {
        real value=(real)INFINITY; int col=0;
        for (int j=0; j<n; ++j) {
            real c=a[i*n+j];
            if (!isfinite(c)) atomicExch(&invalid,1);
            if (c<value) { value=c; col=j; }
        }
        rowmin[i]=col;
        if (atomicAdd(&counts[col],1)!=0) atomicExch(&ambiguous,1);
    }
    __syncthreads();
    if (invalid || !ambiguous) {
        for (int i=t; i<n; i+=blockDim.x) out[i]=invalid ? -1 : rowmin[i];
        return;
    }
    for (int j=t; j<=n; j+=blockDim.x) { u[j]=0; v[j]=0; p[j]=0; }
    __syncthreads();
    for (int row=1; row<=n; ++row) {
        for (int j=t; j<=n; j+=blockDim.x) {
            dist[j]=(real)INFINITY; used[j]=0; way[j]=0;
        }
        if (t==0) { p[0]=row; j0=0; finished=0; }
        __syncthreads();
        while (!finished) {
            if (t==0) used[j0]=1;
            __syncthreads();
            int i0=p[j0];
            real local=(real)INFINITY; int index=n+1;
            for (int j=t+1; j<=n; j+=blockDim.x) {
                if (used[j]) continue;
                real cur=a[(i0-1)*n+j-1]-u[i0]-v[j];
                if (cur<dist[j]) { dist[j]=cur; way[j]=j0; }
                if (dist[j]<local || (dist[j]==local && j<index)) {
                    local=dist[j]; index=j;
                }
            }
            best[t]=local; choice[t]=index;
            __syncthreads();
            for (int stride=blockDim.x/2; stride; stride/=2) {
                if (t<stride && (best[t+stride]<best[t] ||
                    (best[t+stride]==best[t] && choice[t+stride]<choice[t]))) {
                    best[t]=best[t+stride]; choice[t]=choice[t+stride];
                }
                __syncthreads();
            }
            real delta=best[0];
            for (int j=t; j<=n; j+=blockDim.x) {
                if (used[j]) { u[p[j]]+=delta; v[j]-=delta; }
                else dist[j]-=delta;
            }
            __syncthreads();
            if (t==0) { j0=choice[0]; finished=(p[j0]==0); }
            __syncthreads();
        }
        if (t==0) {
            do { int j1=way[j0]; p[j0]=p[j1]; j0=j1; } while (j0);
        }
        __syncthreads();
    }
    for (int j=t+1; j<=n; j+=blockDim.x) out[p[j]-1]=j-1;
}
'''


def _check(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"CUDA assignment {operation} failed (status {status})")


@lru_cache(maxsize=None)
def _kernel(device: int, n: int, real: str):
    import torch

    library_dir = Path(torch.__file__).parent / "lib"
    if sys.platform == "win32":
        candidates = sorted(library_dir.glob("nvrtc64_*.dll"))
        driver = ct.WinDLL("nvcuda.dll")
    else:
        candidates = sorted(library_dir.glob("libnvrtc.so*"))
        candidates += sorted((library_dir.parent.parent / "nvidia" / "cuda_nvrtc" / "lib").glob("libnvrtc.so*"))
        driver = ct.CDLL("libcuda.so.1")
    path = str(candidates[0]) if candidates else ct.util.find_library("nvrtc")
    if not path:
        raise RuntimeError("CUDA optimal assignment requires the NVRTC library supplied with CUDA-enabled PyTorch")
    nvrtc = ct.CDLL(path)
    threads = min(256, max(32, 1 << (n - 1).bit_length()))
    source = _SOURCE.replace("REAL", real).replace("SIZE", str(n)).replace("THREADS", str(threads))
    program = ct.c_void_p()
    _check(nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b"assignment.cu", 0, None, None), "create program")
    try:
        major, minor = torch.cuda.get_device_capability(device)
        flags = (ct.c_char_p * 2)(f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++11")
        status = nvrtc.nvrtcCompileProgram(program, len(flags), flags)
        if status:
            size = ct.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
            log = ct.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"CUDA assignment compilation failed: {log.value.decode()}")
        size = ct.c_size_t()
        _check(nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), "PTX size")
        ptx = ct.create_string_buffer(size.value)
        _check(nvrtc.nvrtcGetPTX(program, ptx), "PTX")
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
    module, function = ct.c_void_p(), ct.c_void_p()
    _check(driver.cuModuleLoadData(ct.byref(module), ptx), "load module")
    _check(driver.cuModuleGetFunction(ct.byref(function), module, b"assignment"), "get function")
    return driver, nvrtc, module, function, threads


def linear_assignment(cost):
    """Minimize finite float32/64 costs (..., n, n); return row -> column.

    CPU tensors use SciPy. CUDA uses its current device and stream, with cached
    runtime compilation. No CPU fallback or host synchronization is hidden.
    """
    import torch

    if cost.ndim < 2 or cost.shape[-1] != cost.shape[-2]:
        raise ValueError("assignment costs must be square (..., n, n)")
    if cost.dtype not in (torch.float32, torch.float64):
        raise ValueError("assignment costs must be float32 or float64")
    n = cost.shape[-1]
    if cost.device.type == "cpu":
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrices = cost.detach().numpy().reshape(-1, n, n) if n else []
        result = np.empty(cost.shape[:-1], dtype=np.int64)
        for matrix, output in zip(matrices, result.reshape(-1, n) if n else []):
            output[:] = linear_sum_assignment(matrix)[1]
        return torch.from_numpy(result)
    if cost.device.type != "cuda":
        raise ValueError("optimal assignment supports CPU and CUDA tensors")
    if n > 512:
        raise ValueError("CUDA assignment supports matrices up to 512 states")
    cost = cost.contiguous()
    result = torch.empty(cost.shape[:-1], dtype=torch.int64, device=cost.device)
    if not result.numel():
        return result
    with torch.cuda.device(cost.device):
        driver, _, _, function, threads = _kernel(cost.device.index, n, "double" if cost.dtype == torch.float64 else "float")
        inputs = (ct.c_void_p(cost.data_ptr()), ct.c_void_p(result.data_ptr()))
        args = (ct.c_void_p * 2)(*(ct.cast(ct.byref(value), ct.c_void_p) for value in inputs))
        stream = ct.c_void_p(torch.cuda.current_stream(cost.device).cuda_stream)
        _check(driver.cuLaunchKernel(function, result.numel() // n, 1, 1, threads, 1, 1, 0, stream, args, None), "launch")
        torch._assert_async((result >= 0).all(), "assignment costs must be finite")
    return result
