# `state_prep_gpu`

Minimal batched GPU (CuPy) implementation for centrex-state-prep.

## Why this exists

The original `state_prep` code builds Hamiltonians through nested OOP calls. This
GPU path is intentionally *flat* and operates on *precomputed arrays* so it's
simple to use in ensemble scans.

## What you provide

- Precomputed slow-Hamiltonian terms (`Hff`, `HSx`, `HSy`, `HSz`, `HZx`, `HZy`, `HZz`).
  These come from `centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(...)`.
- Time series of field vectors `E_t`, `B_t` with shape `(T, B, 3)`.
- Optional precomputed microwave Hamiltonians `H_mu_t` in the QN basis with shape
  `(T-1, B, n, n)` and a rotating-frame/ detuning shift `D_mu`.

## API

- `state_prep_gpu.slow_hamiltonian_batch(terms, E, B)`
- `state_prep_gpu.simulate_batched(...)`

## CuPy

CuPy is optional and not imported at module import time. If you call a GPU
function without CuPy installed, you'll get an actionable error.

### Windows (recommended)

On Windows, CuPy often needs CUDA DLLs (NVRTC, cuBLAS, etc.) available via the
Windows DLL search path. This repo supports a **no-CUDA-Toolkit install** using
NVIDIA's pip packages.

In this repo's venv:

```powershell
uv pip install cupy-cuda12x
uv pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
uv pip install nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-curand-cu12 nvidia-cufft-cu12
uv pip install -e .
```

Notes:
- Your NVIDIA driver must support CUDA 12.x (your `nvidia-smi` prints a CUDA Version).
- You may see a CuPy warning about `CUDA_PATH` not being detected; that’s OK when
  using the pip-provided CUDA DLLs.

### Linux

Install an appropriate build for your CUDA version, e.g.

- `pip install cupy-cuda12x`

(Exact package depends on your system.)

## Tests / examples

- Smoke test (GPU runs + probabilities normalize): `src/state_prep_gpu/example_smoke.py`
- CPU vs GPU numerical check (slow-H only): `src/state_prep_gpu/example_compare_cpu.py`
- CPU vs GPU performance benchmark (slow-H only): `src/state_prep_gpu/benchmark_compare.py`
- CPU vs GPU benchmark including microwaves (SPA2-like scan): `src/state_prep_gpu/benchmark_spa2_real.py`
