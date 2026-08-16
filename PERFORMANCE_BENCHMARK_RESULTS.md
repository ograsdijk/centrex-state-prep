# Performance Benchmark Results

> **Numbers below are single-shot. Read `IMPROVEMENTS.md` for the re-measured
> figures.**
>
> These were recorded before the paired benchmark runner existed, so none carry
> a spread and several cannot be distinguished from no change at all. The
> `1.06x` real-SPA2 GPU result in particular has never been shown to differ
> from unity. Later runs with repeats and interleaving are in `IMPROVEMENTS.md`,
> together with the environment they were taken in.

Generated: 2026-06-14

This file records benchmarks run after the performance scan in `PERFORMANCE_IMPROVEMENTS.md`.

## Environment

- OS/shell: Windows / PowerShell
- Python: 3.11.13 from `.venv`
- NumPy: 2.4.0
- SciPy: 1.16.3
- CuPy: 13.6.0
- GPU: NVIDIA GeForce RTX 5070 Ti, driver 591.86, 16303 MiB
- CPU model: not recorded; `Get-CimInstance Win32_Processor` returned access denied.

GPU benchmark note: CuPy could see the GPU, but NVRTC initially failed to load `nvrtc-builtins64_129.dll`. The DLL exists under `.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin`, so GPU benchmark commands were run with the NVIDIA pip-package `bin` directories prepended to `PATH`.

## Summary

| Benchmark | Result |
| --- | ---: |
| Synthetic slow-H CPU vs GPU | GPU 3.46x faster |
| Real SPA2-like shared-slow CPU vs GPU | GPU 1.06x faster |
| CPU output storage: full vs final-only | final-only 1.17x faster, much less memory |
| CPU output storage: full vs downsampled every 20 | downsampled 1.17x faster, 20x less stored output |
| CPU microwave scan: shared-slow vs repeated runs | shared-slow 1.90x faster |
| GPU `slow_hamiltonian_batch(out=...)` microbenchmark | inconclusive; timed out repeatedly |

## Benchmark 1: Synthetic Batched Slow-Hamiltonian CPU vs GPU

Command:

```powershell
$nvidia = Join-Path (Get-Location) '.venv\Lib\site-packages\nvidia'
$bins = Get-ChildItem $nvidia -Directory | ForEach-Object { Join-Path $_.FullName 'bin' } | Where-Object { Test-Path $_ }
$env:PATH = (($bins -join ';') + ';' + $env:PATH)
.\.venv\Scripts\python.exe src\state_prep_gpu\benchmark_compare.py
```

Configuration:

- `T=200`
- `B=64`
- `S=1`
- `n=32`
- `dt=1e-7`
- warmup: 1
- repeat: 3

Results:

- CPU NumPy best: `3.012 s`
- GPU CuPy best: `0.871 s`
- Speedup: `3.46x`

Interpretation:

The array-based GPU path is clearly faster for a synthetic batched slow-Hamiltonian workload with moderate batch size. This supports keeping the flat GPU API for ensemble-style workloads.

## Benchmark 2: Real SPA2-Like CPU vs GPU

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'
$nvidia = Join-Path (Get-Location) '.venv\Lib\site-packages\nvidia'
$bins = Get-ChildItem $nvidia -Directory | ForEach-Object { Join-Path $_.FullName 'bin' } | Where-Object { Test-Path $_ }
$env:PATH = (($bins -join ';') + ';' + $env:PATH)
.\.venv\Scripts\python.exe src\state_prep_gpu\benchmark_spa2_real.py
```

Configuration:

- `N_steps=6000`
- detuning batch: `25`
- Hilbert-space dimension: `n=64`
- microwave frequency: `26.668808 GHz`
- CPU path: `Simulator.run_microwave_scan(...)`
- GPU path: `simulate_batched_scalar_microwaves_shared_slow(...)`

Results:

- CPU shared-slow scan: `79.255 s`
- GPU shared-slow scalar microwave scan: `74.840 s`
- Speedup: `1.06x`
- Max `|delta probabilities_final|`: `4.228e-06`
- CPU example depletion range: `0.000..1.000`

Interpretation:

The GPU implementation is numerically close for this benchmark, but only slightly faster. For this real SPA2 shape, the GPU path does not yet deliver the synthetic benchmark speedup. Likely reasons include small-to-moderate batch size, repeated per-step eigensolves, and overhead in the current shared-slow GPU implementation. Larger batches may improve GPU utilization, but this should be measured directly.

## Benchmark 3: CPU Output Storage Modes

Inline script used a real non-microwave field/Hamiltonian setup:

- `N_steps=2500`
- `n=36`
- initial states: `4`
- microwave fields: none
- eigensolver: `zheevd`

Results:

| Mode | Time | Stored Output | Saved Steps |
| --- | ---: | ---: | ---: |
| full default | `0.341 s` | `8.93 MiB` | `2500` |
| downsample every 20 | `0.291 s` | `0.45 MiB` | `126` |
| final-only probabilities | `0.292 s` | `0.00 MiB` | `2500` |
| final monitor-only | `0.289 s` | `0.00 MiB` | `2500` |

Relative to full default:

- Downsample every 20: `1.17x` faster and about `20x` less stored output.
- Final-only: `1.17x` faster and eliminates full trajectory output arrays.
- Monitor-final-only: `1.18x` faster and avoids full probability output.

Interpretation:

For this small non-microwave case, runtime is dominated by eigensolves, so storage settings only improve wall time modestly. The memory reduction is large and will matter more for long scans, many initial states, or notebook workflows that hold multiple results at once.

## Benchmark 4: CPU Shared-Slow Microwave Scan vs Repeated Single Runs

Inline script compared `Simulator.run_microwave_scan(...)` against a naive loop over `Simulator.run(...)` with one detuning per run.

Configuration:

- SPA2-like setup using the benchmark helper functions.
- `N_steps=1000`
- detuning batch: `5`
- Hilbert-space dimension: `n=64`
- microwave fields: `2`
- both paths used final-only probability storage.

Results:

- Shared-slow scan: `3.352 s`
- Naive repeated runs: `6.353 s`
- Speedup: `1.90x`
- Max absolute probability difference: `7.314e-10`

Interpretation:

The shared-slow CPU scan is a clear win for microwave parameter scans and agrees closely with repeated single-run results. This validates the recommendation to route detuning/power scans through `run_microwave_scan(...)` instead of notebook-level loops over `run(...)`.

## Inconclusive Benchmark: GPU `out=` Buffer Reuse

I attempted a focused benchmark comparing:

- `slow_hamiltonian_batch(terms, E, B)`
- `slow_hamiltonian_batch(terms, E, B, out=preallocated)`

The benchmark timed out repeatedly:

- `B=512`, `n=64`, `repeat=2000`: timed out after 300 s.
- `B=128`, `n=64`, `repeat=200`: timed out after 180 s.
- `B=32`, `n=32`, `repeat=50`: timed out after 120 s.

No timing result should be inferred from these runs. A better microbenchmark should print progress, force one-time CuPy kernel compilation before timing, and use CUDA events rather than wall-clock timing around many Python-looped calls.

## Actionable Conclusions

1. Keep and prefer `run_microwave_scan(...)` for CPU microwave parameter scans. It measured `1.90x` faster than repeated final-only single runs on a small SPA2-like scan.
2. Use final-only or monitor-only storage for scans. The speed gain was modest in the small benchmark, but memory reduction was substantial.
3. Keep the GPU path, but do not assume it is faster for every real workload. The synthetic benchmark showed `3.46x`, while the real SPA2-like benchmark showed only `1.06x`.
4. Fix or document the CuPy/NVRTC DLL path requirement on Windows. Without prepending NVIDIA package `bin` directories to `PATH`, GPU benchmarks fail before timing.
5. Add parameterized benchmark scripts so smaller and larger scan sizes can be swept without editing source or using inline scripts.
6. Revisit GPU buffer-reuse benchmarking with CUDA events and smaller, validated kernels before implementing broad buffer reuse based on timing claims.

## Parameterized Benchmark Scripts

Reusable benchmark entry points now live in `benchmarks/`. Each script writes JSON with top-level `benchmark`, `generated_at`, `environment`, `config`, and `results` fields. Passing `--csv` also writes flattened result rows as CSV.

Example commands:

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_storage_modes.py --n-steps 500 --output results\storage.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_microwave_scan.py --n-steps 500 --batch 5 --compare-naive --output results\microwave.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_gpu_spa2.py --n-steps 1000 --batch-sweep 5 25 --cpu --gpu --output results\gpu_spa2.json --csv
.\.venv\Scripts\python.exe benchmarks\bench_gpu_kernels.py --batch 32 --n 32 --output results\gpu_kernels.json --csv
```

Smoke-test commands run successfully for storage, CPU microwave scan, and CPU-only SPA2. The GPU kernel benchmark now runs CUDA work in a child process with `--gpu-timeout`; on this machine the smoke command exited cleanly with a skipped result because the worker timed out while initializing/running CuPy kernels.
