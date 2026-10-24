# CPU/GPU performance updates

These are local changes to the existing simulation loops, without an API or
architecture refactor. Numerical units, frozen/Magnus defaults, and CPU public
interfaces are unchanged.

## Changes

- CPU block-index groups are constructed once per timestep, not per scan point.
- Multitone Magnus skips unused frozen-Hamiltonian assembly and midpoint phases.
- GPU tracking now maximizes total absolute overlap one-to-one and aligns phases.
  The assignment kernel uses NVRTC from PyTorch and stays on the current CUDA
  stream, including its full conflict-resolution path. SciPy is the CPU reference.
- Production GPU shared-slow propagation supports exact blocks, with dense fallback.
  `block_eigh="auto"` uses blocks at batches of 250 or more. Force `True`/`False`
  to measure other workloads: small batches can be slower with blocks.
- The serial-fraction profiler accepts and instruments the current block path.
- GPU benchmark prototypes use production tracking instead of a copied greedy rule.

## Run CPU and GPU together

From the repository root, select the project environment. To reuse an existing
environment, set `$python` to its Python executable instead:

```powershell
$python = '.\.venv\Scripts\python.exe'
$env:PYTHONPATH = "$PWD\src"
& $python scripts/run_cpu_gpu_scan.py --steps 10000 --batch 1000 --workers 6 --cpu-fraction 0.6 --output-dir results/scan_01
```

The example uses the existing benchmark SPA2 setup, four initial states, a
detuning scan from -2 MHz to +1 MHz, and unit power prefactors. Choose an output
directory for each request. The CPU fraction is a starting choice, not a promise
of optimal throughput. `0` selects GPU only; `1` selects CPU only. CPU workers
are separate processes with one BLAS thread each; GPU execution reserves a
separate process and limits its PyTorch CPU threads to one.

For two independent simulations, launch separate commands with different output
directories and their own setup/parameter files. Use `--cpu-fraction 1` for the
CPU job and `--cpu-fraction 0` for the GPU job. Their results remain independent.
The measured split-scan speedup does not establish the throughput of two
independent jobs.

For your own setup, pass `--setup-file setup.dill`: a dill-serialized mapping
with the same keys as `benchmarks.common.build_spa2_setup()`. Supply arbitrary
detuning/power points with `--parameters points.npz`; it must contain matching
`detunings_hz` and `intensity_prefactors` arrays of shape `(B, M)`. The launcher
supports uniform midpoint frozen propagation. It rejects same-manifold beating
and checks the GPU scalar microwave factorization at every propagated sample,
so varying polarization cannot silently become a static-polarization model.

The output directory contains:

- `setup.dill` for the default setup, plus `parameters.npz` and `request.json`.
- `cpu.npz` and/or `gpu.npz`, written atomically after each backend's chunk.
- `merged.npz`: ordered physical final states, common readout basis, final
  probabilities and monitor probabilities, parameters and initial-state vectors.
- `timing.json`: preparation/propagation details are in the individual checkpoint
  files; wall time in this report is for the current launcher invocation.

Repeat the identical command to reuse completed chunks. Request/source/setup
fingerprints prevent stale results from being reused after an input or code
change. Each backend's assigned chunk is the restart unit; there is no timestep
checkpoint. Final probabilities are recomputed in one common tracked basis
(CPU when available), rather than concatenating incompatible backend labels.
Merging refuses different physical initial states.

## Validation and measurements

Use the paired runners, with one warmup and at least five interleaved rounds:

```powershell
& $python benchmarks/bench_performance_updates.py --device cpu --steps 500 1000 --batches 5 25 --output results/cpu_performance_updates.json
& $python benchmarks/bench_performance_updates.py --device gpu --steps 500 1000 --batches 25 250 --output results/gpu_performance_updates.json
& $python benchmarks/bench_gpu_assignment.py --steps 500 --batch 25 --output results/gpu_assignment.json
& $python benchmarks/bench_concurrent_scan.py --steps 1000 --batch 1000 --variants cpu:8 cpu:6 hybrid:6:0.6 --output-dir results/concurrent_comparison_new
```

The CPU baseline is revision `315af72`, imported in the same interpreter as the
candidate. GPU dense/block comparisons both use optimal tracking. GPU runner
preparation is excluded from their timings, but block discovery inside each
propagation is included. Concurrent comparisons launch fresh processes and
include startup, preparation, checkpoint writing, and merging. They refuse to
time cached output directories.

See the 2026-10-24 entry in `PERFORMANCE_BENCHMARK_RESULTS.md` for results and
limitations. Generated raw reports remain under ignored `results/`.
