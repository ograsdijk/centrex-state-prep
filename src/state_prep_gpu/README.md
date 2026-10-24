# `state_prep_gpu`

PyTorch implementation of the batched state-preparation simulator. It is a
separate, array-based GPU path; the object-oriented CPU implementation remains
in `state_prep` and does not depend on PyTorch.

## What you provide

- Precomputed slow-Hamiltonian terms (`Hff`, `HSx`, `HSy`, `HSz`, `HZx`, `HZy`,
  `HZz`) from `centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(...)`.
- Electric and magnetic field arrays `E_t`, `B_t`, typically shaped
  `(T, B, 3)`.
- Initial state vectors shaped `(B, S, n)` and, when using the general
  microwave path, microwave Hamiltonians shaped `(T-1, B, n, n)`.

`terms_from_centrex_tlf(...)` converts Hamiltonian terms to PyTorch tensors. It
places them on CUDA when available and otherwise uses the CPU. You can choose a
device explicitly with its `device` argument. Simulation inputs are moved to
the same device as the terms, and outputs are PyTorch tensors on that device.

## API

- `state_prep_gpu.slow_hamiltonian_batch(terms, E, B)`
- `state_prep_gpu.simulate_batched(...)`
- `state_prep_gpu.simulator.simulate_batched_compact_microwaves(...)`
- `state_prep_gpu.simulator.simulate_batched_scalar_microwaves(...)`
- `state_prep_gpu.simulator.simulate_batched_scalar_microwaves_shared_slow(...)`

The scan dimension is `B`; initial states are propagated together on the `S`
axis. The general and scalar-microwave simulators accept per-point time steps
shaped `(T-1, B)`. Scalar-microwave simulators also accept per-point envelopes
shaped `(T-1, B, M)`.

For midpoint propagation, pass midpoint-sampled `E_t`, `B_t`, and microwave
envelopes, and provide separate `E0`/`B0` values at the true initial position
for initial-state and monitor mapping. Simulators return final states and
probabilities by default. Pass `store_every=1` to retain every state, or a
larger positive stride for a downsampled history.

Eigenvector tracking uses optimal one-to-one maximum-absolute-overlap assignment
and phase alignment, matching the CPU objective. CUDA assignment runs on-device
on PyTorch's current stream; it does not transfer an overlap matrix to SciPy.
Its small kernel is compiled once per process/device/size/precision using the
NVRTC library included in CUDA-enabled PyTorch. No standalone CUDA compiler is
required. The CUDA assignment currently supports up to 512 states and finite
float32/float64 costs. CPU tensor execution uses SciPy. Optimal ties in degenerate
subspaces can have different valid permutations across backends.

The shared-slow scan accepts `block_eigh="auto"` (default), `True`, or `False`.
Automatic mode attempts exact blocks for batches of at least 250 points and uses
dense solves below that size. This conservative threshold reflects the measured
64-state SPA2 workload; it is not an autotuner for other dimensions/devices.
`True` forces block detection; `False` forces dense solves. Detection unions all
potentially active slow-Hamiltonian terms over the supplied field arrays,
initial fields, and microwave matrices. Tiny nonzero couplings are retained;
connected support falls back to dense solves. Sorted slow-basis sector indices
are recomputed on-device at every step, rather than cached across crossings.

See `PERFORMANCE_QUICKSTART.md` for concurrent CPU/GPU scans and checkpoints.

## Install PyTorch

PyTorch is optional so CPU-only users can install and import the package without
it. Install a PyTorch build that matches the machine and CUDA driver, then
install the project with its GPU extra:

```powershell
uv pip install torch
uv pip install -e ".[gpu]"
```

If your platform needs a specific PyTorch package index or CUDA build, follow
the selection instructions at pytorch.org and install that build in place of
the first command. The `gpu` extra records the runtime dependency for project
metadata.

## Examples and benchmarks

- Synthetic propagation smoke check: `src/state_prep_gpu/example_smoke.py`
- CPU comparison: `src/state_prep_gpu/example_compare_cpu.py`
- CPU/GPU benchmark: `src/state_prep_gpu/benchmark_compare.py`
- Production SPA2 scan benchmark: `benchmarks/bench_gpu_spa2.py`
- Paired GPU dense/block checks: `benchmarks/bench_performance_updates.py --device gpu`

Run GPU examples with a CUDA-enabled PyTorch build. The SPA2 benchmark compares
the PyTorch implementation against the CPU scan path.
