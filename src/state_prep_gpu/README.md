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
- SPA2 microwave scan benchmark: `benchmarks/bench_gpu_torch_full_spa2.py`

Run GPU examples with a CUDA-enabled PyTorch build. The SPA2 benchmark compares
the PyTorch implementation against the CPU scan path.
