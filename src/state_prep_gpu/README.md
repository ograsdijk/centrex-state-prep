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

Install an appropriate build for your CUDA version, e.g.

- `pip install cupy-cuda12x`

(Exact package depends on your system.)
