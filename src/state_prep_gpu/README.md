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

The shared-slow scan also batches independent slow eigensolves across time in
bounded tiles. `slow_eigh_chunk_size=None` (default) uses 128 timesteps on CUDA
and one timestep on CPU tensors; pass a positive integer to choose a tile size,
or `1` for the original per-step solver route. This works with dense transverse
fields as well as exact blocks. State propagation, label tracking, monitor
mapping, and history storage remain sequential and use the true initial fields.
Only propagated samples are included, and the last tile may be shorter.

At 64 states in complex128, 128 slow eigensystems occupy about 8.45 MB plus
input matrices, labels, and solver workspace. Tile storage scales with the tile
size and state dimension, rather than trajectory length. Each call builds its
own tiles. With profiling enabled, `slow_eigh` includes tile construction and
solve time amortized per covered timestep; other stages use the sampling stride.

See `PERFORMANCE_QUICKSTART.md` for concurrent CPU/GPU scans and checkpoints.

### Forward-velocity ensembles

`simulate_batched_scalar_microwaves_shared_slow` accepts `forward_velocities`
(positive scalar or `(B,)`, in m/s) and `reference_forward_velocity` (m/s).
Supply `E_t`, `B_t`, and microwave envelopes on one common spatial path;
`dt` describes traversal at the reference speed. The function scales each
point's timesteps by `reference_forward_velocity / forward_velocities`.
Its output batch order and storage shapes are unchanged.

For example, with the usual arrays prepared for a reference speed of 184 m/s,
pass `forward_velocities=[150., 184., 230.]` and
`reference_forward_velocity=184.` along with three rows of initial states,
coupling scales and rotating-frame shifts. Identical microwave parameters
share the full eigensolve across these speeds. Rows with different detunings,
powers, or complete per-point envelopes retain distinct solves. Grouping
compares exact inputs once per call; it does not cache results across calls.

Keep trajectory direction fixed when sampling the common path: changing only
the forward component while retaining transverse speed would change the path.
For same-manifold carrier beats, use the multitone entry point described below.
The scan launcher selects it automatically for distinct carriers. The CPU
multitone route also supports per-velocity beat phases and full eigensolves.
No Doppler correction is inferred from the supplied speeds.

### RC backgrounds and multitone beats

`state_prep_gpu.simulate_batched_multitone_shared_slow` supports an RC J12
background at a different carrier from SPA J12. Supply the original fields'
upper coupling matrices as `H_mu_upper_fields` `(M,n,n)`, scalar envelopes
`amp_t` `(T-1,M)` or `(T-1,B,M)`, amplitude coupling scales `(B,M)`, and
`beat_angular_frequencies` `(B,M)` in **rad/s** relative to the chosen frame
reference. Zero frequency marks a static field. The lower component is the
upper component's adjoint. `D_mu_diag_batch` must contain the reference-frame
shifts, without adding a second frame shift for the RC carrier.

`sample_times` are **seconds**, at the Hamiltonian evaluation points, with
shape `(T-1,)` or `(T-1,B)`. The beat phase is
`exp(-1j * beat_angular_frequencies * physical_sample_time)` on the upper
component and its conjugate on the lower. Optional forward velocities scale
both sample times and timesteps by reference speed / speed. Velocities with
different beat phases retain separate full eigensolves.

The default slow tile contains up to 128 samples and the full microwave tile
up to eight. `max_eigh_matrices=256` caps the number of matrices in a solver
call, splitting scan points when necessary and capping slow/time tiles too.
Stored full eigensystems scale as time-tile size × batch × n²; the whole
trajectory is never cached. These are dense solves, valid for transverse
fields and mixed polarization, without an aligned-field assumption.
Set `rotating_eigh_chunk_size=1` to use per-step full solves.

State propagation, label tracking and history remain sequential. True initial
fields (`E0`, `B0`), monitor mapping and final-only storage follow the other
shared-slow entry point. Profiling accepts `profile_timings` on CUDA; tile
intervals are amortized over their actual widths, including partial tails.
Complex128 is the default prepared by the launcher and is recommended for
long scientific runs; complex64 can accumulate substantially more norm error.

The existing CPU/GPU launcher accepts a multitone setup file and automatically
constructs the reference shifts, fixed RC detuning and beat frequencies from
its parameter rows. Spatial envelope/polarization factorization is checked at
every propagated sample before using this scalar-envelope route. There is no
GPU Magnus implementation in this entry point.

For launcher jobs, the optional `--gpu-time-chunk 16 --gpu-max-eigh-matrices 512`
settings reduced runtime by another 3.6% in a 1,000-sample, 25-point RC check,
but approximately doubled measured peak allocated memory. Defaults stay at
8/256. These flags apply to the multitone route and are fingerprinted for
checkpoint reuse; they do not change the scalar no-beat route. See
`RC_PERFORMANCE.md` for repeated timings and memory conditions.

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
