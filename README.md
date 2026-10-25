# centrex-state-prep

Simulates coherent molecular state preparation and time evolution for the
CeNTREX TlF experiment. Selected TlF quantum states are propagated along a
straight-line molecular-beam trajectory through electric, magnetic and
optionally microwave fields.

Two packages live here:

- `state_prep` — the CPU simulation engine. This is the reference implementation
  and what the example notebooks use.
- `state_prep_gpu` — an optional flat, array-based PyTorch path for large batched
  scans. See `src/state_prep_gpu/README.md`.

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.11+.

```bash
git clone https://github.com/ograsdijk/centrex-state-prep.git
cd centrex-state-prep
uv sync              # create .venv and install dependencies from uv.lock
uv pip install -e .  # editable install of the package itself
```

That installs `centrex_tlf`, which supplies the TlF Hamiltonians and state
machinery, along with the rest of the dependencies.

To run the notebooks:

```bash
uv run jupyter lab
```

The examples under `examples/` are the best place to start. `examples/SPA`
covers state preparation by adiabatic passage, `examples/SPB` and
`examples/SPB_high_field` cover the branching variants, and
`examples/batched_mu_scan_cpu.py` is a short end-to-end script.

## A minimal simulation

```python
import numpy as np
import state_prep as sp

trajectory = sp.Trajectory(
    Rini=np.array([0.0, 0.0, 0.0]),
    Vini=np.array([0.0, 0.0, 184.0]),
    zfin=0.3,
)
electric_field = sp.ElectricField(E_R=lambda R: np.array([0.0, 0.0, 1e4]))
magnetic_field = sp.MagneticField(B_R=lambda R: np.array([0.0, 0.0, 1e-4]))

hamiltonian = sp.SlowHamiltonian(
    Js=[0, 1, 2, 3],
    trajectory=trajectory,
    electric_field=electric_field,
    magnetic_field=magnetic_field,
)

simulator = sp.Simulator(
    trajectory, electric_field, magnetic_field,
    [sp.J1_triplet_0], hamiltonian, None,
)
result = simulator.run(N_steps=10_000)
```

## Parameter scans

Forward-velocity scans can share the same spatial samples and eigensolves:

```python
velocities = np.linspace(140.0, 240.0, 25)  # m/s
result = simulator.run_microwave_scan(
    detunings_hz=np.zeros((1, len(simulator.microwave_fields))),
    intensity_prefactors=np.ones((1, len(simulator.microwave_fields))),
    forward_velocities=velocities,
    N_steps=10000,
)
```

A single microwave parameter row broadcasts over the velocities. With multiple
rows, each velocity pairs with its corresponding detuning/power row; use
`np.repeat`/`np.tile` to form a Cartesian grid. Uniform sampling fixes spatial
step size, with each point's timestep scaled by reference forward speed divided
by its speed. The whole velocity vector scales to preserve the reference path's
transverse slope. This models speed variation along one path, not a change in
beam angle. `result.t_array` uses reference-trajectory time;
`result.times_per_point` has shape `(B, T)` and contains physical times.

Frozen microwave eigensolves are reused for identical detuning/power rows.
The CPU's optional multitone path still shares slow eigensolves, but evaluates
beat phases at each velocity's elapsed time and retains separate full solves.
No speed-dependent detuning or Doppler correction is added automatically.
With `workers>1`, sharing is within each worker's chunk; each chunk repeats the
shared solves, so worker count should be benchmarked for a velocity ensemble.

The CPU/GPU launcher also accepts an optional `forward_velocities` array `(B,)`
in its `--parameters` NPZ file, alongside `(B,M)` detunings and intensity
prefactors. The merged output retains the speeds, `reference_forward_velocity`,
and `reference_t_array`; multiply that time array by reference speed / speed
to recover physical times. Checkpoints include the velocity array in their
fingerprint. Distinct carriers on the same manifold, including an RC J12
background, select the GPU multitone path automatically. It preserves beat
phases and per-velocity times, using bounded time/scan tiles and dense solves.
The setup must support fixed polarization matrices with scalar spatial envelopes;
the launcher validates this factorization before evolving the states.
For multitone GPU scans, `--gpu-time-chunk` (default 8) and
`--gpu-max-eigh-matrices` (default 256) control time batching and solver workspace.
Larger settings may improve speed at higher memory cost; they are included in
checkpoint fingerprints. Measurements are in [RC_PERFORMANCE.md](RC_PERFORMANCE.md).

For CPU multitone scans using `propagator="magnus"`, oscillatory integrals are
reused across power points with identical detunings, velocities and beat
frequencies. This changes neither the integrator nor its timestep requirement.
The default `propagator="frozen"` remains unchanged. The GPU multitone path
uses frozen propagation at the supplied sample times; changing integrators or
step counts still requires an accuracy comparison for the intended scan.

For scans where the trajectory, fields and slow Hamiltonian are shared across
scan points and only the microwave parameters vary, use
`Simulator.run_microwave_scan` rather than looping over `run`. It reuses a
single slow-Hamiltonian diagonalisation across the whole batch and is
substantially faster.

```python
detunings, prefactors = sp.scan_grid(
    n_fields=2,
    detunings_hz=np.linspace(-2e6, 1e6, 25),
)
result = simulator.run_microwave_scan(
    detunings_hz=detunings,
    intensity_prefactors=prefactors,
    N_steps=10_000,
    monitor_states=[...],
)
```

`workers > 1` spreads the batch across processes and is worth using for
production-sized scans; see the `run_microwave_scan` docstring for when it helps
and when it does not.

`Simulator.run()` and `run_microwave_scan()` automatically detect exact
Hamiltonian blocks and diagonalize them independently when the detected split
is substantial. Detection includes the active field and microwave couplings
across the sampled trajectory, so transverse terms or differently oriented
microwaves can merge blocks. Use `block_eigh=False` to force dense eigensolves.

For a cascade using distinct microwave carriers, list fields in low-to-high
rotational order (for example J = 0 -> 1, then J = 1 -> 2). The rotating-frame
shift for an upper rung includes the carrier and detuning of every rung below it.
Fields sharing a carrier define one rotating-frame rung; use
`allow_multitone_same_manifold=True` only for distinct carriers addressing the
same excited-J manifold.

## Benchmarks

`benchmarks/` holds parameterised benchmark scripts that emit JSON and CSV.
`benchmarks/paired.py` runs before/after comparisons interleaved with repeats
and reports whether a difference is statistically resolvable — use it rather
than single timing pairs, since the effects here are often the same size as
run-to-run noise.

## Further reading

- `AGENTS.md` — architecture, units and numerical conventions, and the
  self-check list before changing anything.
- `IMPROVEMENTS.md` — measured performance findings and the open backlog.
- `PERFORMANCE_BENCHMARK_RESULTS.md` — recorded benchmark numbers with the
  environment they were taken in.
- `PERFORMANCE_QUICKSTART.md` — local performance updates and resumable CPU/GPU
  concurrent scans, with parameter and setup-file instructions.
