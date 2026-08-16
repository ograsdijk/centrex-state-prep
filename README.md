# centrex-state-prep

Simulates coherent molecular state preparation and time evolution for the
CeNTREX TlF experiment. Selected TlF quantum states are propagated along a
straight-line molecular-beam trajectory through electric, magnetic and
optionally microwave fields.

Two packages live here:

- `state_prep` — the CPU simulation engine. This is the reference implementation
  and what the example notebooks use.
- `state_prep_gpu` — an optional flat, array-based CuPy path for large batched
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
