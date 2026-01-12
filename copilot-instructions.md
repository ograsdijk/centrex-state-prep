# Copilot instructions (centrex-state-prep)

Scope: these instructions are derived from and meant for code under `src/state_prep/`.

## Project shape (src)
- Core modules live directly in `src/state_prep/`:
  - `trajectory.py`: straight-line `Trajectory` with `R_t(t)` and `get_T()`.
  - `electric_fields.py`, `magnetic_fields.py`: `ElectricField` / `MagneticField` wrappers around callables `E_R(R)` / `B_R(R)`, plus `get_*_t_func(R_t)`.
  - `hamiltonians.py`: Hamiltonian wrappers expose `get_H_t_func()`.
  - `microwaves.py`: microwave field + polarization + coupling-matrix generation.
  - `simulator.py`: `Simulator` runs time evolution; `SimulationResult` stores arrays and provides plotting/helpers.
  - `utils.py`: linear-algebra helpers (`reorder_evecs`, `find_max_overlap_idx`, state<->vector conversion).

## Code style & conventions
- Prefer small, composable callables: the package frequently stores functions on instances (e.g., `E_R`, `B_R`, `H_R`, `*_t`).
- Use `numpy` vectorization where practical; the codebase already uses `np.einsum` for overlaps.
- Use `dataclasses` for lightweight data containers; keep `__post_init__` for derived fields.
- Type hints are used but not aggressively enforced; follow existing patterns (`np.ndarray`, `numpy.typing.NDArray` when it helps).
- Keep imports consistent with existing files: standard lib, then third-party, then local `from .module import ...`.

## Numerical/physics conventions (as implemented in src)
- Time arrays are in seconds (`t_array`), but plots commonly display microseconds (`t_array / 1e-6`).
- Positions are in meters; plots sometimes show centimeters (`z_array / 1e-2`).
- Energies in `SimulationResult.energies` are treated as angular frequencies (e.g., displayed as `energies / (2*np.pi)` in Hz).
- Eigenvector ordering matters: when diagonalizing along a sweep, use overlap-based tracking (`utils.reorder_evecs`) to maintain consistent state labels.
- Be careful about basis/coordinate conversions:
  - Fields expose XYZ and an internal conversion matrix `R_to_r`; follow the existing `get_*_R` and `get_*_r` pattern.

## Patterns to follow when adding/changing code
- New fields:
  - Implement as a callable `*_R(R: np.ndarray) -> np.ndarray` and wrap in `ElectricField`/`MagneticField`.
  - If the field is used along a trajectory, provide or use `get_*_t_func(R_t)`.
- New Hamiltonians:
  - Provide `get_H_t_func()` returning `Callable[[float], np.ndarray]` and store `QN` on the object.
- New simulation outputs:
  - Store arrays on `SimulationResult` and add small helper methods (plot/getters) rather than adding side effects in `Simulator`.
- Preserve API names and units; avoid renaming public attributes used across modules (`QN`, `H_list`, `D`, `t_array`, etc.).

## Safety rails (what NOT to do)
- Don’t “fix” physics or units unless explicitly requested; many constants and scalings are domain-specific.
- Don’t introduce new dependencies unless necessary; prefer existing stack (`numpy`, `scipy`, `matplotlib`, `tqdm`, `centrex_tlf`).
- Don’t reformat whole files; keep diffs minimal and consistent with current style.

## Quick self-check before proposing changes
- If you touch diagonalization/time evolution: verify eigenvector ordering and shapes stay consistent.
- If you touch coordinate systems: confirm XYZ vs `r` basis usage (`R_to_r`) and expected units.
- If you touch serialization: `SimulationResult.save_to_pickle` uses `dill`; keep that compatible.
