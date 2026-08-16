"""Helpers for constructing microwave parameter scans.

`Simulator.run_microwave_scan` takes `detunings_hz` and `intensity_prefactors`
shaped `(B, M)` for batch size `B` and `M` microwave fields. Building those by
hand is easy to get wrong in three ways:

- the two arrays must have identical shapes,
- `intensity_prefactors` scales *intensity*, so couplings scale as its square
  root: a prefactor of 4 doubles the Rabi rate,
- fields that share a carrier frequency must share a detuning, while a field
  representing a physically fixed source (an RC background, say) must keep a
  detuning of zero while the scanned fields move.

`scan_grid` builds both arrays, taking the outer product when both a detuning
axis and a prefactor axis are given.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


#: Storage settings appropriate for scans built on repeated `Simulator.run`
#: calls: keep only the final populations, which is what a scan point is for.
#: Full trajectory storage is what makes long scans expensive in memory.
#: Use as ``simulator.run(N_steps=..., **SCAN_STORAGE_DEFAULTS)``.
#:
#: `run_microwave_scan` already stores final-only and does not need these.
SCAN_STORAGE_DEFAULTS = {
    "store_psis": False,
    "store_energies": False,
    "store_probabilities": False,
    "store_final_probabilities": True,
    "store_monitor_probabilities": False,
    "store_final_monitor_probabilities": True,
}


def scan_grid(
    *,
    n_fields: int,
    detunings_hz: np.ndarray | float = 0.0,
    intensity_prefactors: np.ndarray | float = 1.0,
    detuning_fields: Optional[Sequence[int]] = None,
    prefactor_fields: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(B, n_fields)` detuning and prefactor arrays for a scan.

    Parameters
    ----------
    n_fields:
        Number of microwave fields on the `Simulator`, i.e. `M`.
    detunings_hz:
        Scalar or 1D array of detunings in Hz.
    intensity_prefactors:
        Scalar or 1D array of dimensionless *intensity* scalings. Couplings
        scale as the square root, so 4.0 doubles the Rabi rate.
    detuning_fields:
        Indices of fields that follow the detuning axis. Fields not listed get a
        detuning of zero, which is what a physically fixed source needs: scanning
        one microwave's frequency does not move an independent oscillator.
        Defaults to all fields.
    prefactor_fields:
        Indices of fields whose intensity is scaled. Fields not listed stay at
        1.0. Defaults to all fields.

    Returns
    -------
    (detunings, prefactors), both shaped `(B, n_fields)`, where `B` is the
    product of the detuning and prefactor axis lengths. The detuning axis varies
    slowest, so results reshape as `(n_detunings, n_prefactors, ...)`.

    Examples
    --------
    A 1D detuning sweep over two fields sharing a carrier::

        det, pref = scan_grid(n_fields=2, detunings_hz=np.linspace(-2e6, 1e6, 25))

    A 2D detuning by power sweep::

        det, pref = scan_grid(
            n_fields=2,
            detunings_hz=np.linspace(-2e6, 1e6, 25),
            intensity_prefactors=[0.5, 1.0, 2.0],
        )

    Multitone, where field 2 is a fixed background source that must not follow
    the scan::

        det, pref = scan_grid(
            n_fields=3,
            detunings_hz=np.linspace(-2e6, 1e6, 25),
            detuning_fields=[0, 1],
        )
    """
    if n_fields < 1:
        raise ValueError("n_fields must be >= 1")

    det_axis = np.atleast_1d(np.asarray(detunings_hz, dtype=float))
    pref_axis = np.atleast_1d(np.asarray(intensity_prefactors, dtype=float))
    if det_axis.ndim != 1 or pref_axis.ndim != 1:
        raise ValueError("detunings_hz and intensity_prefactors must be scalar or 1D")
    if np.any(pref_axis < 0):
        raise ValueError("intensity_prefactors must be >= 0")

    det_idx = _resolve_fields(detuning_fields, n_fields, "detuning_fields")
    pref_idx = _resolve_fields(prefactor_fields, n_fields, "prefactor_fields")

    # Outer product, detuning axis slowest.
    det_flat = np.repeat(det_axis, pref_axis.size)
    pref_flat = np.tile(pref_axis, det_axis.size)
    batch = det_flat.size

    detunings = np.zeros((batch, n_fields), dtype=float)
    detunings[:, det_idx] = det_flat[:, None]

    prefactors = np.ones((batch, n_fields), dtype=float)
    prefactors[:, pref_idx] = pref_flat[:, None]

    return detunings, prefactors


def _resolve_fields(
    fields: Optional[Sequence[int]], n_fields: int, name: str
) -> np.ndarray:
    if fields is None:
        return np.arange(n_fields)
    idx = np.asarray(list(fields), dtype=int)
    if idx.size and (idx.min() < 0 or idx.max() >= n_fields):
        raise ValueError(f"{name} contains an index outside [0, {n_fields - 1}]")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{name} contains duplicate indices")
    return idx
