"""The shared-slow scan must agree with repeated single runs.

This is the core regression test: `run_microwave_scan` reuses one slow-Hamiltonian
diagonalisation across the whole batch, which is a substantial optimisation over
looping `run()`. If that reuse were wrong, the two would diverge.

Both paths use the same eigensolver here, so the tolerance is tight. Anything
that loosens this materially is a real change in the physics, not noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import CROSS_BACKEND_TOL, SAME_BACKEND_TOL

N_STEPS = 400


def _scan(simulator, setup, detunings, **kwargs):
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    return simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=N_STEPS,
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        **kwargs,
    )


def _loop(simulator, setup, detunings):
    from state_prep import SCAN_STORAGE_DEFAULTS

    base = setup["frequency"]
    out = []
    try:
        for det in detunings:
            for mw in setup["microwave_fields"]:
                mw.set_frequency(base + det)
            result = simulator.run(
                N_steps=N_STEPS,
                monitor_states=setup["monitor_states"],
                progress=False,
                **SCAN_STORAGE_DEFAULTS,
            )
            out.append(result.probabilities_final)
    finally:
        for mw in setup["microwave_fields"]:
            mw.set_frequency(base)
    return np.stack(out)


def test_shared_slow_matches_repeated_runs(simulator, spa2_setup, detunings):
    scan = _scan(simulator, spa2_setup, detunings)
    loop = _loop(simulator, spa2_setup, detunings)

    assert scan.probabilities_final.shape == loop.shape
    assert np.max(np.abs(scan.probabilities_final - loop)) < SAME_BACKEND_TOL


def test_probabilities_normalise(simulator, spa2_setup, detunings):
    scan = _scan(simulator, spa2_setup, detunings)
    totals = scan.probabilities_final.sum(axis=-1)
    assert np.allclose(totals, 1.0, atol=1e-10)


def test_blas_pinning_does_not_change_results(simulator, spa2_setup, detunings):
    """Pinning BLAS threads is a scheduling change and must be bit-for-bit inert."""
    pinned = _scan(simulator, spa2_setup, detunings, blas_threads=1)
    unpinned = _scan(simulator, spa2_setup, detunings, blas_threads=None)
    assert np.array_equal(pinned.probabilities_final, unpinned.probabilities_final)


def test_loky_workers_match_serial(simulator, spa2_setup, detunings):
    """Splitting the batch across processes must not change results."""
    serial = _scan(simulator, spa2_setup, detunings, workers=1)
    parallel = _scan(simulator, spa2_setup, detunings, workers=2)
    assert np.array_equal(
        serial.probabilities_final, parallel.probabilities_final
    )


def test_eig_backends_agree_within_cross_backend_tolerance(
    simulator, spa2_setup, detunings
):
    """zheevd and numpy eigh agree, but only to ~1e-6, not to machine precision.

    Near-degenerate eigenvectors are not unique, so the two LAPACK paths pick
    different bases within a degenerate subspace and adiabatic tracking amplifies
    the difference. This test pins the expectation so a future tolerance choice
    is made against a measured number.
    """
    a = _scan(simulator, spa2_setup, detunings, eig_backend="zheevd")
    b = _scan(simulator, spa2_setup, detunings, eig_backend="numpy")
    diff = np.max(np.abs(a.probabilities_final - b.probabilities_final))
    assert diff < CROSS_BACKEND_TOL
