"""CPU/GPU agreement. Skipped entirely when CuPy is unavailable.

The GPU path is a separate implementation, not a wrapper, so it can drift from
the CPU engine silently. These tests only run where CuPy is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy", reason="CuPy not installed; GPU path untestable")

from conftest import CROSS_BACKEND_TOL  # noqa: E402


def test_slow_hamiltonian_matches_cpu(spa2_setup):
    """Batched GPU slow-Hamiltonian construction must match the CPU one."""
    from state_prep_gpu import slow_hamiltonian_batch, terms_from_centrex_tlf

    hamiltonian = spa2_setup["hamiltonian"]
    terms = terms_from_centrex_tlf(hamiltonian.QN)

    rng = np.random.default_rng(0)
    E = rng.normal(size=(3, 2, 3)) * 1e3
    B = rng.normal(size=(3, 2, 3)) * 1e-4

    gpu = cupy.asnumpy(slow_hamiltonian_batch(terms, E, B))

    for t in range(E.shape[0]):
        for b in range(E.shape[1]):
            cpu = hamiltonian.H_EB(E[t, b], B[t, b])
            assert np.max(np.abs(gpu[t, b] - cpu)) < 1e-6


def test_gpu_probabilities_normalise(spa2_setup):
    """A GPU run must conserve probability, whatever else it does."""
    from state_prep_gpu import simulate_batched  # noqa: F401

    pytest.skip(
        "Full GPU evolution needs precomputed field and microwave time series; "
        "see benchmarks/bench_gpu_spa2.py for the end-to-end comparison."
    )
