"""Shared fixtures for the test suite.

Building the SPA2 setup is the expensive part (it constructs the TlF
Hamiltonian via centrex_tlf), so it is session-scoped and shared.

Tolerances used across the suite, all measured rather than guessed:

- ``EXACT_TOL``: for changes that reorder identical arithmetic. Float noise only.
- ``SAME_BACKEND_TOL``: two code paths, same eigensolver. Agreement is tight
  because the eigenvector choices coincide.
- ``CROSS_BACKEND_TOL``: paths that differ in eigensolver, device, or ordering.
  Near-degenerate eigenvectors are not unique, so different LAPACK routines pick
  different bases within a degenerate subspace and adiabatic tracking amplifies
  that over the timestep loop. Pin ``eig_backend`` explicitly if a tight
  tolerance is wanted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EXACT_TOL = 1e-12
SAME_BACKEND_TOL = 1e-8
CROSS_BACKEND_TOL = 1e-5


@pytest.fixture(scope="session")
def spa2_setup():
    from common import build_spa2_setup

    return build_spa2_setup()


@pytest.fixture(scope="session")
def simulator(spa2_setup):
    from state_prep import Simulator

    return Simulator(
        spa2_setup["trajectory"],
        spa2_setup["electric_field"],
        spa2_setup["magnetic_field"],
        spa2_setup["initial_states"],
        spa2_setup["hamiltonian"],
        spa2_setup["microwave_fields"],
    )


@pytest.fixture(scope="session")
def multitone_fields(spa2_setup):
    """SPA2 + SPA background + RC background, all on the same excited-J manifold."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import analyze_spa2_bg_feature as bg

    rc_bg, _ = bg.make_rc_bg_field(spa2_setup, bg_fraction=1 / 35, rc_offset_mhz=0.0)
    return [
        spa2_setup["microwave_fields"][0],
        spa2_setup["microwave_fields"][1],
        rc_bg,
    ]


@pytest.fixture
def detunings():
    return np.linspace(-1.0e6, 0.5e6, 4)
