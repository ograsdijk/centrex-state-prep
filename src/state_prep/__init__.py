"""Coherent state preparation and time evolution for the CeNTREX TlF experiment.

Propagates selected TlF quantum states along a straight-line molecular-beam
trajectory through electric, magnetic and optionally microwave fields.

Typical use::

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
        [sp.J1_triplet_0], hamiltonian, microwave_fields=None,
    )
    result = simulator.run(N_steps=10_000)

For parameter scans where the trajectory and fields are shared across scan
points, use ``Simulator.run_microwave_scan`` rather than looping over ``run``:
it reuses one slow-Hamiltonian diagonalisation across the whole batch.
``sp.scan_grid`` builds the parameter arrays and ``sp.SCAN_STORAGE_DEFAULTS``
gives the low-memory storage settings appropriate to scans.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

from .approximate_states import (
    J0_singlet,
    J0_triplet_0,
    J0_triplet_m,
    J0_triplet_p,
    J1_singlet,
    J1_triplet_0,
    J1_triplet_m,
    J1_triplet_p,
    J2_singlet,
    J2_triplet_0,
    J2_triplet_m,
    J2_triplet_p,
)
from .core import StaticField
from .electric_fields import (
    ElectricField,
    E_field_lens,
    E_field_ring,
    E_SPB_from_pickle,
    Ez_from_csv,
    Ez_from_csv_offset,
    linear_E_field,
)
from .hamiltonians import Hamiltonian, SlowHamiltonian
from .intensity_profiles import (
    BackgroundField,
    BesselGaussianBeam,
    GaussianBeam,
    Intensity,
    MeasuredBeam,
)
from .magnetic_fields import MagneticField
from .microwaves import MicrowaveField, Polarization
from .plotters import CouplingPlotter
from .scans import SCAN_STORAGE_DEFAULTS, scan_grid
from .simulator import (
    MicrowaveScanResult,
    SimulationResult,
    Simulator,
    limit_blas_threads,
)
from .trajectory import Trajectory
from .utils import (
    LabelGapTracker,
    calculate_transition_frequency,
    eigenstate_quantum_numbers,
    find_max_overlap_idx,
    matrix_to_states,
    reorder_evecs,
    select_eigenstate,
    vector_to_state,
)

try:
    __version__ = _version("centrex-state-prep")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # trajectory and fields
    "Trajectory",
    "StaticField",
    "ElectricField",
    "MagneticField",
    "linear_E_field",
    "E_field_lens",
    "E_field_ring",
    "Ez_from_csv",
    "Ez_from_csv_offset",
    "E_SPB_from_pickle",
    # hamiltonians
    "Hamiltonian",
    "SlowHamiltonian",
    # microwaves
    "MicrowaveField",
    "Polarization",
    "Intensity",
    "GaussianBeam",
    "BesselGaussianBeam",
    "MeasuredBeam",
    "BackgroundField",
    # simulation
    "Simulator",
    "SimulationResult",
    "MicrowaveScanResult",
    "limit_blas_threads",
    "scan_grid",
    "SCAN_STORAGE_DEFAULTS",
    # plotting and helpers
    "CouplingPlotter",
    "reorder_evecs",
    "find_max_overlap_idx",
    "vector_to_state",
    "matrix_to_states",
    "calculate_transition_frequency",
    # identifying eigenstates by quantum numbers rather than by tracked index
    "eigenstate_quantum_numbers",
    "select_eigenstate",
    "LabelGapTracker",
    # approximate states
    "J0_singlet",
    "J0_triplet_0",
    "J0_triplet_p",
    "J0_triplet_m",
    "J1_singlet",
    "J1_triplet_0",
    "J1_triplet_p",
    "J1_triplet_m",
    "J2_singlet",
    "J2_triplet_0",
    "J2_triplet_p",
    "J2_triplet_m",
]
