"""GPU-accelerated (PyTorch) batched simulation utilities.

This package is intentionally separate from `state_prep` and keeps the API
flat/functional to avoid the deep OOP call chain in the original code.

PyTorch is an optional dependency. Importing `state_prep_gpu` does not require
PyTorch; functions that execute the simulation will raise a helpful error if
PyTorch is missing.
"""

from .slow_hamiltonian import (
    BatchedSlowHamiltonianTerms,
    slow_hamiltonian_batch,
    terms_from_centrex_tlf,
)
from .simulator import simulate_batched
from .multitone import simulate_batched_multitone_shared_slow
from .microwave_factors import (
    compute_eb_t,
    microwave_amp_t_np,
    microwave_base_matrices_np,
    microwave_factors_np,
    microwave_upper_factors_np,
)

__all__ = [
    "BatchedSlowHamiltonianTerms",
    "slow_hamiltonian_batch",
    "terms_from_centrex_tlf",
    "simulate_batched",
    "simulate_batched_multitone_shared_slow",
    "compute_eb_t",
    "microwave_amp_t_np",
    "microwave_base_matrices_np",
    "microwave_factors_np",
    "microwave_upper_factors_np",
]
