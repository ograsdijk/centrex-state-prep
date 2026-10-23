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

__all__ = [
    "BatchedSlowHamiltonianTerms",
    "slow_hamiltonian_batch",
    "terms_from_centrex_tlf",
    "simulate_batched",
]
