"""GPU-accelerated (CuPy) batched simulation utilities.

This package is intentionally separate from `state_prep` and keeps the API
flat/functional to avoid the deep OOP call chain in the original code.

CuPy is an optional dependency. Importing `state_prep_gpu` does not require
CuPy; functions that execute on GPU will raise a helpful error if CuPy is
missing.
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
