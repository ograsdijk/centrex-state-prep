from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ._torch import torch as get_torch


@dataclass(frozen=True)
class BatchedSlowHamiltonianTerms:
    """Field-independent matrix terms for the slow Hamiltonian.

    Mirrors centrex_tlf's `HamiltonianUncoupledX` structure:

        H(E,B) = 2π (Hff + Ex HSx + Ey HSy + Ez HSz + Bx HZx + By HZy + Bz HZz)

    All matrices are PyTorch tensors with shape (n, n).
    """

    Hff: Any
    HSx: Any
    HSy: Any
    HSz: Any
    HZx: Any
    HZy: Any
    HZz: Any

    @property
    def n(self) -> int:
        return int(self.Hff.shape[0])


def terms_from_centrex_tlf(
    H_uncoupled_x: Any,
    *,
    dtype: Any = None,
    device: Any = None,
) -> BatchedSlowHamiltonianTerms:
    """Convert a centrex_tlf `HamiltonianUncoupledX` dataclass to tensors.

    By default, terms are placed on CUDA when available, otherwise on CPU.
    """

    torch = get_torch()
    if dtype is None:
        dtype = torch.complex128
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return BatchedSlowHamiltonianTerms(
        Hff=torch.as_tensor(H_uncoupled_x.Hff, dtype=dtype, device=device),
        HSx=torch.as_tensor(H_uncoupled_x.HSx, dtype=dtype, device=device),
        HSy=torch.as_tensor(H_uncoupled_x.HSy, dtype=dtype, device=device),
        HSz=torch.as_tensor(H_uncoupled_x.HSz, dtype=dtype, device=device),
        HZx=torch.as_tensor(H_uncoupled_x.HZx, dtype=dtype, device=device),
        HZy=torch.as_tensor(H_uncoupled_x.HZy, dtype=dtype, device=device),
        HZz=torch.as_tensor(H_uncoupled_x.HZz, dtype=dtype, device=device),
    )


def slow_hamiltonian_batch(
    terms: BatchedSlowHamiltonianTerms,
    E: np.ndarray | Any,
    B: np.ndarray | Any,
    *,
    out: Optional[Any] = None,
) -> Any:
    """Build slow Hamiltonians on the device holding ``terms``.

    ``E`` and ``B`` have shape ``(..., 3)`` and may be NumPy arrays or
    PyTorch tensors. ``out`` may be a preallocated tensor with shape
    ``(..., n, n)``. Fields are converted to the real precision of the terms,
    so the result retains the terms' complex dtype.
    """

    torch = get_torch()
    field_dtype = terms.Hff.real.dtype
    E_t = torch.as_tensor(E, dtype=field_dtype, device=terms.Hff.device)
    B_t = torch.as_tensor(B, dtype=field_dtype, device=terms.Hff.device)

    if E_t.shape[-1] != 3 or B_t.shape[-1] != 3:
        raise ValueError("E and B must have shape (..., 3)")

    Ex, Ey, Ez = (E_t[..., i, None, None] for i in range(3))
    Bx, By, Bz = (B_t[..., i, None, None] for i in range(3))
    two_pi = 2 * np.pi

    if out is None:
        return two_pi * (
            terms.Hff
            + Ex * terms.HSx
            + Ey * terms.HSy
            + Ez * terms.HSz
            + Bx * terms.HZx
            + By * terms.HZy
            + Bz * terms.HZz
        )

    out.copy_(terms.Hff)
    out.add_(Ex * terms.HSx)
    out.add_(Ey * terms.HSy)
    out.add_(Ez * terms.HSz)
    out.add_(Bx * terms.HZx)
    out.add_(By * terms.HZy)
    out.add_(Bz * terms.HZz)
    out.mul_(two_pi)
    return out
