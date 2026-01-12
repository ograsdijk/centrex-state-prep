from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ._cupy import cupy

if TYPE_CHECKING:  # pragma: no cover
    import cupy as cp


@dataclass(frozen=True)
class BatchedSlowHamiltonianTerms:
    """Field-independent matrix terms for the slow Hamiltonian.

    Mirrors centrex_tlf's `HamiltonianUncoupledX` structure:

        H(E,B) = 2π (Hff + Ex HSx + Ey HSy + Ez HSz + Bx HZx + By HZy + Bz HZz)

    All matrices are CuPy arrays with shape (n, n).
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
) -> BatchedSlowHamiltonianTerms:
    """Convert a centrex_tlf `HamiltonianUncoupledX` dataclass to CuPy terms."""

    cp = cupy()
    if dtype is None:
        dtype = cp.complex128

    return BatchedSlowHamiltonianTerms(
        Hff=cp.asarray(H_uncoupled_x.Hff, dtype=dtype),
        HSx=cp.asarray(H_uncoupled_x.HSx, dtype=dtype),
        HSy=cp.asarray(H_uncoupled_x.HSy, dtype=dtype),
        HSz=cp.asarray(H_uncoupled_x.HSz, dtype=dtype),
        HZx=cp.asarray(H_uncoupled_x.HZx, dtype=dtype),
        HZy=cp.asarray(H_uncoupled_x.HZy, dtype=dtype),
        HZz=cp.asarray(H_uncoupled_x.HZz, dtype=dtype),
    )


def slow_hamiltonian_batch(
    terms: BatchedSlowHamiltonianTerms,
    E: np.ndarray | Any,
    B: np.ndarray | Any,
    *,
    out: Optional[Any] = None,
) -> Any:
    """Compute a batch of slow Hamiltonians on GPU.

    Parameters
    ----------
    terms:
        Precomputed field-independent matrices (CuPy arrays).
    E, B:
        Arrays with shape (..., 3). Can be NumPy or CuPy.
    out:
        Optional preallocated CuPy array with shape (..., n, n).

    Returns
    -------
    H:
        CuPy array with shape (..., n, n), dtype complex.
    """

    cp = cupy()

    E_cp = cp.asarray(E)
    B_cp = cp.asarray(B)

    if E_cp.shape[-1] != 3 or B_cp.shape[-1] != 3:
        raise ValueError("E and B must have shape (..., 3)")

    # Broadcast scalars to (..., 1, 1)
    Ex = E_cp[..., 0][..., None, None]
    Ey = E_cp[..., 1][..., None, None]
    Ez = E_cp[..., 2][..., None, None]
    Bx = B_cp[..., 0][..., None, None]
    By = B_cp[..., 1][..., None, None]
    Bz = B_cp[..., 2][..., None, None]

    two_pi = cp.asarray(2 * np.pi, dtype=terms.Hff.dtype)

    if out is None:
        H = (
            terms.Hff
            + Ex * terms.HSx
            + Ey * terms.HSy
            + Ez * terms.HSz
            + Bx * terms.HZx
            + By * terms.HZy
            + Bz * terms.HZz
        )
        return two_pi * H

    out[...] = terms.Hff
    out[...] += Ex * terms.HSx
    out[...] += Ey * terms.HSy
    out[...] += Ez * terms.HSz
    out[...] += Bx * terms.HZx
    out[...] += By * terms.HZy
    out[...] += Bz * terms.HZz
    out[...] *= two_pi
    return out
