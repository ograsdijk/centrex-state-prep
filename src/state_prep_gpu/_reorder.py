from __future__ import annotations

from typing import Any, Tuple

from ._assignment import linear_assignment


def reorder_evecs_batched(
    V_in: Any,
    E_in: Any,
    V_ref: Any,
) -> Tuple[Any, Any]:
    """Maximize total absolute overlap one-to-one and align reference phases.

    Uses the CPU matching objective. CUDA costs, assignment and phase alignment
    remain on-device. Equally optimal ties need not give identical indices.
    """

    import torch

    overlap = torch.matmul(V_ref.mH, V_in)
    order = linear_assignment(-torch.abs(overlap))
    V_out = torch.gather(
        V_in,
        -1,
        order.unsqueeze(-2).expand(*order.shape[:-1], V_in.shape[-2], -1),
    )
    E_out = torch.gather(E_in, -1, order)
    phases = torch.gather(overlap, -1, order.unsqueeze(-1)).squeeze(-1)
    magnitude = torch.abs(phases)
    alignment = torch.where(
        magnitude > 0,
        phases / magnitude.clamp_min(torch.finfo(magnitude.dtype).tiny),
        torch.ones_like(phases),
    )
    V_out = V_out * alignment.conj().unsqueeze(-2)
    return E_out, V_out
