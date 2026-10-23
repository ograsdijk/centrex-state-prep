from __future__ import annotations

from typing import Any, Tuple


def reorder_evecs_batched(
    V_in: Any,
    E_in: Any,
    V_ref: Any,
) -> Tuple[Any, Any]:
    """Greedily reorder eigenpairs by overlap, matching the previous GPU rule.

    This preserves the existing greedy GPU matching rule. It does not guarantee
    a one-to-one assignment near strong
    mixing; the CPU path uses an optimal assignment instead.
    """

    import torch

    overlap = torch.abs(torch.matmul(V_in.mH, V_ref))
    best_j = torch.argmax(overlap, dim=-1)
    order = torch.argsort(best_j, dim=-1, stable=True)
    V_out = torch.gather(
        V_in,
        -1,
        order.unsqueeze(-2).expand(*order.shape[:-1], V_in.shape[-2], -1),
    )
    E_out = torch.gather(E_in, -1, order)
    return E_out, V_out
