"""Exact support detection and block eigensolves for shared-slow scans."""
from __future__ import annotations


def shared_slow_blocks(terms, E_t, B_t, E0, B0, microwave_fields):
    """Union every potentially active coupling, including initial fields.

    This conservative term union may merge blocks that cancellation would split,
    but never drops a small coupling. Only the final support graph goes to CPU,
    once during preparation; no per-timestep block detection or host transfer.
    """
    import torch
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    matrices = torch.stack([getattr(terms, name) for name in
                            ("Hff", "HSx", "HSy", "HSz", "HZx", "HZy", "HZz")])
    active = torch.cat((torch.ones(1, dtype=torch.bool, device=E_t.device),
                        (E_t != 0).any(dim=0) | (E0 != 0),
                        (B_t != 0).any(dim=0) | (B0 != 0)))
    support = ((matrices != 0) & active[:, None, None]).any(dim=0)
    support |= (microwave_fields != 0).any(dim=0)
    support |= support.T.clone()
    count, labels = connected_components(csr_matrix(support.cpu().numpy()), directed=False)
    groups = [[i for i, label in enumerate(labels) if label == k] for k in range(count)]
    if count <= 1 or sum(len(group) ** 3 for group in groups) >= 0.8 * terms.n ** 3:
        return None
    return [torch.tensor(group, dtype=torch.int64, device=E_t.device) for group in groups]


def eigh_in_blocks(matrix, blocks):
    """Solve exact blocks and sort the complete spectrum with sector labels."""
    import torch

    n = matrix.shape[-1]
    values_full = torch.empty(matrix.shape[:-2] + (n,), dtype=matrix.real.dtype, device=matrix.device)
    vectors_full = torch.zeros_like(matrix)
    labels_full = torch.empty(matrix.shape[:-2] + (n,), dtype=torch.int64, device=matrix.device)
    # Equal-sized sectors can share one batched solver call. Keep each sector's
    # original indices/label: their columns move after every slow-spectrum sort.
    by_width = {}
    for block_id, indices in enumerate(blocks):
        by_width.setdefault(len(indices), []).append((block_id, indices))
        labels_full[..., indices] = block_id
    for width, sectors in by_width.items():
        indices = torch.stack([indices for _, indices in sectors])
        rows, columns = indices[:, :, None], indices[:, None, :]
        submatrix = matrix[..., rows, columns]
        if width == 1:
            values = submatrix[..., 0].real
            vectors = torch.ones_like(submatrix)
        else:
            values, vectors = torch.linalg.eigh(submatrix)
        values_full[..., indices] = values
        vectors_full[..., rows, columns] = vectors
    order = torch.argsort(values_full, dim=-1, stable=True)
    return (
        torch.gather(values_full, -1, order),
        torch.gather(vectors_full, -1, order.unsqueeze(-2).expand(*order.shape[:-1], n, n)),
        torch.gather(labels_full, -1, order),
    )


def rotating_frame_blocks(labels, blocks):
    """Recompute sector columns after each slow spectrum sort, on-device."""
    import torch

    order = torch.argsort(labels, stable=True)
    return list(torch.split(order, [len(block) for block in blocks]))
