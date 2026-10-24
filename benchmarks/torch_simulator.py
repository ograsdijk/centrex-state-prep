from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TorchSimulationOutput:
    psi_final: torch.Tensor
    probabilities_final: torch.Tensor
    monitor_probabilities_final: torch.Tensor | None
    last_evecs: torch.Tensor


def _slow_hamiltonian(terms: tuple[torch.Tensor, ...], E, B):
    hff, hsx, hsy, hsz, hzx, hzy, hzz = terms
    return 2 * torch.pi * (
        hff + E[0] * hsx + E[1] * hsy + E[2] * hsz
        + B[0] * hzx + B[1] * hzy + B[2] * hzz
    )


def _reorder_eigenvectors(values, vectors, reference):
    from state_prep_gpu._reorder import reorder_evecs_batched

    return reorder_evecs_batched(vectors, values, reference)


def _eigh_in_blocks(matrix, blocks):
    """Eigh exact independent subspaces and return each eigenvector's block ID."""
    n = int(matrix.shape[-1])
    values_full = torch.empty(
        matrix.shape[:-2] + (n,), dtype=matrix.real.dtype, device=matrix.device
    )
    vectors_full = torch.zeros_like(matrix)
    labels_full = torch.empty(
        matrix.shape[:-2] + (n,), dtype=torch.int64, device=matrix.device
    )
    for block_id, block in enumerate(blocks):
        indices = torch.as_tensor(block, dtype=torch.int64, device=matrix.device)
        submatrix = matrix[..., indices[:, None], indices[None, :]]
        values, vectors = torch.linalg.eigh(submatrix)
        values_full[..., indices] = values
        vectors_full[..., indices[:, None], indices[None, :]] = vectors
        labels_full[..., indices] = block_id

    order = torch.argsort(values_full, dim=-1, stable=True)
    values_sorted = torch.gather(values_full, -1, order)
    vectors_sorted = torch.gather(
        vectors_full,
        -1,
        order.unsqueeze(-2).expand(*order.shape[:-1], n, n),
    )
    labels_sorted = torch.gather(labels_full, -1, order)
    return values_sorted, vectors_sorted, labels_sorted


def _rotating_frame_blocks(slow_labels, block_sizes):
    """Map slow-Hamiltonian sectors to their (possibly interleaved) columns."""
    grouped_indices = torch.argsort(slow_labels, stable=True)
    blocks = []
    start = 0
    for size in block_sizes:
        blocks.append(grouped_indices[start : start + size])
        start += size
    return blocks


def simulate_torch_shared_slow(
    *,
    terms: tuple[torch.Tensor, ...],
    E_t: torch.Tensor,
    B_t: torch.Tensor,
    psi0: torch.Tensor,
    dt: float,
    H_mu_base_fields: torch.Tensor,
    amp_t: torch.Tensor,
    coupling_scales: torch.Tensor,
    D_mu_diag_batch: torch.Tensor,
    E0: torch.Tensor,
    B0: torch.Tensor,
    monitor_state_vectors: torch.Tensor | None = None,
    time_chunk_size: int = 1,
    scan_chunk_size: int | None = None,
    propagator: str = "eigh",
    eigen_blocks: list[Any] | None = None,
) -> TorchSimulationOutput:
    """Full PyTorch implementation of the CuPy shared-slow SPA2 loop.

    All tensors are expected to be resident on one CUDA device for the whole
    evolution. Eigenvector matching delegates to the production optimal
    assignment and phase alignment; no CuPy operations are used here.
    """
    batch, n_states, n = psi0.shape
    T = E_t.shape[0]
    tones = H_mu_base_fields.shape[0]

    if eigen_blocks is not None and time_chunk_size != 1:
        raise ValueError("blockwise eigensolving currently requires time_chunk_size=1")
    if eigen_blocks is not None and propagator != "eigh":
        raise ValueError("blockwise eigensolving currently requires propagator='eigh'")
    if scan_chunk_size is None:
        scan_chunk_size = batch
    if scan_chunk_size < 1:
        raise ValueError("scan_chunk_size must be positive")

    if eigen_blocks is None:
        slow_values, V_ref = torch.linalg.eigh(_slow_hamiltonian(terms, E0, B0))
    else:
        slow_values, V_ref, _ = _eigh_in_blocks(
            _slow_hamiltonian(terms, E0, B0), eigen_blocks
        )
    del slow_values
    initial_reference = V_ref
    monitor_indices = None
    if monitor_state_vectors is not None:
        overlaps = torch.abs(initial_reference.mH @ monitor_state_vectors.mT)
        monitor_indices = torch.argmax(overlaps, dim=0)

    diag_indices = torch.arange(n, device=psi0.device)
    last_evecs = V_ref

    if time_chunk_size < 1:
        raise ValueError("time_chunk_size must be positive")
    if propagator not in ("eigh", "matrix_exp"):
        raise ValueError("propagator must be 'eigh' or 'matrix_exp'")
    if propagator == "matrix_exp" and time_chunk_size != 1:
        raise ValueError("matrix_exp currently requires time_chunk_size=1")

    if time_chunk_size == 1:
        step_ranges = ((i, i + 1) for i in range(T - 1))
    else:
        step_ranges = (
            (start, min(start + time_chunk_size, T - 1))
            for start in range(0, T - 1, time_chunk_size)
        )

    for start, end in step_ranges:
        if end - start == 1:
            step_data = ((start, None, None, None, None, None),)
        elif scan_chunk_size < batch:
            E_chunk = E_t[start:end]
            B_chunk = B_t[start:end]
            hff, hsx, hsy, hsz, hzx, hzy, hzz = terms
            H_slow_chunk = 2 * torch.pi * (
                hff
                + E_chunk[:, 0, None, None] * hsx
                + E_chunk[:, 1, None, None] * hsy
                + E_chunk[:, 2, None, None] * hsz
                + B_chunk[:, 0, None, None] * hzx
                + B_chunk[:, 1, None, None] * hzy
                + B_chunk[:, 2, None, None] * hzz
            )
            D_chunk, V_chunk = torch.linalg.eigh(H_slow_chunk)
            rotated_fields_chunk = V_chunk.mH[:, None] @ (
                H_mu_base_fields[None] @ V_chunk[:, None]
            )
            weights_all = coupling_scales[None] * amp_t[start:end, None, :]
            psi_next = torch.empty_like(psi0)

            # Track the shared slow eigenbasis once per step, independently of
            # the scan tiles. This preserves final-state labels while bounding
            # the scan-dependent rotating-frame eigensolver workspace.
            for i in range(start, end):
                _, tracked = _reorder_eigenvectors(
                    D_chunk[i - start].unsqueeze(0),
                    V_chunk[i - start].unsqueeze(0),
                    V_ref.unsqueeze(0),
                )
                V_ref = tracked[0]
                last_evecs = V_ref

            for tile_start in range(0, batch, scan_chunk_size):
                tile_end = min(tile_start + scan_chunk_size, batch)
                tile = slice(tile_start, tile_end)
                tile_weights = weights_all[:, tile]
                H_rot_chunk = torch.einsum(
                    "kbt,ktij->kbij", tile_weights, rotated_fields_chunk
                )
                H_rot_chunk[:, :, diag_indices, diag_indices] += D_chunk[:, None]
                H_rot_chunk[:, :, diag_indices, diag_indices] += D_mu_diag_batch[
                    tile
                ][None]
                D_rot_chunk, V_rot_chunk = torch.linalg.eigh(H_rot_chunk)

                psi_tile = psi0[tile]
                for i in range(end - start):
                    V_i = V_chunk[i]
                    D_rot_i = D_rot_chunk[i]
                    V_rot_i = V_rot_chunk[i]
                    psi_slow = psi_tile @ V_i.conj()
                    phases = torch.exp(-1j * D_rot_i * dt)
                    temp = psi_slow @ V_rot_i.conj()
                    temp = temp * phases[:, None, :]
                    psi_slow = temp @ V_rot_i.transpose(-2, -1)
                    psi_tile = psi_slow @ V_i.T
                psi_next[tile] = psi_tile

            psi0 = psi_next
            continue
        else:
            E_chunk = E_t[start:end]
            B_chunk = B_t[start:end]
            hff, hsx, hsy, hsz, hzx, hzy, hzz = terms
            H_slow_chunk = 2 * torch.pi * (
                hff
                + E_chunk[:, 0, None, None] * hsx
                + E_chunk[:, 1, None, None] * hsy
                + E_chunk[:, 2, None, None] * hsz
                + B_chunk[:, 0, None, None] * hzx
                + B_chunk[:, 1, None, None] * hzy
                + B_chunk[:, 2, None, None] * hzz
            )
            D_chunk, V_chunk = torch.linalg.eigh(H_slow_chunk)
            rotated_fields_chunk = V_chunk.mH[:, None] @ (
                H_mu_base_fields[None] @ V_chunk[:, None]
            )
            weights_chunk = coupling_scales[None] * amp_t[start:end, None, :]
            H_rot_chunk = torch.einsum(
                "kbt,ktij->kbij", weights_chunk, rotated_fields_chunk
            )
            H_rot_chunk[:, :, diag_indices, diag_indices] += D_chunk[:, None]
            H_rot_chunk[:, :, diag_indices, diag_indices] += D_mu_diag_batch[None]
            D_rot_chunk, V_rot_chunk = torch.linalg.eigh(H_rot_chunk)
            step_data = (
                (
                    i,
                    D_chunk[i - start],
                    V_chunk[i - start],
                    D_rot_chunk[i - start],
                    V_rot_chunk[i - start],
                    None,
                )
                for i in range(start, end)
            )

        for i, D_i, V_i, D_rot_i, V_rot_i, U_i in step_data:
            if D_i is None:
                H_slow = _slow_hamiltonian(terms, E_t[i], B_t[i])
                if eigen_blocks is None:
                    D_i, V_i = torch.linalg.eigh(H_slow)
                    slow_labels = None
                else:
                    D_i, V_i, slow_labels = _eigh_in_blocks(H_slow, eigen_blocks)
                weights = coupling_scales * amp_t[i].unsqueeze(0)
                if weights.shape != (batch, tones):
                    raise ValueError("microwave amplitude and coupling batch shapes do not match")
                rotated_fields = V_i.mH @ (H_mu_base_fields @ V_i)
                H_rot = torch.tensordot(weights, rotated_fields, dims=([1], [0]))
                H_rot[:, diag_indices, diag_indices] += D_i
                H_rot[:, diag_indices, diag_indices] += D_mu_diag_batch
                if propagator == "matrix_exp":
                    U_i = torch.linalg.matrix_exp(
                        -1j * H_rot.transpose(-2, -1) * dt
                    )
                else:
                    if eigen_blocks is None:
                        D_rot_i, V_rot_i = torch.linalg.eigh(H_rot)
                    else:
                        rotating_blocks = _rotating_frame_blocks(
                            slow_labels,
                            [len(block) for block in eigen_blocks],
                        )
                        D_rot_i, V_rot_i, _ = _eigh_in_blocks(
                            H_rot, rotating_blocks
                        )

            _, tracked = _reorder_eigenvectors(
                D_i.unsqueeze(0), V_i.unsqueeze(0), V_ref.unsqueeze(0)
            )
            V_ref = tracked[0]
            last_evecs = V_ref

            psi_slow = psi0 @ V_i.conj()
            if U_i is not None:
                psi_slow = psi_slow @ U_i
            else:
                phases = torch.exp(-1j * D_rot_i * dt)
                temp = psi_slow @ V_rot_i.conj()
                temp = temp * phases[:, None, :]
                psi_slow = temp @ V_rot_i.transpose(-2, -1)
            psi0 = psi_slow @ V_i.T

    amplitudes = psi0 @ last_evecs.conj()
    probabilities = torch.abs(amplitudes) ** 2
    monitor_probabilities = (
        probabilities[:, :, monitor_indices] if monitor_indices is not None else None
    )
    return TorchSimulationOutput(
        psi_final=psi0,
        probabilities_final=probabilities,
        monitor_probabilities_final=monitor_probabilities,
        last_evecs=last_evecs,
    )
