from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ._torch import torch as get_torch
from ._reorder import reorder_evecs_batched
from .slow_hamiltonian import BatchedSlowHamiltonianTerms, slow_hamiltonian_batch


@dataclass(frozen=True)
class BatchedSimulationOutput:
    psi_final: Any
    probabilities_final: Optional[Any]
    monitor_probabilities_final: Optional[Any]
    last_evecs: Any
    psi_history: Optional[Any] = None
    history_indices: Optional[Any] = None


def _validate_store_every(store_every: Optional[int]) -> Optional[int]:
    if store_every is None:
        return None
    if isinstance(store_every, bool) or int(store_every) != store_every or store_every < 1:
        raise ValueError("store_every must be None or a positive integer")
    return int(store_every)


def _new_history(torch: Any, psi: Any, T: int, store_every: Optional[int]):
    if store_every is None:
        return None, None, 0
    indices = list(range(0, T, store_every))
    if indices[-1] != T - 1:
        indices.append(T - 1)
    history = torch.empty(
        (len(indices), *psi.shape), dtype=psi.dtype, device=psi.device
    )
    history[0] = psi
    return history, indices, 1


def _store_history(history: Any, indices: Optional[list[int]], cursor: int,
                   step: int, psi: Any) -> int:
    if indices is not None and cursor < len(indices) and indices[cursor] == step:
        history[cursor] = psi
        return cursor + 1
    return cursor


def _normalize_dt(torch: Any, dt: Any, T: int, batch: int, device: Any) -> Any:
    dt_arr = torch.as_tensor(dt, dtype=torch.float64, device=device)
    if dt_arr.ndim == 0:
        return torch.full((T - 1,), float(dt_arr.item()), dtype=torch.float64, device=device)
    if dt_arr.shape not in ((T - 1,), (T - 1, batch)):
        raise ValueError("dt must be scalar, (T-1,), or (T-1,B)")
    return dt_arr


def _dt_for_step(dt_arr: Any, step: int) -> Any:
    return dt_arr[step] if dt_arr.ndim == 1 else dt_arr[step, :, None]


def _initial_field_batch(torch: Any, field: Any, fallback: Any, batch: int) -> Any:
    values = torch.as_tensor(
        fallback if field is None else field, device=fallback.device
    )
    if values.shape == (3,):
        return torch.broadcast_to(values[None, :], (batch, 3))
    if values.shape != (batch, 3):
        raise ValueError("initial field must have shape (3,) or (B,3)")
    return values


def simulate_batched_compact_microwaves(
    *,
    terms: BatchedSlowHamiltonianTerms,
    E_t: Any,
    B_t: Any,
    psi0: Any,
    dt: float | Any,
    Hu: Any,
    Hl: Any,
    coeff_u_t: Any,
    coeff_l_t: Any,
    coupling_scales: Any,
    D_mu_diag_batch: Any,
    monitor_state_vectors: Optional[Any] = None,
    return_final_probabilities: bool = True,
) -> BatchedSimulationOutput:
    """Batched GPU time evolution with microwaves (compact representation).

    This avoids constructing a dense `H_mu_t` array with shape (T-1,B,n,n).
    Instead, the microwave Hamiltonian for each field is built on-GPU each step:

        H_mu(field, t) = sum_k coeff_u[t, field, k] * Hu[field, k]
                      + sum_k coeff_l[t, field, k] * Hl[field, k]

    and then combined per batch point via `coupling_scales[b, field]`.

    Shapes
    ------
    - E_t, B_t: (T,B,3) or (T,3)
    - psi0: (B,S,n)
    - Hu, Hl: (M,3,n,n)
    - coeff_u_t, coeff_l_t: (T-1,M,3)
    - coupling_scales: (B,M)
    - D_mu_diag_batch: (B,n)

    Notes
    -----
    `D_mu_diag_batch` is added to the *diagonal of the rotating-frame Hamiltonian*
    (matching the CPU batched scan semantics).
    """

    torch = get_torch()

    E_t = torch.as_tensor(E_t, device=terms.Hff.device)
    B_t = torch.as_tensor(B_t, device=terms.Hff.device)

    if E_t.ndim == 2:
        # (T,3) -> (T,B,3)
        if E_t.shape[-1] != 3:
            raise ValueError("E_t must have shape (T,3) or (T,B,3)")
        if B_t.ndim != 2 or B_t.shape != E_t.shape:
            raise ValueError("If E_t is (T,3), B_t must also be (T,3)")
        psi_probe = torch.as_tensor(psi0, device=terms.Hff.device)
        if psi_probe.ndim != 3:
            raise ValueError("psi0 must have shape (B,S,n)")
        batch = int(psi_probe.shape[0])
        E_t = torch.broadcast_to(E_t[:, None, :], (E_t.shape[0], batch, 3))
        B_t = torch.broadcast_to(B_t[:, None, :], (B_t.shape[0], batch, 3))

    psi = _as_torch_complex(psi0, dtype=terms.Hff.dtype, device=terms.Hff.device)

    if E_t.ndim != 3 or B_t.ndim != 3 or E_t.shape[-1] != 3 or B_t.shape[-1] != 3:
        raise ValueError("E_t and B_t must have shape (T, B, 3)")

    T, batch, _ = E_t.shape
    if B_t.shape[0] != T or B_t.shape[1] != batch:
        raise ValueError("B_t must have the same (T,B,3) shape as E_t")

    if psi.ndim != 3 or psi.shape[0] != batch:
        raise ValueError("psi0 must have shape (B, S, n)")

    n = terms.n
    if psi.shape[-1] != n:
        raise ValueError(f"psi0 last dimension must be n={n}")

    if isinstance(dt, (float, int)):
        dt_arr = torch.full((T - 1,), float(dt), dtype=torch.float64, device=psi.device)
    else:
        dt_arr = torch.as_tensor(dt, dtype=torch.float64, device=psi.device)
        if dt_arr.shape != (T - 1,):
            raise ValueError("dt must be a scalar or have shape (T-1,)")

    Hu = _as_torch_complex(Hu, dtype=terms.Hff.dtype, device=terms.Hff.device)
    Hl = _as_torch_complex(Hl, dtype=terms.Hff.dtype, device=terms.Hff.device)
    if Hu.shape != Hl.shape or Hu.ndim != 4 or Hu.shape[1] != 3 or Hu.shape[2] != n or Hu.shape[3] != n:
        raise ValueError("Hu and Hl must both have shape (M,3,n,n)")
    M = int(Hu.shape[0])

    coeff_u_t = _as_torch_complex(coeff_u_t, dtype=terms.Hff.dtype, device=terms.Hff.device)
    coeff_l_t = _as_torch_complex(coeff_l_t, dtype=terms.Hff.dtype, device=terms.Hff.device)
    if coeff_u_t.shape != (T - 1, M, 3) or coeff_l_t.shape != (T - 1, M, 3):
        raise ValueError("coeff_u_t and coeff_l_t must have shape (T-1,M,3)")

    coupling_scales = torch.as_tensor(coupling_scales, dtype=torch.float64, device=psi.device)
    if coupling_scales.shape != (batch, M):
        raise ValueError("coupling_scales must have shape (B,M)")

    D_mu_diag_batch = torch.as_tensor(D_mu_diag_batch, device=psi.device)
    if D_mu_diag_batch.shape != (batch, n):
        raise ValueError("D_mu_diag_batch must have shape (B,n)")

    # Initial slow Hamiltonian and reference eigenvectors (for adiabatic tracking)
    H0 = slow_hamiltonian_batch(terms, E_t[0], B_t[0])  # (B,n,n)
    D0, V0 = torch.linalg.eigh(H0)
    V_ref = V0
    V_ref_ini = V0

    # Monitor index mapping at t=0 (closest eigenstate)
    monitor_idx = None
    if monitor_state_vectors is not None:
        mv = _as_torch_complex(monitor_state_vectors, dtype=V_ref_ini.dtype, device=psi.device)
        if mv.ndim == 2:
            mv = torch.broadcast_to(mv[None, :, :], (batch, mv.shape[0], mv.shape[1]))
        if mv.shape[0] != batch or mv.shape[-1] != n:
            raise ValueError("monitor_state_vectors must have shape (M,n) or (B,M,n)")

        overlap = torch.abs(
            torch.matmul(
                torch.conj(torch.swapaxes(V_ref_ini, -2, -1)),
                torch.swapaxes(mv, -2, -1),
            )
        )
        monitor_idx = torch.argmax(overlap, dim=1).to(torch.int64)

    diag_idx = torch.arange(n, device=psi.device)

    last_evecs = V_ref
    for i in range(T - 1):
        H_slow = slow_hamiltonian_batch(terms, E_t[i], B_t[i])  # (B,n,n)
        D, V = torch.linalg.eigh(H_slow)

        # Track eigenvectors (for reporting/probabilities/monitor states)
        _, evecs_tracked = reorder_evecs_batched(V, D, V_ref)
        last_evecs = evecs_tracked

        # Build per-field microwave Hamiltonians on GPU (M,n,n)
        cu = coeff_u_t[i]  # (M,3)
        cl = coeff_l_t[i]  # (M,3)
        H_fields = (
            cu[:, 0, None, None] * Hu[:, 0]
            + cu[:, 1, None, None] * Hu[:, 1]
            + cu[:, 2, None, None] * Hu[:, 2]
            + cl[:, 0, None, None] * Hl[:, 0]
            + cl[:, 1, None, None] * Hl[:, 1]
            + cl[:, 2, None, None] * Hl[:, 2]
        )

        # Combine across microwave fields per batch point: (B,M) x (M,n,n) -> (B,n,n)
        H_mu = torch.tensordot(
            coupling_scales.to(dtype=H_fields.dtype),
            H_fields,
            dims=([1], [0]),
        )

        # Rotating-frame Hamiltonian in slow-eigenbasis
        H_rot = torch.matmul(torch.conj(torch.swapaxes(V, -2, -1)), torch.matmul(H_mu, V))
        # Add diagonal shifts: slow energies + detuning shifts (CPU batched semantics)
        H_rot[:, diag_idx, diag_idx] += D
        H_rot[:, diag_idx, diag_idx] += D_mu_diag_batch

        D_rot, V_rot = torch.linalg.eigh(H_rot)
        A = torch.matmul(V, V_rot)

        phases_rot = torch.exp(-1j * D_rot * dt_arr[i])
        tmp = torch.matmul(psi, torch.conj(A))
        tmp *= phases_rot[:, None, :]
        psi = torch.matmul(tmp, torch.swapaxes(A, -2, -1))

        V_ref = evecs_tracked

    probabilities_final = None
    if return_final_probabilities:
        probabilities_final = _batched_probabilities(psi, last_evecs)

    monitor_probabilities_final = None
    if monitor_idx is not None:
        probs_all = _batched_probabilities(psi, last_evecs)  # (B,S,n)
        monitor_probabilities_final = torch.gather(probs_all, -1, monitor_idx[:, None, :].expand(-1, probs_all.shape[1], -1))

    return BatchedSimulationOutput(
        psi_final=psi,
        probabilities_final=probabilities_final,
        monitor_probabilities_final=monitor_probabilities_final,
        last_evecs=last_evecs,
    )


def simulate_batched_scalar_microwaves(
    *,
    terms: BatchedSlowHamiltonianTerms,
    E_t: Any,
    B_t: Any,
    psi0: Any,
    dt: float | Any,
    H_mu_base_fields: Any,
    amp_t: Any,
    coupling_scales: Any,
    D_mu_diag_batch: Any,
    E0: Optional[Any] = None,
    B0: Optional[Any] = None,
    monitor_state_vectors: Optional[Any] = None,
    return_final_probabilities: bool = True,
    store_every: Optional[int] = None,
) -> BatchedSimulationOutput:
    """Batched GPU time evolution with microwaves (scalar envelope).

    Assumes polarization (and all other microwave matrix structure) is fixed for
    each microwave field across the full evolution. Then for each field j:

        H_mu_j(t) = amp_t[t, j] * H_mu_base_fields[j]

    and for each batch point b:

        H_mu(t)[b] = sum_j coupling_scales[b, j] * H_mu_j(t)

    Shapes
    ------
    - E_t, B_t: (T,B,3) or (T,3)
    - psi0: (B,S,n)
    - H_mu_base_fields: (M,n,n)
    - amp_t: (T-1,M) or (T-1,B,M)
    - coupling_scales: (B,M)
    - D_mu_diag_batch: (B,n)
    - dt: scalar, (T-1,), or per-point (T-1,B)
    - E0, B0: optional true initial fields, shape (3,) or (B,3). When omitted,
      the first entries of E_t/B_t are also used for initialization.
    - store_every: optional stride for retaining state vectors, including the
      initial and final states. `None` keeps final-only storage.
    """

    torch = get_torch()

    E_t = torch.as_tensor(E_t, device=terms.Hff.device)
    B_t = torch.as_tensor(B_t, device=terms.Hff.device)

    if E_t.ndim == 2:
        if E_t.shape[-1] != 3:
            raise ValueError("E_t must have shape (T,3) or (T,B,3)")
        if B_t.ndim != 2 or B_t.shape != E_t.shape:
            raise ValueError("If E_t is (T,3), B_t must also be (T,3)")
        psi_probe = torch.as_tensor(psi0, device=terms.Hff.device)
        if psi_probe.ndim != 3:
            raise ValueError("psi0 must have shape (B,S,n)")
        batch = int(psi_probe.shape[0])
        E_t = torch.broadcast_to(E_t[:, None, :], (E_t.shape[0], batch, 3))
        B_t = torch.broadcast_to(B_t[:, None, :], (B_t.shape[0], batch, 3))

    psi = _as_torch_complex(psi0, dtype=terms.Hff.dtype, device=terms.Hff.device)

    if E_t.ndim != 3 or B_t.ndim != 3 or E_t.shape[-1] != 3 or B_t.shape[-1] != 3:
        raise ValueError("E_t and B_t must have shape (T, B, 3)")

    T, batch, _ = E_t.shape
    if B_t.shape[0] != T or B_t.shape[1] != batch:
        raise ValueError("B_t must have the same (T,B,3) shape as E_t")

    if psi.ndim != 3 or psi.shape[0] != batch:
        raise ValueError("psi0 must have shape (B, S, n)")

    n = terms.n
    if psi.shape[-1] != n:
        raise ValueError(f"psi0 last dimension must be n={n}")

    store_every = _validate_store_every(store_every)

    dt_arr = _normalize_dt(torch, dt, T, batch, psi.device)

    H_mu_base_fields = _as_torch_complex(H_mu_base_fields, dtype=terms.Hff.dtype, device=terms.Hff.device)
    if H_mu_base_fields.ndim != 3 or H_mu_base_fields.shape[1] != n or H_mu_base_fields.shape[2] != n:
        raise ValueError("H_mu_base_fields must have shape (M,n,n)")
    M = int(H_mu_base_fields.shape[0])

    amp_t = _as_torch_complex(amp_t, dtype=terms.Hff.dtype, device=terms.Hff.device)
    if amp_t.shape not in ((T - 1, M), (T - 1, batch, M)):
        raise ValueError("amp_t must have shape (T-1,M) or (T-1,B,M)")

    coupling_scales = torch.as_tensor(coupling_scales, dtype=torch.float64, device=psi.device)
    if coupling_scales.shape != (batch, M):
        raise ValueError("coupling_scales must have shape (B,M)")

    D_mu_diag_batch = torch.as_tensor(D_mu_diag_batch, device=psi.device)
    if D_mu_diag_batch.shape != (batch, n):
        raise ValueError("D_mu_diag_batch must have shape (B,n)")

    # Initial slow Hamiltonian and reference eigenvectors (for adiabatic tracking)
    E_initial = _initial_field_batch(torch, E0, E_t[0], batch)
    B_initial = _initial_field_batch(torch, B0, B_t[0], batch)
    H0 = slow_hamiltonian_batch(terms, E_initial, B_initial)  # (B,n,n)
    _, V0 = torch.linalg.eigh(H0)
    V_ref = V0
    V_ref_ini = V0

    monitor_idx = None
    if monitor_state_vectors is not None:
        mv = _as_torch_complex(monitor_state_vectors, dtype=V_ref_ini.dtype, device=psi.device)
        if mv.ndim == 2:
            mv = torch.broadcast_to(mv[None, :, :], (batch, mv.shape[0], mv.shape[1]))
        if mv.shape[0] != batch or mv.shape[-1] != n:
            raise ValueError("monitor_state_vectors must have shape (M,n) or (B,M,n)")

        overlap = torch.abs(
            torch.matmul(
                torch.conj(torch.swapaxes(V_ref_ini, -2, -1)),
                torch.swapaxes(mv, -2, -1),
            )
        )
        monitor_idx = torch.argmax(overlap, dim=1).to(torch.int64)

    diag_idx = torch.arange(n, device=psi.device)

    psi_history, history_indices, history_cursor = _new_history(
        torch, psi, T, store_every
    )

    last_evecs = V_ref
    for i in range(T - 1):
        H_slow = slow_hamiltonian_batch(terms, E_t[i], B_t[i])  # (B,n,n)
        D, V = torch.linalg.eigh(H_slow)

        _, evecs_tracked = reorder_evecs_batched(V, D, V_ref)
        last_evecs = evecs_tracked

        # weights: (B,M) complex
        amp_i = amp_t[i][None, :] if amp_t.ndim == 2 else amp_t[i]
        weights = coupling_scales * amp_i
        weights = _as_torch_complex(weights, dtype=terms.Hff.dtype, device=terms.Hff.device)

        # Combine across fields per batch: (B,M) x (M,n,n) -> (B,n,n)
        H_mu = torch.tensordot(weights, H_mu_base_fields, dims=([1], [0]))

        H_rot = torch.matmul(torch.conj(torch.swapaxes(V, -2, -1)), torch.matmul(H_mu, V))
        H_rot[:, diag_idx, diag_idx] += D
        H_rot[:, diag_idx, diag_idx] += D_mu_diag_batch

        D_rot, V_rot = torch.linalg.eigh(H_rot)
        A = torch.matmul(V, V_rot)

        phases_rot = torch.exp(-1j * D_rot * _dt_for_step(dt_arr, i))
        tmp = torch.matmul(psi, torch.conj(A))
        tmp *= phases_rot[:, None, :]
        psi = torch.matmul(tmp, torch.swapaxes(A, -2, -1))

        history_cursor = _store_history(
            psi_history, history_indices, history_cursor, i + 1, psi
        )

        V_ref = evecs_tracked

    probabilities_final = None
    if return_final_probabilities:
        probabilities_final = _batched_probabilities(psi, last_evecs)

    monitor_probabilities_final = None
    if monitor_idx is not None:
        probs_all = _batched_probabilities(psi, last_evecs)
        monitor_probabilities_final = torch.gather(probs_all, -1, monitor_idx[:, None, :].expand(-1, probs_all.shape[1], -1))

    return BatchedSimulationOutput(
        psi_final=psi,
        probabilities_final=probabilities_final,
        monitor_probabilities_final=monitor_probabilities_final,
        last_evecs=last_evecs,
        psi_history=psi_history,
        history_indices=(torch.tensor(history_indices, dtype=torch.int64, device=psi.device) if history_indices is not None else None),
    )


def simulate_batched_scalar_microwaves_shared_slow(
    *,
    terms: BatchedSlowHamiltonianTerms,
    E_t: Any,
    B_t: Any,
    psi0: Any,
    dt: float | Any,
    H_mu_base_fields: Any,
    amp_t: Any,
    coupling_scales: Any,
    D_mu_diag_batch: Any,
    E0: Optional[Any] = None,
    B0: Optional[Any] = None,
    monitor_state_vectors: Optional[Any] = None,
    return_final_probabilities: bool = True,
    store_every: Optional[int] = None,
    profile_timings: Optional[dict[str, list[float]]] = None,
    profile_step_stride: int = 25,
) -> BatchedSimulationOutput:
    """GPU time evolution for microwave scans with a *shared* slow Hamiltonian.

    This matches the CPU batched-scan idea: H_slow(t) is identical for all scan
    points, so we diagonalize it only once per timestep.

    Expected shapes
    ---------------
    - E_t, B_t: (T,3) (shared across batch)
    - psi0: (B,S,n)
    - H_mu_base_fields: (M,n,n)
    - amp_t: (T-1,M) or (T-1,B,M)
    - coupling_scales: (B,M)
    - D_mu_diag_batch: (B,n)
    - monitor_state_vectors: (K,n) optional (shared)
    - dt: scalar, (T-1,), or per-point (T-1,B)
    - E0, B0: optional true initial fields, each shape (3,)
    - store_every: optional stride for retaining state vectors, including the
      initial and final states. `None` keeps final-only storage.
    - profile_timings: optional dict populated with sampled CUDA-event timings
      for propagation stages. Profiling synchronizes once at the end.
    - profile_step_stride: sample one step every this many propagation steps.
    """

    torch = get_torch()

    E_t = torch.as_tensor(E_t, device=terms.Hff.device)
    B_t = torch.as_tensor(B_t, device=terms.Hff.device)
    psi = _as_torch_complex(psi0, dtype=terms.Hff.dtype, device=terms.Hff.device)

    if E_t.ndim != 2 or B_t.ndim != 2 or E_t.shape[-1] != 3 or B_t.shape[-1] != 3:
        raise ValueError("E_t and B_t must have shape (T,3) for shared-slow evolution")
    if E_t.shape != B_t.shape:
        raise ValueError("E_t and B_t must have identical shape (T,3)")

    if psi.ndim != 3:
        raise ValueError("psi0 must have shape (B,S,n)")
    batch = int(psi.shape[0])

    n = terms.n
    if psi.shape[-1] != n:
        raise ValueError(f"psi0 last dimension must be n={n}")

    store_every = _validate_store_every(store_every)

    T = int(E_t.shape[0])
    if T < 2:
        raise ValueError("E_t/B_t must have T>=2")

    dt_arr = _normalize_dt(torch, dt, T, batch, psi.device)

    H_mu_base_fields = _as_torch_complex(H_mu_base_fields, dtype=terms.Hff.dtype, device=terms.Hff.device)
    if H_mu_base_fields.ndim != 3 or H_mu_base_fields.shape[1] != n or H_mu_base_fields.shape[2] != n:
        raise ValueError("H_mu_base_fields must have shape (M,n,n)")
    M = int(H_mu_base_fields.shape[0])

    amp_t = _as_torch_complex(amp_t, dtype=terms.Hff.dtype, device=terms.Hff.device)
    if amp_t.shape not in ((T - 1, M), (T - 1, batch, M)):
        raise ValueError("amp_t must have shape (T-1,M) or (T-1,B,M)")

    coupling_scales = torch.as_tensor(coupling_scales, dtype=torch.float64, device=psi.device)
    if coupling_scales.shape != (batch, M):
        raise ValueError("coupling_scales must have shape (B,M)")

    D_mu_diag_batch = torch.as_tensor(D_mu_diag_batch, device=psi.device)
    if D_mu_diag_batch.shape != (batch, n):
        raise ValueError("D_mu_diag_batch must have shape (B,n)")

    # Initial slow Hamiltonian and reference eigenvectors (for adiabatic tracking)
    E_initial = E_t[0] if E0 is None else torch.as_tensor(E0, device=psi.device)
    B_initial = B_t[0] if B0 is None else torch.as_tensor(B0, device=psi.device)
    if E_initial.shape != (3,) or B_initial.shape != (3,):
        raise ValueError("E0 and B0 must each have shape (3,) for shared-slow evolution")
    H0 = slow_hamiltonian_batch(terms, E_initial, B_initial)  # (n,n)
    _, V0 = torch.linalg.eigh(H0)
    V_ref = V0
    V_ref_ini = V0

    monitor_idx = None
    if monitor_state_vectors is not None:
        mv = _as_torch_complex(monitor_state_vectors, dtype=V_ref_ini.dtype, device=psi.device)
        if mv.ndim != 2 or mv.shape[-1] != n:
            raise ValueError("monitor_state_vectors must have shape (K,n)")
        # overlap[i,k] = | <v_i | m_k> |
        overlap = torch.abs(torch.matmul(torch.conj(V_ref_ini.T), mv.T))  # (n,K)
        monitor_idx = torch.argmax(overlap, dim=0).to(torch.int64)  # (K,)

    diag_idx = torch.arange(n, device=psi.device)
    if profile_step_stride < 1:
        raise ValueError("profile_step_stride must be >= 1")
    if profile_timings is not None and psi.device.type != "cuda":
        raise ValueError("profile_timings requires CUDA tensors")
    profile_events: dict[str, list[tuple[Any, Any]]] = {}

    psi_history, history_indices, history_cursor = _new_history(
        torch, psi, T, store_every
    )

    last_evecs = V_ref
    for i in range(T - 1):
        profile_this_step = profile_timings is not None and i % profile_step_stride == 0
        H_slow = slow_hamiltonian_batch(terms, E_t[i], B_t[i])  # (n,n)
        if profile_this_step:
            slow_start, slow_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            slow_start.record()
        D, V = torch.linalg.eigh(H_slow)
        if profile_this_step:
            slow_end.record()
            profile_events.setdefault("slow_eigh", []).append((slow_start, slow_end))

        # Track eigenvectors for output/monitor semantics
        if profile_this_step:
            track_start, track_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            track_start.record()
        _, evecs_tracked_b = reorder_evecs_batched(V[None, :, :], D[None, :], V_ref[None, :, :])
        evecs_tracked = evecs_tracked_b[0]
        last_evecs = evecs_tracked
        if profile_this_step:
            track_end.record()
            profile_events.setdefault("track_slow_eigenvectors", []).append((track_start, track_end))

        # weights: (B,M) complex
        amp_i = amp_t[i][None, :] if amp_t.ndim == 2 else amp_t[i]
        weights = coupling_scales * amp_i
        weights = _as_torch_complex(weights, dtype=terms.Hff.dtype, device=terms.Hff.device)

        # V is shared across scan points. Rotate each microwave field once,
        # then combine the rotated fields per point. Rotating a combined
        # (B,n,n) Hamiltonian would repeat two dense n x n multiplies B times.
        if profile_this_step:
            assemble_start, assemble_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            assemble_start.record()
        H_mu_rot_fields = torch.matmul(
            torch.conj(V.T), torch.matmul(H_mu_base_fields, V)
        )  # (M,n,n)
        H_rot = torch.tensordot(weights, H_mu_rot_fields, dims=([1], [0]))  # (B,n,n)
        H_rot[:, diag_idx, diag_idx] += D
        H_rot[:, diag_idx, diag_idx] += D_mu_diag_batch
        if profile_this_step:
            assemble_end.record()
            profile_events.setdefault("microwave_rotate_and_assemble", []).append((assemble_start, assemble_end))

        if profile_this_step:
            rot_start, rot_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            rot_start.record()
        D_rot, V_rot = torch.linalg.eigh(H_rot)
        if profile_this_step:
            rot_end.record()
            profile_events.setdefault("batched_rotating_eigh", []).append((rot_start, rot_end))

        # Factor A = V @ V_rot in the state update. This avoids constructing
        # one dense A matrix per scan point and matches the CPU shared-basis
        # update: transform the states into the shared slow basis, propagate
        # with V_rot, then transform back once for all scan points.
        if profile_this_step:
            update_start, update_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            update_start.record()
        psi_slow = torch.matmul(psi, torch.conj(V))
        phases_rot = torch.exp(-1j * D_rot * _dt_for_step(dt_arr, i))
        tmp = torch.matmul(psi_slow, torch.conj(V_rot))
        tmp *= phases_rot[:, None, :]
        psi_slow = torch.matmul(tmp, torch.swapaxes(V_rot, -2, -1))
        psi = torch.matmul(psi_slow, V.T)
        if profile_this_step:
            update_end.record()
            profile_events.setdefault("state_update", []).append((update_start, update_end))

        history_cursor = _store_history(
            psi_history, history_indices, history_cursor, i + 1, psi
        )

        V_ref = evecs_tracked

    if profile_timings is not None:
        torch.cuda.synchronize(terms.Hff.device)
        profile_timings.clear()
        for name, pairs in profile_events.items():
            profile_timings[name] = [start.elapsed_time(end) / 1000.0 for start, end in pairs]

    probabilities_final = None
    if return_final_probabilities:
        amps = torch.matmul(psi, torch.conj(last_evecs))  # (B,S,n) @ (n,n)
        probabilities_final = torch.abs(amps) ** 2

    monitor_probabilities_final = None
    if monitor_idx is not None:
        amps = torch.matmul(psi, torch.conj(last_evecs))
        probs_all = torch.abs(amps) ** 2
        monitor_probabilities_final = probs_all[:, :, monitor_idx]

    return BatchedSimulationOutput(
        psi_final=psi,
        probabilities_final=probabilities_final,
        monitor_probabilities_final=monitor_probabilities_final,
        last_evecs=last_evecs,
        psi_history=psi_history,
        history_indices=(torch.tensor(history_indices, dtype=torch.int64, device=psi.device) if history_indices is not None else None),
    )


def _as_torch_complex(x: Any, *, dtype: Any, device: Any) -> Any:
    torch = get_torch()
    return torch.as_tensor(x, dtype=dtype, device=device)


def _batched_probabilities(psi: Any, evecs: Any) -> Any:
    """|<evec | psi>|^2 with batch support.

    psi: (B, S, n)
    evecs: (B, n, n) columns are eigenvectors
    returns: (B, S, n)
    """

    torch = get_torch()
    amps = torch.matmul(psi, torch.conj(evecs))  # (B,S,n) @ (B,n,n) -> (B,S,n)
    return torch.abs(amps) ** 2


def simulate_batched(
    *,
    terms: BatchedSlowHamiltonianTerms,
    E_t: Any,
    B_t: Any,
    psi0: Any,
    dt: float | Any,
    H_mu_t: Optional[Any] = None,
    D_mu: Optional[Any] = None,
    E0: Optional[Any] = None,
    B0: Optional[Any] = None,
    monitor_state_vectors: Optional[Any] = None,
    return_final_probabilities: bool = True,
    store_every: Optional[int] = None,
) -> BatchedSimulationOutput:
    """Batched GPU time evolution.

    This is a minimal, functional GPU mirror of `state_prep.simulator.Simulator`.

    Inputs are *precomputed arrays* to keep the GPU path disentangled from the
    original code's nested OOP structure.

    Parameters
    ----------
    terms:
        Slow Hamiltonian term matrices (PyTorch tensors).
    E_t, B_t:
        Electric/magnetic field vectors with shape (T, B, 3).
    psi0:
        Initial state vectors with shape (B, S, n) in the QN basis.
    dt:
        Scalar, shared array (T-1,), or per-point array (T-1,B).
    E0, B0:
        Optional true initial field vectors, each shape (3,) or (B,3). If
        omitted, the first E_t/B_t samples are used for initialization.
    H_mu_t:
        Optional microwave Hamiltonian in the QN basis with shape (T-1, B, n, n).
        If omitted, evolves only under H_slow.
    D_mu:
        Optional detuning/rotating-frame diagonal shift.
        Shape can be (n, n) or (B, n, n). Only used if H_mu_t is provided.
    monitor_state_vectors:
        Optional monitor vectors in the QN basis. Shape (M, n) or (B, M, n).
        Populations are returned in the *adiabatically tracked eigenbasis* (same
        semantics as CPU monitor states), mapped at t=0 and tracked via reordering.
    return_final_probabilities:
        If True, returns probabilities in the final tracked eigenbasis.
    store_every:
        Optional stride for retaining states, including the initial and final
        states. None keeps final-only storage.

    Returns
    -------
    BatchedSimulationOutput
    """

    torch = get_torch()

    E_t = torch.as_tensor(E_t, device=terms.Hff.device)
    B_t = torch.as_tensor(B_t, device=terms.Hff.device)
    psi = _as_torch_complex(psi0, dtype=terms.Hff.dtype, device=terms.Hff.device)

    if E_t.ndim != 3 or B_t.ndim != 3 or E_t.shape[-1] != 3 or B_t.shape[-1] != 3:
        raise ValueError("E_t and B_t must have shape (T, B, 3)")

    T, batch, _ = E_t.shape
    if B_t.shape[0] != T or B_t.shape[1] != batch:
        raise ValueError("B_t must have the same (T,B,3) shape as E_t")

    if psi.ndim != 3 or psi.shape[0] != batch:
        raise ValueError("psi0 must have shape (B, S, n)")

    n = terms.n
    if psi.shape[-1] != n:
        raise ValueError(f"psi0 last dimension must be n={n}")

    store_every = _validate_store_every(store_every)

    dt_arr = _normalize_dt(torch, dt, T, batch, psi.device)

    if H_mu_t is not None:
        H_mu_t_arr = _as_torch_complex(H_mu_t, dtype=terms.Hff.dtype, device=terms.Hff.device)
        if H_mu_t_arr.shape != (T - 1, batch, n, n):
            raise ValueError("H_mu_t must have shape (T-1, B, n, n)")

        if D_mu is None:
            raise ValueError("D_mu must be provided when H_mu_t is provided")

        D_mu_arr = _as_torch_complex(D_mu, dtype=terms.Hff.dtype, device=terms.Hff.device)
        if D_mu_arr.shape == (n, n):
            D_mu_arr = torch.broadcast_to(D_mu_arr, (batch, n, n))
        elif D_mu_arr.shape != (batch, n, n):
            raise ValueError("D_mu must have shape (n,n) or (B,n,n)")
        D_mu = D_mu_arr
        H_mu_t = H_mu_t_arr

    # Initial slow Hamiltonian and reference eigenvectors (for adiabatic tracking)
    E_initial = _initial_field_batch(torch, E0, E_t[0], batch)
    B_initial = _initial_field_batch(torch, B0, B_t[0], batch)
    H0 = slow_hamiltonian_batch(terms, E_initial, B_initial)  # (B,n,n)
    D0, V0 = torch.linalg.eigh(H0)
    V_ref = V0
    V_ref_ini = V0

    # Monitor index mapping at t=0 (closest eigenstate)
    monitor_idx = None
    if monitor_state_vectors is not None:
        mv = _as_torch_complex(monitor_state_vectors, dtype=V_ref_ini.dtype, device=psi.device)
        if mv.ndim == 2:
            mv = torch.broadcast_to(mv[None, :, :], (batch, mv.shape[0], mv.shape[1]))
        if mv.shape[0] != batch or mv.shape[-1] != n:
            raise ValueError("monitor_state_vectors must have shape (M,n) or (B,M,n)")

        # overlap[b, i, m] = | <v_i | m> |
        overlap = torch.abs(torch.matmul(torch.conj(torch.swapaxes(V_ref_ini, -2, -1)), torch.swapaxes(mv, -2, -1)))
        # overlap shape: (B, n, M) -> take argmax over i
        monitor_idx = torch.argmax(overlap, dim=1).to(torch.int64)

    psi_history, history_indices, history_cursor = _new_history(
        torch, psi, T, store_every
    )

    # Evolve
    last_evecs = V_ref
    for i in range(T - 1):
        H_slow = slow_hamiltonian_batch(terms, E_t[i], B_t[i])  # (B,n,n)
        D, V = torch.linalg.eigh(H_slow)

        # Track eigenvectors (for reporting/probabilities/monitor states)
        _, evecs_tracked = reorder_evecs_batched(V, D, V_ref)
        last_evecs = evecs_tracked

        if H_mu_t is None:
            phases = torch.exp(-1j * D * _dt_for_step(dt_arr, i))  # (B,n)
            tmp = torch.matmul(psi, torch.conj(V))
            tmp *= phases[:, None, :]
            psi = torch.matmul(tmp, torch.swapaxes(V, -2, -1))
        else:
            H_mu = H_mu_t[i]
            H_rot = torch.matmul(torch.conj(torch.swapaxes(V, -2, -1)), torch.matmul(H_mu, V))
            H_rot = H_rot + D_mu
            H_rot = H_rot.clone()
            diag_idx = torch.arange(n, device=psi.device)
            H_rot[:, diag_idx, diag_idx] += D

            D_rot, V_rot = torch.linalg.eigh(H_rot)
            A = torch.matmul(V, V_rot)

            phases_rot = torch.exp(-1j * D_rot * _dt_for_step(dt_arr, i))
            tmp = torch.matmul(psi, torch.conj(A))
            tmp *= phases_rot[:, None, :]
            psi = torch.matmul(tmp, torch.swapaxes(A, -2, -1))

        history_cursor = _store_history(
            psi_history, history_indices, history_cursor, i + 1, psi
        )

        V_ref = evecs_tracked

    probabilities_final = None
    if return_final_probabilities:
        probabilities_final = _batched_probabilities(psi, last_evecs)

    monitor_probabilities_final = None
    if monitor_idx is not None:
        # Compute amplitudes onto tracked eigenvectors, then pick monitor indices
        probs_all = _batched_probabilities(psi, last_evecs)  # (B,S,n)
        monitor_probabilities_final = torch.gather(probs_all, -1, monitor_idx[:, None, :].expand(-1, probs_all.shape[1], -1))

    return BatchedSimulationOutput(
        psi_final=psi,
        probabilities_final=probabilities_final,
        monitor_probabilities_final=monitor_probabilities_final,
        last_evecs=last_evecs,
        psi_history=psi_history,
        history_indices=(torch.tensor(history_indices, dtype=torch.int64, device=psi.device) if history_indices is not None else None),
    )
