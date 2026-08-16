from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ._cupy import cupy
from ._reorder import reorder_evecs_batched
from .slow_hamiltonian import BatchedSlowHamiltonianTerms, slow_hamiltonian_batch


@dataclass(frozen=True)
class BatchedSimulationOutput:
    psi_final: Any
    probabilities_final: Optional[Any]
    monitor_probabilities_final: Optional[Any]
    last_evecs: Any


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

    cp = cupy()

    E_t = cp.asarray(E_t)
    B_t = cp.asarray(B_t)

    if E_t.ndim == 2:
        # (T,3) -> (T,B,3)
        if E_t.shape[-1] != 3:
            raise ValueError("E_t must have shape (T,3) or (T,B,3)")
        if B_t.ndim != 2 or B_t.shape != E_t.shape:
            raise ValueError("If E_t is (T,3), B_t must also be (T,3)")
        psi_probe = cp.asarray(psi0)
        if psi_probe.ndim != 3:
            raise ValueError("psi0 must have shape (B,S,n)")
        batch = int(psi_probe.shape[0])
        E_t = cp.broadcast_to(E_t[:, None, :], (E_t.shape[0], batch, 3))
        B_t = cp.broadcast_to(B_t[:, None, :], (B_t.shape[0], batch, 3))

    psi = _as_cupy_complex(psi0, dtype=terms.Hff.dtype)

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
        dt_arr = cp.full((T - 1,), float(dt), dtype=cp.float64)
    else:
        dt_arr = cp.asarray(dt, dtype=cp.float64)
        if dt_arr.shape != (T - 1,):
            raise ValueError("dt must be a scalar or have shape (T-1,)")

    Hu = _as_cupy_complex(Hu, dtype=terms.Hff.dtype)
    Hl = _as_cupy_complex(Hl, dtype=terms.Hff.dtype)
    if Hu.shape != Hl.shape or Hu.ndim != 4 or Hu.shape[1] != 3 or Hu.shape[2] != n or Hu.shape[3] != n:
        raise ValueError("Hu and Hl must both have shape (M,3,n,n)")
    M = int(Hu.shape[0])

    coeff_u_t = _as_cupy_complex(coeff_u_t, dtype=terms.Hff.dtype)
    coeff_l_t = _as_cupy_complex(coeff_l_t, dtype=terms.Hff.dtype)
    if coeff_u_t.shape != (T - 1, M, 3) or coeff_l_t.shape != (T - 1, M, 3):
        raise ValueError("coeff_u_t and coeff_l_t must have shape (T-1,M,3)")

    coupling_scales = cp.asarray(coupling_scales, dtype=cp.float64)
    if coupling_scales.shape != (batch, M):
        raise ValueError("coupling_scales must have shape (B,M)")

    D_mu_diag_batch = cp.asarray(D_mu_diag_batch)
    if D_mu_diag_batch.shape != (batch, n):
        raise ValueError("D_mu_diag_batch must have shape (B,n)")

    # Initial slow Hamiltonian and reference eigenvectors (for adiabatic tracking)
    H0 = slow_hamiltonian_batch(terms, E_t[0], B_t[0])  # (B,n,n)
    D0, V0 = cp.linalg.eigh(H0)
    V_ref = V0
    V_ref_ini = V0

    # Monitor index mapping at t=0 (closest eigenstate)
    monitor_idx = None
    if monitor_state_vectors is not None:
        mv = cp.asarray(monitor_state_vectors)
        if mv.ndim == 2:
            mv = cp.broadcast_to(mv[None, :, :], (batch, mv.shape[0], mv.shape[1]))
        if mv.shape[0] != batch or mv.shape[-1] != n:
            raise ValueError("monitor_state_vectors must have shape (M,n) or (B,M,n)")

        overlap = cp.abs(
            cp.matmul(
                cp.conj(cp.swapaxes(V_ref_ini, -2, -1)),
                cp.swapaxes(mv, -2, -1),
            )
        )
        monitor_idx = cp.argmax(overlap, axis=1).astype(cp.int64)

    diag_idx = cp.arange(n)

    last_evecs = V_ref
    for i in range(T - 1):
        H_slow = slow_hamiltonian_batch(terms, E_t[i], B_t[i])  # (B,n,n)
        D, V = cp.linalg.eigh(H_slow)

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
        H_mu = cp.tensordot(coupling_scales, H_fields, axes=(1, 0))

        # Rotating-frame Hamiltonian in slow-eigenbasis
        H_rot = cp.matmul(cp.conj(cp.swapaxes(V, -2, -1)), cp.matmul(H_mu, V))
        # Add diagonal shifts: slow energies + detuning shifts (CPU batched semantics)
        H_rot[:, diag_idx, diag_idx] += D
        H_rot[:, diag_idx, diag_idx] += D_mu_diag_batch

        D_rot, V_rot = cp.linalg.eigh(H_rot)
        A = cp.matmul(V, V_rot)

        phases_rot = cp.exp(-1j * D_rot * dt_arr[i])
        tmp = cp.matmul(psi, cp.conj(A))
        tmp *= phases_rot[:, None, :]
        psi = cp.matmul(tmp, cp.swapaxes(A, -2, -1))

        V_ref = evecs_tracked

    probabilities_final = None
    if return_final_probabilities:
        probabilities_final = _batched_probabilities(psi, last_evecs)

    monitor_probabilities_final = None
    if monitor_idx is not None:
        probs_all = _batched_probabilities(psi, last_evecs)  # (B,S,n)
        monitor_probabilities_final = cp.take_along_axis(
            probs_all,
            monitor_idx[:, None, :],
            axis=-1,
        )

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
    monitor_state_vectors: Optional[Any] = None,
    return_final_probabilities: bool = True,
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
    - amp_t: (T-1,M)
    - coupling_scales: (B,M)
    - D_mu_diag_batch: (B,n)
    """

    cp = cupy()

    E_t = cp.asarray(E_t)
    B_t = cp.asarray(B_t)

    if E_t.ndim == 2:
        if E_t.shape[-1] != 3:
            raise ValueError("E_t must have shape (T,3) or (T,B,3)")
        if B_t.ndim != 2 or B_t.shape != E_t.shape:
            raise ValueError("If E_t is (T,3), B_t must also be (T,3)")
        psi_probe = cp.asarray(psi0)
        if psi_probe.ndim != 3:
            raise ValueError("psi0 must have shape (B,S,n)")
        batch = int(psi_probe.shape[0])
        E_t = cp.broadcast_to(E_t[:, None, :], (E_t.shape[0], batch, 3))
        B_t = cp.broadcast_to(B_t[:, None, :], (B_t.shape[0], batch, 3))

    psi = _as_cupy_complex(psi0, dtype=terms.Hff.dtype)

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
        dt_arr = cp.full((T - 1,), float(dt), dtype=cp.float64)
    else:
        dt_arr = cp.asarray(dt, dtype=cp.float64)
        if dt_arr.shape != (T - 1,):
            raise ValueError("dt must be a scalar or have shape (T-1,)")

    H_mu_base_fields = _as_cupy_complex(H_mu_base_fields, dtype=terms.Hff.dtype)
    if H_mu_base_fields.ndim != 3 or H_mu_base_fields.shape[1] != n or H_mu_base_fields.shape[2] != n:
        raise ValueError("H_mu_base_fields must have shape (M,n,n)")
    M = int(H_mu_base_fields.shape[0])

    amp_t = _as_cupy_complex(amp_t, dtype=terms.Hff.dtype)
    if amp_t.shape != (T - 1, M):
        raise ValueError("amp_t must have shape (T-1,M)")

    coupling_scales = cp.asarray(coupling_scales, dtype=cp.float64)
    if coupling_scales.shape != (batch, M):
        raise ValueError("coupling_scales must have shape (B,M)")

    D_mu_diag_batch = cp.asarray(D_mu_diag_batch)
    if D_mu_diag_batch.shape != (batch, n):
        raise ValueError("D_mu_diag_batch must have shape (B,n)")

    # Initial slow Hamiltonian and reference eigenvectors (for adiabatic tracking)
    H0 = slow_hamiltonian_batch(terms, E_t[0], B_t[0])  # (B,n,n)
    _, V0 = cp.linalg.eigh(H0)
    V_ref = V0
    V_ref_ini = V0

    monitor_idx = None
    if monitor_state_vectors is not None:
        mv = cp.asarray(monitor_state_vectors)
        if mv.ndim == 2:
            mv = cp.broadcast_to(mv[None, :, :], (batch, mv.shape[0], mv.shape[1]))
        if mv.shape[0] != batch or mv.shape[-1] != n:
            raise ValueError("monitor_state_vectors must have shape (M,n) or (B,M,n)")

        overlap = cp.abs(
            cp.matmul(
                cp.conj(cp.swapaxes(V_ref_ini, -2, -1)),
                cp.swapaxes(mv, -2, -1),
            )
        )
        monitor_idx = cp.argmax(overlap, axis=1).astype(cp.int64)

    diag_idx = cp.arange(n)

    last_evecs = V_ref
    for i in range(T - 1):
        H_slow = slow_hamiltonian_batch(terms, E_t[i], B_t[i])  # (B,n,n)
        D, V = cp.linalg.eigh(H_slow)

        _, evecs_tracked = reorder_evecs_batched(V, D, V_ref)
        last_evecs = evecs_tracked

        # weights: (B,M) complex
        weights = coupling_scales * amp_t[i][None, :]
        weights = _as_cupy_complex(weights, dtype=terms.Hff.dtype)

        # Combine across fields per batch: (B,M) x (M,n,n) -> (B,n,n)
        H_mu = cp.tensordot(weights, H_mu_base_fields, axes=(1, 0))

        H_rot = cp.matmul(cp.conj(cp.swapaxes(V, -2, -1)), cp.matmul(H_mu, V))
        H_rot[:, diag_idx, diag_idx] += D
        H_rot[:, diag_idx, diag_idx] += D_mu_diag_batch

        D_rot, V_rot = cp.linalg.eigh(H_rot)
        A = cp.matmul(V, V_rot)

        phases_rot = cp.exp(-1j * D_rot * dt_arr[i])
        tmp = cp.matmul(psi, cp.conj(A))
        tmp *= phases_rot[:, None, :]
        psi = cp.matmul(tmp, cp.swapaxes(A, -2, -1))

        V_ref = evecs_tracked

    probabilities_final = None
    if return_final_probabilities:
        probabilities_final = _batched_probabilities(psi, last_evecs)

    monitor_probabilities_final = None
    if monitor_idx is not None:
        probs_all = _batched_probabilities(psi, last_evecs)
        monitor_probabilities_final = cp.take_along_axis(
            probs_all,
            monitor_idx[:, None, :],
            axis=-1,
        )

    return BatchedSimulationOutput(
        psi_final=psi,
        probabilities_final=probabilities_final,
        monitor_probabilities_final=monitor_probabilities_final,
        last_evecs=last_evecs,
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
    monitor_state_vectors: Optional[Any] = None,
    return_final_probabilities: bool = True,
) -> BatchedSimulationOutput:
    """GPU time evolution for microwave scans with a *shared* slow Hamiltonian.

    This matches the CPU batched-scan idea: H_slow(t) is identical for all scan
    points, so we diagonalize it only once per timestep.

    Expected shapes
    ---------------
    - E_t, B_t: (T,3) (shared across batch)
    - psi0: (B,S,n)
    - H_mu_base_fields: (M,n,n)
    - amp_t: (T-1,M)
    - coupling_scales: (B,M)
    - D_mu_diag_batch: (B,n)
    - monitor_state_vectors: (K,n) optional (shared)
    """

    cp = cupy()

    E_t = cp.asarray(E_t)
    B_t = cp.asarray(B_t)
    psi = _as_cupy_complex(psi0, dtype=terms.Hff.dtype)

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

    T = int(E_t.shape[0])
    if T < 2:
        raise ValueError("E_t/B_t must have T>=2")

    if isinstance(dt, (float, int)):
        dt_arr = cp.full((T - 1,), float(dt), dtype=cp.float64)
    else:
        dt_arr = cp.asarray(dt, dtype=cp.float64)
        if dt_arr.shape != (T - 1,):
            raise ValueError("dt must be a scalar or have shape (T-1,)")

    H_mu_base_fields = _as_cupy_complex(H_mu_base_fields, dtype=terms.Hff.dtype)
    if H_mu_base_fields.ndim != 3 or H_mu_base_fields.shape[1] != n or H_mu_base_fields.shape[2] != n:
        raise ValueError("H_mu_base_fields must have shape (M,n,n)")
    M = int(H_mu_base_fields.shape[0])

    amp_t = _as_cupy_complex(amp_t, dtype=terms.Hff.dtype)
    if amp_t.shape != (T - 1, M):
        raise ValueError("amp_t must have shape (T-1,M)")

    coupling_scales = cp.asarray(coupling_scales, dtype=cp.float64)
    if coupling_scales.shape != (batch, M):
        raise ValueError("coupling_scales must have shape (B,M)")

    D_mu_diag_batch = cp.asarray(D_mu_diag_batch)
    if D_mu_diag_batch.shape != (batch, n):
        raise ValueError("D_mu_diag_batch must have shape (B,n)")

    # Initial slow Hamiltonian and reference eigenvectors (for adiabatic tracking)
    H0 = slow_hamiltonian_batch(terms, E_t[0], B_t[0])  # (n,n)
    _, V0 = cp.linalg.eigh(H0)
    V_ref = V0
    V_ref_ini = V0

    monitor_idx = None
    if monitor_state_vectors is not None:
        mv = cp.asarray(monitor_state_vectors)
        if mv.ndim != 2 or mv.shape[-1] != n:
            raise ValueError("monitor_state_vectors must have shape (K,n)")
        # overlap[i,k] = | <v_i | m_k> |
        overlap = cp.abs(cp.matmul(cp.conj(V_ref_ini.T), mv.T))  # (n,K)
        monitor_idx = cp.argmax(overlap, axis=0).astype(cp.int64)  # (K,)

    diag_idx = cp.arange(n)
    V_b = None
    VH_b = None

    last_evecs = V_ref
    for i in range(T - 1):
        H_slow = slow_hamiltonian_batch(terms, E_t[i], B_t[i])  # (n,n)
        D, V = cp.linalg.eigh(H_slow)

        # Track eigenvectors for output/monitor semantics
        _, evecs_tracked_b = reorder_evecs_batched(V[None, :, :], D[None, :], V_ref[None, :, :])
        evecs_tracked = evecs_tracked_b[0]
        last_evecs = evecs_tracked

        # Broadcast V for batch matmuls
        V_b = V[None, :, :]
        VH_b = cp.conj(V.T)[None, :, :]

        # weights: (B,M) complex
        weights = coupling_scales * amp_t[i][None, :]
        weights = _as_cupy_complex(weights, dtype=terms.Hff.dtype)

        # H_mu: (B,n,n)
        H_mu = cp.tensordot(weights, H_mu_base_fields, axes=(1, 0))

        # H_rot: (B,n,n)
        H_rot = cp.matmul(VH_b, cp.matmul(H_mu, V_b))
        H_rot[:, diag_idx, diag_idx] += D
        H_rot[:, diag_idx, diag_idx] += D_mu_diag_batch

        D_rot, V_rot = cp.linalg.eigh(H_rot)
        A = cp.matmul(V_b, V_rot)  # (B,n,n)

        phases_rot = cp.exp(-1j * D_rot * dt_arr[i])
        tmp = cp.matmul(psi, cp.conj(A))
        tmp *= phases_rot[:, None, :]
        psi = cp.matmul(tmp, cp.swapaxes(A, -2, -1))

        V_ref = evecs_tracked

    probabilities_final = None
    if return_final_probabilities:
        amps = cp.matmul(psi, cp.conj(last_evecs))  # (B,S,n) @ (n,n)
        probabilities_final = cp.abs(amps) ** 2

    monitor_probabilities_final = None
    if monitor_idx is not None:
        amps = cp.matmul(psi, cp.conj(last_evecs))
        probs_all = cp.abs(amps) ** 2
        monitor_probabilities_final = probs_all[:, :, monitor_idx]

    return BatchedSimulationOutput(
        psi_final=psi,
        probabilities_final=probabilities_final,
        monitor_probabilities_final=monitor_probabilities_final,
        last_evecs=last_evecs,
    )


def _as_cupy_complex(x: Any, *, dtype: Any) -> Any:
    cp = cupy()
    arr = cp.asarray(x)
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _batched_probabilities(psi: Any, evecs: Any) -> Any:
    """|<evec | psi>|^2 with batch support.

    psi: (B, S, n)
    evecs: (B, n, n) columns are eigenvectors
    returns: (B, S, n)
    """

    cp = cupy()
    amps = cp.matmul(psi, cp.conj(evecs))  # (B,S,n) @ (B,n,n) -> (B,S,n)
    return cp.abs(amps) ** 2


def simulate_batched(
    *,
    terms: BatchedSlowHamiltonianTerms,
    E_t: Any,
    B_t: Any,
    psi0: Any,
    dt: float | Any,
    H_mu_t: Optional[Any] = None,
    D_mu: Optional[Any] = None,
    monitor_state_vectors: Optional[Any] = None,
    return_final_probabilities: bool = True,
) -> BatchedSimulationOutput:
    """Batched GPU time evolution.

    This is a minimal, functional GPU mirror of `state_prep.simulator.Simulator`.

    Inputs are *precomputed arrays* to keep the GPU path disentangled from the
    original code's nested OOP structure.

    Parameters
    ----------
    terms:
        Slow Hamiltonian term matrices (CuPy arrays).
    E_t, B_t:
        Electric/magnetic field vectors with shape (T, B, 3).
    psi0:
        Initial state vectors with shape (B, S, n) in the QN basis.
    dt:
        Either a scalar time step, or array with shape (T-1,).
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

    Returns
    -------
    BatchedSimulationOutput
    """

    cp = cupy()

    E_t = cp.asarray(E_t)
    B_t = cp.asarray(B_t)
    psi = _as_cupy_complex(psi0, dtype=terms.Hff.dtype)

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
        dt_arr = cp.full((T - 1,), float(dt), dtype=cp.float64)
    else:
        dt_arr = cp.asarray(dt, dtype=cp.float64)
        if dt_arr.shape != (T - 1,):
            raise ValueError("dt must be a scalar or have shape (T-1,)")

    if H_mu_t is not None:
        H_mu_t_arr = _as_cupy_complex(H_mu_t, dtype=terms.Hff.dtype)
        if H_mu_t_arr.shape != (T - 1, batch, n, n):
            raise ValueError("H_mu_t must have shape (T-1, B, n, n)")

        if D_mu is None:
            raise ValueError("D_mu must be provided when H_mu_t is provided")

        D_mu_arr = _as_cupy_complex(D_mu, dtype=terms.Hff.dtype)
        if D_mu_arr.shape == (n, n):
            D_mu_arr = cp.broadcast_to(D_mu_arr, (batch, n, n))
        elif D_mu_arr.shape != (batch, n, n):
            raise ValueError("D_mu must have shape (n,n) or (B,n,n)")
        D_mu = D_mu_arr
        H_mu_t = H_mu_t_arr

    # Initial slow Hamiltonian and reference eigenvectors (for adiabatic tracking)
    H0 = slow_hamiltonian_batch(terms, E_t[0], B_t[0])  # (B,n,n)
    D0, V0 = cp.linalg.eigh(H0)
    V_ref = V0
    V_ref_ini = V0

    # Monitor index mapping at t=0 (closest eigenstate)
    monitor_idx = None
    if monitor_state_vectors is not None:
        mv = cp.asarray(monitor_state_vectors)
        if mv.ndim == 2:
            mv = cp.broadcast_to(mv[None, :, :], (batch, mv.shape[0], mv.shape[1]))
        if mv.shape[0] != batch or mv.shape[-1] != n:
            raise ValueError("monitor_state_vectors must have shape (M,n) or (B,M,n)")

        # overlap[b, i, m] = | <v_i | m> |
        overlap = cp.abs(cp.matmul(cp.conj(cp.swapaxes(V_ref_ini, -2, -1)), cp.swapaxes(mv, -2, -1)))
        # overlap shape: (B, n, M) -> take argmax over i
        monitor_idx = cp.argmax(overlap, axis=1).astype(cp.int64)

    # Evolve
    last_evecs = V_ref
    for i in range(T - 1):
        H_slow = slow_hamiltonian_batch(terms, E_t[i], B_t[i])  # (B,n,n)
        D, V = cp.linalg.eigh(H_slow)

        # Track eigenvectors (for reporting/probabilities/monitor states)
        _, evecs_tracked = reorder_evecs_batched(V, D, V_ref)
        last_evecs = evecs_tracked

        if H_mu_t is None:
            phases = cp.exp(-1j * D * dt_arr[i])  # (B,n)
            tmp = cp.matmul(psi, cp.conj(V))
            tmp *= phases[:, None, :]
            psi = cp.matmul(tmp, cp.swapaxes(V, -2, -1))
        else:
            H_mu = H_mu_t[i]
            H_rot = cp.matmul(cp.conj(cp.swapaxes(V, -2, -1)), cp.matmul(H_mu, V))
            H_rot = H_rot + D_mu
            H_rot = H_rot.copy()
            H_rot[:, cp.arange(n), cp.arange(n)] += D

            D_rot, V_rot = cp.linalg.eigh(H_rot)
            A = cp.matmul(V, V_rot)

            phases_rot = cp.exp(-1j * D_rot * dt_arr[i])
            tmp = cp.matmul(psi, cp.conj(A))
            tmp *= phases_rot[:, None, :]
            psi = cp.matmul(tmp, cp.swapaxes(A, -2, -1))

        V_ref = evecs_tracked

    probabilities_final = None
    if return_final_probabilities:
        probabilities_final = _batched_probabilities(psi, last_evecs)

    monitor_probabilities_final = None
    if monitor_idx is not None:
        # Compute amplitudes onto tracked eigenvectors, then pick monitor indices
        probs_all = _batched_probabilities(psi, last_evecs)  # (B,S,n)
        monitor_probabilities_final = cp.take_along_axis(
            probs_all,
            monitor_idx[:, None, :],
            axis=-1,
        )

    return BatchedSimulationOutput(
        psi_final=psi,
        probabilities_final=probabilities_final,
        monitor_probabilities_final=monitor_probabilities_final,
        last_evecs=last_evecs,
    )
