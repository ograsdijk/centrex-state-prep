from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from ._cupy import cupy
from ._reorder import reorder_evecs_batched
from .slow_hamiltonian import BatchedSlowHamiltonianTerms, slow_hamiltonian_batch


@dataclass(frozen=True)
class BatchedSimulationOutput:
    psi_final: Any
    probabilities_final: Optional[Any]
    monitor_probabilities_final: Optional[Any]
    last_evecs: Any


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
        H_mu_t = _as_cupy_complex(H_mu_t, dtype=terms.Hff.dtype)
        if H_mu_t.shape != (T - 1, batch, n, n):
            raise ValueError("H_mu_t must have shape (T-1, B, n, n)")

        if D_mu is None:
            raise ValueError("D_mu must be provided when H_mu_t is provided")

        D_mu = _as_cupy_complex(D_mu, dtype=terms.Hff.dtype)
        if D_mu.shape == (n, n):
            D_mu = cp.broadcast_to(D_mu, (batch, n, n))
        elif D_mu.shape != (batch, n, n):
            raise ValueError("D_mu must have shape (n,n) or (B,n,n)")

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
