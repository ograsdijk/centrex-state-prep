from __future__ import annotations

import numpy as np


def _reorder_evecs_batched_np(V_in: np.ndarray, E_in: np.ndarray, V_ref: np.ndarray):
    # overlap[b, i, j] = | <v_in_i | v_ref_j> |
    overlap = np.abs(np.conj(np.swapaxes(V_in, -2, -1)) @ V_ref)
    best_j = np.argmax(overlap, axis=-1)  # (B, n)
    order = np.argsort(best_j, axis=-1)  # (B, n)

    V_out = np.take_along_axis(V_in, order[..., None, :], axis=-1)
    E_out = np.take_along_axis(E_in, order, axis=-1)
    return E_out, V_out


def _slow_hamiltonian_batch_np(
    terms: dict[str, np.ndarray], E: np.ndarray, B: np.ndarray
) -> np.ndarray:
    # terms are (n,n); E,B are (B,3)
    Ex = E[:, 0][:, None, None]
    Ey = E[:, 1][:, None, None]
    Ez = E[:, 2][:, None, None]
    Bx = B[:, 0][:, None, None]
    By = B[:, 1][:, None, None]
    Bz = B[:, 2][:, None, None]

    H = (
        terms["Hff"][None]
        + Ex * terms["HSx"][None]
        + Ey * terms["HSy"][None]
        + Ez * terms["HSz"][None]
        + Bx * terms["HZx"][None]
        + By * terms["HZy"][None]
        + Bz * terms["HZz"][None]
    )
    return (2 * np.pi) * H


def _simulate_batched_cpu(
    *,
    terms: dict[str, np.ndarray],
    E_t: np.ndarray,
    B_t: np.ndarray,
    psi0: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    # E_t,B_t: (T,B,3); psi0: (B,S,n)
    T, batch, _ = E_t.shape
    psi = psi0.astype(np.complex128, copy=True)

    H0 = _slow_hamiltonian_batch_np(terms, E_t[0], B_t[0])
    D0, V0 = np.linalg.eigh(H0)
    V_ref = V0

    last_evecs = V_ref
    for i in range(T - 1):
        H = _slow_hamiltonian_batch_np(terms, E_t[i], B_t[i])
        D, V = np.linalg.eigh(H)

        _, evecs_tracked = _reorder_evecs_batched_np(V, D, V_ref)
        last_evecs = evecs_tracked

        phases = np.exp(-1j * D * dt)  # (B,n)
        tmp = psi @ np.conj(V)  # (B,S,n)
        tmp *= phases[:, None, :]
        psi = tmp @ np.swapaxes(V, -2, -1)

        V_ref = evecs_tracked

    amps = psi @ np.conj(last_evecs)
    probs_tracked = np.abs(amps) ** 2
    return psi, probs_tracked


def main() -> None:
    from state_prep_gpu._cupy import cupy

    cp = cupy()

    from state_prep_gpu.slow_hamiltonian import BatchedSlowHamiltonianTerms
    from state_prep_gpu.simulator import simulate_batched

    rng = np.random.default_rng(0)

    T = 6
    B = 3
    S = 2
    n = 8
    dt = 1e-7

    def hermitian() -> np.ndarray:
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return (a + a.conj().T) / 2

    terms_np = {
        "Hff": hermitian(),
        "HSx": hermitian(),
        "HSy": hermitian(),
        "HSz": hermitian(),
        "HZx": hermitian(),
        "HZy": hermitian(),
        "HZz": hermitian(),
    }

    terms_gpu = BatchedSlowHamiltonianTerms(
        **{k: cp.asarray(v, dtype=cp.complex128) for k, v in terms_np.items()}
    )

    E_t = rng.normal(size=(T, B, 3))
    B_t = rng.normal(size=(T, B, 3))

    psi0 = rng.normal(size=(B, S, n)) + 1j * rng.normal(size=(B, S, n))
    psi0 = psi0 / np.linalg.norm(psi0, axis=-1, keepdims=True)

    out_gpu = simulate_batched(
        terms=terms_gpu,
        E_t=cp.asarray(E_t, dtype=cp.float64),
        B_t=cp.asarray(B_t, dtype=cp.float64),
        psi0=psi0,
        dt=dt,
        H_mu_t=None,
        D_mu=None,
        return_final_probabilities=True,
    )
    probs_gpu = out_gpu.probabilities_final
    assert probs_gpu is not None
    probs_gpu_np = cp.asnumpy(probs_gpu)

    psi_cpu, probs_cpu_tracked = _simulate_batched_cpu(
        terms=terms_np, E_t=E_t, B_t=B_t, psi0=psi0, dt=dt
    )

    psi_gpu_np = cp.asnumpy(out_gpu.psi_final)
    last_evecs_gpu_np = cp.asnumpy(out_gpu.last_evecs)

    # Compare final states in QN basis (phase-sensitive) and QN populations (phase-insensitive).
    max_abs_psi = float(np.max(np.abs(psi_gpu_np - psi_cpu)))
    max_abs_pop_qn = float(
        np.max(np.abs(np.abs(psi_gpu_np) ** 2 - np.abs(psi_cpu) ** 2))
    )

    # Compare probabilities in the *same* (GPU) tracked eigenbasis.
    probs_cpu_in_gpu_basis = np.abs(psi_cpu @ np.conj(last_evecs_gpu_np)) ** 2
    max_abs_prob_same_basis = float(
        np.max(np.abs(probs_gpu_np - probs_cpu_in_gpu_basis))
    )

    print("max_abs_psi_diff:", max_abs_psi)
    print("max_abs_qn_pop_diff:", max_abs_pop_qn)
    print("max_abs_prob_diff_same_basis:", max_abs_prob_same_basis)
    print("gpu_prob_sum[0,0]:", float(np.sum(probs_gpu_np[0, 0])))

    # Sanity: probabilities are normalized.
    assert abs(float(np.sum(probs_gpu_np[0, 0])) - 1.0) < 1e-9


if __name__ == "__main__":
    main()
