from __future__ import annotations

import numpy as np


def main() -> None:
    try:
        import cupy as cp  # type: ignore
    except ModuleNotFoundError:
        print("CuPy not installed; skipping GPU smoke test.")
        return

    from state_prep_gpu.slow_hamiltonian import BatchedSlowHamiltonianTerms
    from state_prep_gpu.simulator import simulate_batched

    rng = np.random.default_rng(0)

    T = 6
    B = 3
    S = 2
    n = 8

    def hermitian() -> np.ndarray:
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return (a + a.conj().T) / 2

    terms = BatchedSlowHamiltonianTerms(
        Hff=cp.asarray(hermitian(), dtype=cp.complex128),
        HSx=cp.asarray(hermitian(), dtype=cp.complex128),
        HSy=cp.asarray(hermitian(), dtype=cp.complex128),
        HSz=cp.asarray(hermitian(), dtype=cp.complex128),
        HZx=cp.asarray(hermitian(), dtype=cp.complex128),
        HZy=cp.asarray(hermitian(), dtype=cp.complex128),
        HZz=cp.asarray(hermitian(), dtype=cp.complex128),
    )

    E_t = cp.asarray(rng.normal(size=(T, B, 3)), dtype=cp.float64)
    B_t = cp.asarray(rng.normal(size=(T, B, 3)), dtype=cp.float64)

    psi0 = rng.normal(size=(B, S, n)) + 1j * rng.normal(size=(B, S, n))
    psi0 = psi0 / np.linalg.norm(psi0, axis=-1, keepdims=True)

    out = simulate_batched(
        terms=terms,
        E_t=E_t,
        B_t=B_t,
        psi0=psi0,
        dt=1e-7,
        H_mu_t=None,
        D_mu=None,
        return_final_probabilities=True,
    )

    probs = out.probabilities_final
    assert probs is not None
    print("ok", probs.shape, float(cp.sum(probs[0, 0]).get()))


if __name__ == "__main__":
    main()
