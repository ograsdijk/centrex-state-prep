from __future__ import annotations

import numpy as np


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch not installed; skipping tensor smoke test.")
        return

    from state_prep_gpu.slow_hamiltonian import BatchedSlowHamiltonianTerms
    from state_prep_gpu.simulator import simulate_batched

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    T = 6
    B = 3
    S = 2
    n = 8

    def hermitian() -> np.ndarray:
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return (a + a.conj().T) / 2

    terms = BatchedSlowHamiltonianTerms(
        Hff=torch.as_tensor(hermitian(), dtype=torch.complex128, device=device),
        HSx=torch.as_tensor(hermitian(), dtype=torch.complex128, device=device),
        HSy=torch.as_tensor(hermitian(), dtype=torch.complex128, device=device),
        HSz=torch.as_tensor(hermitian(), dtype=torch.complex128, device=device),
        HZx=torch.as_tensor(hermitian(), dtype=torch.complex128, device=device),
        HZy=torch.as_tensor(hermitian(), dtype=torch.complex128, device=device),
        HZz=torch.as_tensor(hermitian(), dtype=torch.complex128, device=device),
    )

    E_t = rng.normal(size=(T, B, 3))
    B_t = rng.normal(size=(T, B, 3))

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
    print("ok", tuple(probs.shape), float(probs[0, 0].sum().item()))


if __name__ == "__main__":
    main()
