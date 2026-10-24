from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


def _reorder_evecs_batched_np(V_in: np.ndarray, E_in: np.ndarray, V_ref: np.ndarray):
    from state_prep_gpu.example_compare_cpu import _reorder_evecs_batched_np as reorder

    E_out, V_out = reorder(V_in, E_in, V_ref)
    return E_out, V_out


def _slow_hamiltonian_batch_np(terms: dict[str, np.ndarray], E: np.ndarray, B: np.ndarray) -> np.ndarray:
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
) -> np.ndarray:
    T, batch, _ = E_t.shape
    psi = psi0.astype(np.complex128, copy=True)

    H0 = _slow_hamiltonian_batch_np(terms, E_t[0], B_t[0])
    _, V0 = np.linalg.eigh(H0)
    V_ref = V0

    for i in range(T - 1):
        H = _slow_hamiltonian_batch_np(terms, E_t[i], B_t[i])
        D, V = np.linalg.eigh(H)

        _, evecs_tracked = _reorder_evecs_batched_np(V, D, V_ref)

        phases = np.exp(-1j * D * dt)
        tmp = psi @ np.conj(V)
        tmp *= phases[:, None, :]
        psi = tmp @ np.swapaxes(V, -2, -1)

        V_ref = evecs_tracked

    return psi


@dataclass(frozen=True)
class BenchConfig:
    T: int = 200
    B: int = 64
    S: int = 1
    n: int = 32
    dt: float = 1e-7
    warmup: int = 1
    repeat: int = 3


def _make_problem(cfg: BenchConfig, seed: int = 0):
    rng = np.random.default_rng(seed)

    def hermitian(n: int) -> np.ndarray:
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return (a + a.conj().T) / 2

    terms_np = {
        "Hff": hermitian(cfg.n),
        "HSx": hermitian(cfg.n),
        "HSy": hermitian(cfg.n),
        "HSz": hermitian(cfg.n),
        "HZx": hermitian(cfg.n),
        "HZy": hermitian(cfg.n),
        "HZz": hermitian(cfg.n),
    }

    E_t = rng.normal(size=(cfg.T, cfg.B, 3))
    B_t = rng.normal(size=(cfg.T, cfg.B, 3))

    psi0 = rng.normal(size=(cfg.B, cfg.S, cfg.n)) + 1j * rng.normal(size=(cfg.B, cfg.S, cfg.n))
    psi0 = psi0 / np.linalg.norm(psi0, axis=-1, keepdims=True)

    return terms_np, E_t, B_t, psi0


def _time_cpu(cfg: BenchConfig, terms_np, E_t, B_t, psi0) -> float:
    # Warmup
    for _ in range(cfg.warmup):
        _simulate_batched_cpu(terms=terms_np, E_t=E_t, B_t=B_t, psi0=psi0, dt=cfg.dt)

    t_best = float("inf")
    for _ in range(cfg.repeat):
        t0 = time.perf_counter()
        _simulate_batched_cpu(terms=terms_np, E_t=E_t, B_t=B_t, psi0=psi0, dt=cfg.dt)
        t_best = min(t_best, time.perf_counter() - t0)
    return t_best


def _time_torch(cfg: BenchConfig, terms_np, E_t, B_t, psi0) -> float:
    import torch

    from state_prep_gpu.slow_hamiltonian import BatchedSlowHamiltonianTerms
    from state_prep_gpu.simulator import simulate_batched

    terms_gpu = BatchedSlowHamiltonianTerms(
        **{k: torch.as_tensor(v, dtype=torch.complex128, device="cuda") for k, v in terms_np.items()}
    )

    E_t_gpu = torch.as_tensor(E_t, dtype=torch.float64, device="cuda")
    B_t_gpu = torch.as_tensor(B_t, dtype=torch.float64, device="cuda")
    psi0_gpu = torch.as_tensor(psi0, dtype=torch.complex128, device="cuda")

    # Warmup
    for _ in range(cfg.warmup):
        out = simulate_batched(
            terms=terms_gpu,
            E_t=E_t_gpu,
            B_t=B_t_gpu,
            psi0=psi0_gpu,
            dt=cfg.dt,
            H_mu_t=None,
            D_mu=None,
            return_final_probabilities=False,
        )
        torch.cuda.synchronize()
        _ = out.psi_final

    t_best = float("inf")
    for _ in range(cfg.repeat):
        t0 = time.perf_counter()
        out = simulate_batched(
            terms=terms_gpu,
            E_t=E_t_gpu,
            B_t=B_t_gpu,
            psi0=psi0_gpu,
            dt=cfg.dt,
            H_mu_t=None,
            D_mu=None,
            return_final_probabilities=False,
        )
        torch.cuda.synchronize()
        _ = out.psi_final
        t_best = min(t_best, time.perf_counter() - t0)

    return t_best


def main() -> None:
    try:
        import torch

        have_torch_cuda = torch.cuda.is_available()
    except ModuleNotFoundError:
        have_torch_cuda = False

    cfg = BenchConfig()
    terms_np, E_t, B_t, psi0 = _make_problem(cfg)

    print(f"Config: T={cfg.T} B={cfg.B} S={cfg.S} n={cfg.n} dt={cfg.dt}")
    print(f"Warmup={cfg.warmup} repeat={cfg.repeat}")

    t_cpu = _time_cpu(cfg, terms_np, E_t, B_t, psi0)
    print(f"CPU (NumPy) best: {t_cpu:.3f} s")

    if not have_torch_cuda:
        print("CUDA-enabled PyTorch not available; skipping GPU benchmark.")
        return

    t_gpu = _time_torch(cfg, terms_np, E_t, B_t, psi0)
    print(f"GPU (PyTorch) best: {t_gpu:.3f} s")

    if t_gpu > 0:
        print(f"Speedup: {t_cpu / t_gpu:.2f}x")


if __name__ == "__main__":
    main()
