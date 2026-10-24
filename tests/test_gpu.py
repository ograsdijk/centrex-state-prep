"""PyTorch tensor backend checks; CUDA is used automatically when available."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is an optional GPU dependency")

from conftest import CROSS_BACKEND_TOL  # noqa: E402

# Exercise CPU fallback even on CUDA machines. Single-precision eigensolves
# accumulate more rounding error than the complex128 paths checked below.
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
SINGLE_PRECISION_TOL = 3e-6


@pytest.mark.parametrize("device_name", DEVICES)
def test_reorder_finds_optimal_assignment_for_equal_overlap_matches(device_name):
    """Ties must preserve the optimal objective and align complex phases."""
    from state_prep_gpu._reorder import reorder_evecs_batched

    n = 32
    pair = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
    vectors = torch.block_diag(*([pair] * (n // 2)))
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(8))
    vectors = vectors[:, permutation].to(device_name)
    values = torch.arange(n, dtype=torch.float64, device=device_name)
    reference = torch.eye(n, dtype=torch.complex128, device=device_name)
    actual_values, actual_vectors = reorder_evecs_batched(
        vectors[None], values[None], reference[None]
    )

    overlaps = reference.mH @ actual_vectors[0]
    assert torch.allclose(overlaps.diagonal().real, torch.full((n,), 1 / np.sqrt(2), dtype=torch.float64, device=device_name))
    assert overlaps.diagonal().imag.abs().max() < 1e-14
    assert torch.equal(actual_values[0].sort().values, values)


def test_slow_hamiltonian_matches_cpu(spa2_setup):
    """Batched tensor slow-Hamiltonian construction must match the CPU one."""
    import centrex_tlf

    from state_prep_gpu import slow_hamiltonian_batch, terms_from_centrex_tlf

    hamiltonian = spa2_setup["hamiltonian"]
    uncoupled = centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(
        list(hamiltonian.QN)
    )
    terms = terms_from_centrex_tlf(uncoupled)

    rng = np.random.default_rng(0)
    E = rng.normal(size=(3, 2, 3)) * 1e3
    B = rng.normal(size=(3, 2, 3)) * 1e-4
    actual = slow_hamiltonian_batch(terms, E, B).detach().cpu().numpy()

    for t in range(E.shape[0]):
        for b in range(E.shape[1]):
            expected = hamiltonian.H_EB(E[t, b], B[t, b])
            assert np.max(np.abs(actual[t, b] - expected)) < CROSS_BACKEND_TOL


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_torch_simulation_probabilities_normalise_and_store_history(device_name, dtype):
    """The public tensor simulator preserves norm and history indexing."""
    from state_prep_gpu import BatchedSlowHamiltonianTerms, simulate_batched

    device = torch.device(device_name)
    tolerance = SINGLE_PRECISION_TOL if dtype == torch.complex64 else 1e-12
    n, batch, states, steps = 4, 3, 2, 5
    zero = torch.zeros((n, n), dtype=dtype, device=device)
    terms = BatchedSlowHamiltonianTerms(*(zero.clone() for _ in range(7)))
    rng = np.random.default_rng(4)
    psi0 = rng.normal(size=(batch, states, n)) + 1j * rng.normal(
        size=(batch, states, n)
    )
    psi0 /= np.linalg.norm(psi0, axis=-1, keepdims=True)
    fields = np.zeros((steps, batch, 3), dtype=np.float64)

    result = simulate_batched(
        terms=terms,
        E_t=fields,
        B_t=fields,
        psi0=psi0,
        dt=1e-7,
        monitor_state_vectors=np.eye(n)[:2],
        store_every=2,
    )

    assert isinstance(result.psi_final, torch.Tensor)
    assert result.psi_final.device == device
    assert result.psi_final.dtype == dtype
    assert result.psi_history.dtype == dtype
    assert result.psi_history.shape == (3, batch, states, n)
    assert result.history_indices.tolist() == [0, 2, 4]
    assert torch.allclose(
        result.probabilities_final.sum(dim=-1),
        torch.ones((batch, states), dtype=zero.real.dtype, device=device),
        atol=tolerance,
        rtol=0,
    )
    assert result.monitor_probabilities_final.shape == (batch, states, 2)
    assert torch.allclose(
        result.monitor_probabilities_final,
        result.probabilities_final[..., :2],
        atol=tolerance,
        rtol=0,
    )


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("monitor_dtype", [np.float64, np.complex128])
def test_torch_microwave_paths_agree_for_shared_scan_fields(device_name, dtype, monitor_dtype):
    """Dense, scalar-envelope, compact, and shared-slow paths agree."""
    from scipy.linalg import expm

    from state_prep_gpu import BatchedSlowHamiltonianTerms, slow_hamiltonian_batch
    from state_prep_gpu.simulator import (
        simulate_batched,
        simulate_batched_compact_microwaves,
        simulate_batched_scalar_microwaves,
        simulate_batched_scalar_microwaves_shared_slow,
    )

    device = torch.device(device_name)
    tolerance = SINGLE_PRECISION_TOL if dtype == torch.complex64 else 1e-12
    n, batch, states, steps = 3, 2, 1, 4
    hff = torch.diag(
        torch.tensor([0.0, 0.3, 0.8], dtype=dtype, device=device)
    )
    zero = torch.zeros((n, n), dtype=dtype, device=device)
    hsz = 0.1 * hff
    terms = BatchedSlowHamiltonianTerms(
        hff, zero, zero, hsz, zero, zero, zero
    )
    fields = np.zeros((steps, 3), dtype=np.float64)
    fields[:, 2] = 0.5
    batched_fields = np.repeat(fields[:, None, :], batch, axis=1)
    psi0 = torch.tensor(
        [[[1.0, 0.2j, 0.1]], [[0.3j, 0.5, 0.7j]]],
        dtype=dtype,
        device=device,
    )
    psi0 = psi0 / torch.linalg.vector_norm(psi0, dim=-1, keepdim=True)
    mu = torch.tensor(
        [[[0.0, 0.04, 0.0], [0.04, 0.0, 0.03], [0.0, 0.03, 0.0]]],
        dtype=dtype,
        device=device,
    )
    amp = torch.ones((steps - 1, 1), dtype=dtype, device=device)
    scales = torch.ones((batch, 1), dtype=torch.float64, device=device)
    shifts = torch.zeros((batch, n), dtype=dtype, device=device)
    monitors = np.eye(n, dtype=monitor_dtype)[:2]
    common = dict(
        terms=terms,
        psi0=psi0,
        dt=0.02,
        monitor_state_vectors=monitors,
    )

    scalar = simulate_batched_scalar_microwaves(
        **common,
        E_t=fields,
        B_t=fields,
        H_mu_base_fields=mu,
        amp_t=amp,
        coupling_scales=scales,
        D_mu_diag_batch=shifts,
    )
    shared = simulate_batched_scalar_microwaves_shared_slow(
        **common,
        E_t=fields,
        B_t=fields,
        H_mu_base_fields=mu,
        amp_t=amp,
        coupling_scales=scales,
        D_mu_diag_batch=shifts,
    )
    dense = simulate_batched(
        **common,
        E_t=batched_fields,
        B_t=batched_fields,
        H_mu_t=mu[None].expand(steps - 1, batch, -1, -1),
        D_mu=torch.diag_embed(shifts),
    )

    Hu = torch.zeros((1, 3, n, n), dtype=dtype, device=device)
    Hu[:, 0] = mu
    Hl = torch.zeros_like(Hu)
    coeff_u = torch.zeros((steps - 1, 1, 3), dtype=dtype, device=device)
    coeff_u[:, :, 0] = 1
    coeff_l = torch.zeros_like(coeff_u)
    compact = simulate_batched_compact_microwaves(
        terms=terms,
        E_t=batched_fields,
        B_t=batched_fields,
        psi0=psi0,
        dt=0.02,
        Hu=Hu,
        Hl=Hl,
        coeff_u_t=coeff_u,
        coeff_l_t=coeff_l,
        coupling_scales=scales,
        D_mu_diag_batch=shifts,
        monitor_state_vectors=monitors,
    )

    # Compare every evolution path to an independent matrix exponential for
    # these constant fields, as well as checking tracked monitor populations.
    h_slow = slow_hamiltonian_batch(terms, fields[0], fields[0])
    out = torch.empty_like(h_slow)
    assert h_slow.dtype == dtype
    assert slow_hamiltonian_batch(terms, fields[0], fields[0], out=out) is out
    assert torch.allclose(out, h_slow, atol=tolerance, rtol=tolerance)
    h_expected = 2 * np.pi * (
        hff.cpu().numpy().astype(np.complex128)
        + fields[0, 2] * hsz.cpu().numpy().astype(np.complex128)
    ) + mu[0].cpu().numpy()
    expected_psi = psi0.cpu().numpy() @ expm(-1j * h_expected * 0.02 * (steps - 1)).T

    for candidate in (scalar, shared, dense, compact):
        assert candidate.psi_final.dtype == dtype
        assert candidate.last_evecs.dtype == dtype
        np.testing.assert_allclose(
            candidate.psi_final.cpu().numpy(), expected_psi,
            atol=tolerance, rtol=tolerance,
        )
        assert torch.allclose(candidate.psi_final, scalar.psi_final, atol=tolerance, rtol=tolerance)
        assert torch.allclose(
            candidate.probabilities_final,
            scalar.probabilities_final,
            atol=tolerance,
            rtol=tolerance,
        )
        assert torch.allclose(
            candidate.monitor_probabilities_final,
            candidate.probabilities_final[..., :2],
            atol=tolerance,
            rtol=tolerance,
        )
