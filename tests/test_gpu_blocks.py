"""Exact blocks must survive changing sorted eigenvalue order and fields."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from state_prep_gpu._blocks import shared_slow_blocks
from state_prep_gpu.slow_hamiltonian import BatchedSlowHamiltonianTerms
from state_prep_gpu.simulator import simulate_batched_scalar_microwaves_shared_slow

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transverse", [False, True])
def test_block_propagation_matches_dense_across_sector_crossing(device, transverse):
    dtype = torch.complex128
    n, steps, batch = 4, 9, 3
    zero = torch.zeros((n, n), dtype=dtype, device=device)
    hff = torch.diag(torch.tensor([0.1, 0.4, 0.8, 1.1], dtype=dtype, device=device))
    hsz = torch.diag(torch.tensor([1., 1., -1., -1.], dtype=dtype, device=device))
    hsx = zero.clone()
    hsx[1, 2] = hsx[2, 1] = 0.15
    mu = zero.clone()
    mu[0, 1] = mu[1, 0] = 0.1
    mu[2, 3] = mu[3, 2] = 0.1
    terms = BatchedSlowHamiltonianTerms(hff, hsx, zero, hsz, zero, zero, zero)
    e = torch.zeros((steps, 3), dtype=torch.float64, device=device)
    e[:, 2] = torch.linspace(-1, 1, steps, device=device)
    if transverse:
        e[-2, 0] = 0.3  # Absent initially: initial-only detection is unsafe.
    b = torch.zeros_like(e)
    blocks = shared_slow_blocks(terms, e, b, e[0], b[0], mu[None])
    assert (blocks is None) == transverse
    psi0 = torch.eye(n, dtype=dtype, device=device)[None].repeat(batch, 1, 1)
    options = dict(terms=terms, E_t=e, B_t=b, psi0=psi0, dt=0.03,
                   H_mu_base_fields=mu[None], amp_t=torch.ones((steps-1, 1), device=device),
                   coupling_scales=torch.ones((batch, 1), device=device),
                   D_mu_diag_batch=torch.zeros((batch, n), device=device),
                   store_every=2)
    dense = simulate_batched_scalar_microwaves_shared_slow(**options, block_eigh=False)
    blocked = simulate_batched_scalar_microwaves_shared_slow(**options, block_eigh=True)
    np.testing.assert_allclose(blocked.psi_final.cpu(), dense.psi_final.cpu(), atol=1e-12)
    np.testing.assert_allclose(blocked.probabilities_final.cpu(), dense.probabilities_final.cpu(), atol=1e-12)
    np.testing.assert_allclose(blocked.psi_history.cpu(), dense.psi_history.cpu(), atol=1e-12)
