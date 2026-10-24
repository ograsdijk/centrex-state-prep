"""Block solver spectra, reconstruction, and labels on non-contiguous sectors."""
import numpy as np
import pytest

from state_prep.simulator import _eigh_in_blocks


@pytest.mark.parametrize("backend", ["numpy", "zheevd"])
def test_cpu_blocks_reconstruct_complex_matrix(backend):
    rng = np.random.default_rng(42)
    blocks = [np.array([5]), np.array([3, 0]), np.array([4]), np.array([2, 1])]
    matrix = np.zeros((6, 6), dtype=complex)
    for indices in blocks:
        a = rng.normal(size=(len(indices), len(indices))) + 1j*rng.normal(size=(len(indices), len(indices)))
        matrix[indices[:, None], indices] = a + a.conj().T
    values, vectors, labels = _eigh_in_blocks(matrix, blocks, backend)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(matrix), atol=1e-12)
    np.testing.assert_allclose((vectors*values) @ vectors.conj().T, matrix, atol=1e-12)
    for column, block_id in enumerate(labels):
        outside = np.setdiff1d(np.arange(6), blocks[block_id])
        np.testing.assert_array_equal(vectors[outside, column], 0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype_name", ["complex64", "complex128"])
def test_gpu_grouped_blocks_reconstruct_complex_batches(device, dtype_name):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from state_prep_gpu._blocks import eigh_in_blocks

    dtype = getattr(torch, dtype_name)
    rng = np.random.default_rng(42)
    matrix = np.zeros((2, 3, 6, 6), dtype=complex)
    groups = [[5], [3, 0], [4], [2, 1]]
    for group in groups:
        indices = np.array(group)
        a = rng.normal(size=(2, 3, len(group), len(group))) + 1j*rng.normal(size=(2, 3, len(group), len(group)))
        matrix[..., indices[:, None], indices] = a + a.conj().swapaxes(-1, -2)
    matrix = torch.as_tensor(matrix, dtype=dtype, device=device)
    blocks = [torch.tensor(group, device=device) for group in groups]
    values, vectors, labels = eigh_in_blocks(matrix, blocks)
    tolerance = 2e-5 if dtype_name == "complex64" else 1e-12
    torch.testing.assert_close(values, torch.linalg.eigvalsh(matrix), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close((vectors*values.unsqueeze(-2)) @ vectors.mH, matrix, atol=tolerance, rtol=tolerance)
    for block_id, group in enumerate(groups):
        outside = [index for index in range(6) if index not in group]
        leakage = torch.where((labels == block_id).unsqueeze(-2), vectors[..., outside, :], 0)
        assert torch.count_nonzero(leakage).item() == 0
