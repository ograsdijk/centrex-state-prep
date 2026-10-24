"""Optimality and phase checks independent of simulation agreement."""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

torch = pytest.importorskip("torch")
from state_prep_gpu._assignment import linear_assignment
from state_prep_gpu._reorder import reorder_evecs_batched

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 2, 7, 32, 64, 100, 256, 512])
def test_assignment_matches_scipy_objective(device, dtype, n):
    rng = np.random.default_rng(11 + n)
    costs = rng.normal(size=(5, n, n))
    costs[0] = 0  # Fully tied: compare objectives, not arbitrary permutations.
    if n > 1:
        costs[1, :, 0] = -20  # Deliberately conflicting row minima.
    tensor = torch.tensor(costs, device=device, dtype=dtype)
    costs = tensor.cpu().numpy()
    orders = linear_assignment(tensor).cpu().numpy()
    for cost, order in zip(costs, orders):
        np.testing.assert_array_equal(np.sort(order), np.arange(n))
        optimum = linear_sum_assignment(cost)[1]
        # Sum in float64, including when costs arrived as float32.
        actual = cost[np.arange(n), order].astype(float).sum()
        expected = cost[np.arange(n), optimum].astype(float).sum()
        assert abs(actual - expected) < (2e-5 if dtype == torch.float32 else 1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_complex_reorder_matches_cpu_and_preserves_inputs(device):
    from state_prep.utils import reorder_evecs

    rng = np.random.default_rng(54)
    vectors, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    reference = np.eye(12, dtype=complex)
    values = np.arange(12, dtype=float)
    expected_e, expected_v = reorder_evecs(vectors.copy(), values, reference)
    v = torch.tensor(vectors, device=device)
    actual_e, actual_v = reorder_evecs_batched(v[None], torch.tensor(values, device=device)[None], torch.tensor(reference, device=device)[None])
    np.testing.assert_allclose(actual_e.cpu()[0], expected_e, atol=0, rtol=0)
    np.testing.assert_allclose(actual_v.cpu()[0], expected_v, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(v.cpu(), vectors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_assignment_uses_current_cuda_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        costs = torch.rand((3, 64, 64), dtype=torch.float64, device="cuda")
        order = linear_assignment(costs)
        score = costs.gather(-1, order.unsqueeze(-1)).sum()
    stream.synchronize()
    reference = costs.cpu().numpy()
    expected = sum(c[np.arange(64), linear_sum_assignment(c)[1]].sum() for c in reference)
    assert abs(score.item() - expected) < 1e-10


@pytest.mark.parametrize("device", DEVICES)
def test_zero_overlap_phase_is_finite(device):
    vectors = torch.zeros((1, 2, 2), dtype=torch.complex128, device=device)
    _, result = reorder_evecs_batched(vectors, torch.zeros((1, 2), device=device), vectors)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("module", ["example_compare_cpu", "benchmark_compare"])
def test_numpy_reference_helpers_use_cpu_optimal_tracking(module):
    import importlib
    from state_prep.utils import reorder_evecs

    rng = np.random.default_rng(54)
    vectors, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    values = np.arange(12, dtype=float)
    reference = np.eye(12, dtype=complex)
    expected_e, expected_v = reorder_evecs(vectors, values, reference)
    helper = importlib.import_module(f"state_prep_gpu.{module}")._reorder_evecs_batched_np
    actual_e, actual_v = helper(vectors[None], values[None], reference[None])
    np.testing.assert_allclose(actual_e[0], expected_e, atol=0, rtol=0)
    np.testing.assert_allclose(actual_v[0], expected_v, atol=1e-12, rtol=0)
