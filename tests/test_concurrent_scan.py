"""CPU/GPU checkpoints must be merged in one explicit readout basis."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("run_cpu_gpu_scan", Path(__file__).resolve().parents[1] / "scripts" / "run_cpu_gpu_scan.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_merge_reprojects_permuted_backend_bases(tmp_path):
    cpu, gpu = tmp_path / "cpu.npz", tmp_path / "gpu.npz"
    basis = np.eye(2, dtype=complex)
    initial = basis[:1]
    launcher._save(cpu, fingerprint="same", indices=np.array([0]), psi_final=np.array([[[1., 0.]]]),
                   final_basis=basis, initial_basis=basis, initial_vectors=initial)
    launcher._save(gpu, fingerprint="same", indices=np.array([1]), psi_final=np.array([[[0., 1.]]]),
                   final_basis=basis[:, ::-1] * 1j, initial_basis=basis, initial_vectors=initial * 1j)
    merged = launcher.merge_checkpoints([str(cpu), str(gpu)], 2, "same")
    np.testing.assert_array_equal(merged["probabilities_final"], [[[1., 0.]], [[0., 1.]]])
    with pytest.raises(ValueError, match="missing, duplicated"):
        launcher.merge_checkpoints([str(cpu), str(cpu)], 2, "same")
    with pytest.raises(ValueError, match="fingerprint"):
        launcher.merge_checkpoints([str(cpu), str(gpu)], 2, "different")


def test_merge_rejects_different_physical_initial_states(tmp_path):
    paths = [tmp_path / "cpu.npz", tmp_path / "gpu.npz"]
    for i, path in enumerate(paths):
        launcher._save(path, fingerprint="same", indices=np.array([i]), psi_final=np.eye(2)[None, :1],
                       final_basis=np.eye(2), initial_basis=np.eye(2), initial_vectors=np.eye(2)[i:i+1])
    with pytest.raises(ValueError, match="initial states disagree"):
        launcher.merge_checkpoints(paths, 2, "same")
