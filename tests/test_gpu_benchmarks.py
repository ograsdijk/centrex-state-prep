"""Benchmark failure paths must remain usable on CPU-only installations."""

import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("flags", [[], ["--cpu"], ["--gpu"]])
@pytest.mark.parametrize("failure", ["setup", "propagation"])
def test_spa2_keeps_cpu_baseline_when_gpu_is_unavailable(monkeypatch, flags, failure):
    import bench_gpu_spa2 as benchmark

    cpu_calls = []
    payloads = []
    monkeypatch.setattr(
        benchmark, "build_spa2_setup",
        lambda: {"initial_states": [0], "hamiltonian": SimpleNamespace(QN=[0, 1])},
    )
    monkeypatch.setattr(benchmark, "run_cpu", lambda *a, **k: cpu_calls.append(True))

    def unavailable(*args, **kwargs):
        raise RuntimeError("CUDA is unavailable")

    monkeypatch.setattr(
        benchmark, "make_torch_runner",
        unavailable if failure == "setup" else lambda *a, **k: (unavailable, lambda: None),
    )
    monkeypatch.setattr(benchmark, "make_payload", lambda **kwargs: kwargs)
    monkeypatch.setattr(benchmark, "write_results", lambda payload, **k: payloads.append(payload))
    monkeypatch.setattr(sys, "argv", [
        "bench_gpu_spa2.py", "--batch", "1", "--scan-modes", "detuning",
        "--initial-state-sweep", "1", "--warmup", "0", "--repeat", "1", *flags,
    ])

    benchmark.main()

    rows = payloads[0]["results"]
    cpu_rows = [row for row in rows if row["case"] == "cpu_shared_slow"]
    gpu_rows = [row for row in rows if row["case"] == "pytorch_shared_slow"]
    assert len(cpu_rows) == (0 if flags == ["--gpu"] else 1)
    assert bool(cpu_calls) == (flags != ["--gpu"])
    if flags != ["--cpu"]:
        assert gpu_rows[0]["status"] == "skipped"
        assert "CUDA is unavailable" in gpu_rows[0]["error"]


def test_retired_jacobi_cli_reports_supported_replacements(monkeypatch, capsys):
    import bench_gpu_jacobi_tolerance as benchmark

    monkeypatch.setattr(sys, "argv", ["bench_gpu_jacobi_tolerance.py"])
    with pytest.raises(SystemExit) as error:
        benchmark.main()

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "retired" in message
    assert "bench_gpu_spa2.py" in message
    assert "bench_gpu_eig_tolerance.py" in message
