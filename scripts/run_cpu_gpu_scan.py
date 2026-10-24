"""Run independent shared-slow scan points on CPU and GPU, with checkpoints.

Default setup is the benchmark SPA2 configuration. --setup-file accepts a dill
file containing the same setup mapping (trajectory, fields, Hamiltonian,
microwaves, initial_states, monitor_states). Only uniform midpoint frozen
propagation and static-polarization scalar microwave envelopes are supported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import dill
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _save(path, **arrays):
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _worker(backend, indices, config, fingerprint, output, setup_path):
    from bench_gpu_spa2 import make_torch_runner, run_cpu
    from state_prep import Simulator

    output = Path(output)
    path = output / f"{backend}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) != fingerprint or not np.array_equal(saved["indices"], indices):
                raise ValueError(f"Checkpoint does not match this request: {path}")
        return str(path)
    started = time.perf_counter()
    with open(setup_path, "rb") as stream:
        setup = dill.load(stream)
    with np.load(output / "parameters.npz", allow_pickle=False) as params:
        det, power = params["detunings_hz"][indices], params["intensity_prefactors"][indices]
    setup["initial_states"] = setup["initial_states"][:config["initial_states"]]
    if backend == "cpu":
        preparation = time.perf_counter() - started
        propagation_started = time.perf_counter()
        result = run_cpu(setup, n_steps=config["steps"], detunings=det,
                         intensity_prefactors=power, workers=config["workers"])
        propagation = time.perf_counter() - propagation_started
        # A spawned outer worker must close its reusable loky children before
        # multiprocessing's exit handler tries to join them on Windows.
        if config["workers"] > 1:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor(reuse=True).shutdown(wait=True)
        psi, basis = result.psis_final, result.V_fin
        initial_basis = result.V_ini
        initial_vectors = np.stack([s.state_vector(setup["hamiltonian"].QN) for s in result.initial_states])
    else:
        import torch
        from common import microwave_amp_t_np, microwave_base_matrices_np

        torch.set_num_threads(1)
        # Reject same-manifold beating: the scalar GPU kernel does not model it.
        carriers = {}
        for field in setup["microwave_fields"]:
            if field.Je in carriers and not np.isclose(carriers[field.Je], field.muW_freq):
                raise ValueError("GPU scan does not support distinct carriers on the same manifold")
            carriers[field.Je] = field.muW_freq
        times = np.linspace(0, setup["trajectory"].get_T(), config["steps"])
        mids = (times[:-1] + times[1:]) / 2
        base = microwave_base_matrices_np(setup["microwave_fields"], trajectory=setup["trajectory"], t0=float(mids[0]), n=len(setup["hamiltonian"].QN))
        amp = microwave_amp_t_np(setup["microwave_fields"], trajectory=setup["trajectory"], t_array=np.r_[mids, times[-1]])
        # Validate the scalar factorization at every time actually propagated.
        for j, field in enumerate(setup["microwave_fields"]):
            ham = field.get_H_t_func(setup["trajectory"].R_t, setup["hamiltonian"].QN)
            for i, t in enumerate(mids):
                actual = ham(float(t))
                if not np.allclose(actual, amp[i, j] * base[j], atol=1e-10, rtol=1e-12):
                    raise ValueError("GPU scalar envelope does not reproduce this microwave field")
        run, sync = make_torch_runner(setup, n_steps=config["steps"], detunings=det, intensity_prefactors=power)
        sync()
        preparation = time.perf_counter() - started
        propagation_started = time.perf_counter()
        result = run()
        sync()
        propagation = time.perf_counter() - propagation_started
        psi, basis = result.psi_final.cpu().numpy(), result.last_evecs.cpu().numpy()
        sim = Simulator(setup["trajectory"], setup["electric_field"], setup["magnetic_field"], setup["initial_states"], setup["hamiltonian"], setup["microwave_fields"])
        h0 = setup["hamiltonian"].get_H_t_func()(0.)
        _, initial_basis = np.linalg.eigh(h0)
        sim.init_state_vecs(h0, V_0=initial_basis)
        initial_vectors = sim.psis
    _save(path, fingerprint=fingerprint, indices=indices, psi_final=psi,
          final_basis=basis, initial_basis=initial_basis, initial_vectors=initial_vectors,
          preparation_s=preparation, propagation_s=propagation)
    print(f"{backend}: {len(indices)} points, preparation={preparation:.3f}s propagation={propagation:.3f}s", flush=True)
    return str(path)


def merge_checkpoints(paths, batch, fingerprint):
    """Reproject physical states into one canonical tracked readout basis.

    Backend eigenvectors may differ in phases, permutations or degenerate
    subspaces. Concatenating their labeled probabilities directly is unsafe.
    """
    checkpoints = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            checkpoints.append({key: saved[key] for key in saved.files})
    reference = checkpoints[0]  # CPU first when both backends participate.
    indices = np.concatenate([c["indices"] for c in checkpoints])
    if not np.array_equal(np.sort(indices), np.arange(batch)):
        raise ValueError("Checkpoints have missing, duplicated or out-of-range points")
    for checkpoint in checkpoints:
        if str(checkpoint["fingerprint"]) != fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        a, b = checkpoint["initial_vectors"], reference["initial_vectors"]
        overlaps = np.abs(np.sum(a.conj() * b, axis=-1))
        if not np.allclose(overlaps, 1, atol=1e-8, rtol=0):
            raise ValueError("CPU/GPU initial states disagree; refusing to merge")
    order = np.argsort(indices)
    psi = np.concatenate([c["psi_final"] for c in checkpoints])[order]
    return dict(psi_final=psi,
                probabilities_final=np.abs(psi @ reference["final_basis"].conj()) ** 2,
                final_basis=reference["final_basis"], initial_basis=reference["initial_basis"],
                initial_vectors=reference["initial_vectors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--setup-file", type=Path)
    parser.add_argument("--parameters", type=Path, help="NPZ: detunings_hz and intensity_prefactors, both (B,M)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cpu-fraction", type=float, default=0.6)
    args = parser.parse_args()
    if args.steps < 2 or args.batch < 1 or args.workers < 1 or args.initial_states < 1 or not 0 <= args.cpu_fraction <= 1:
        parser.error("steps>=2, batch/workers/initial-states>=1 and 0<=cpu-fraction<=1 required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_path = args.setup_file.resolve() if args.setup_file else output / "setup.dill"
    if args.setup_file is None and not setup_path.exists():
        from common import build_spa2_setup
        with setup_path.open("wb") as stream:
            dill.dump(build_spa2_setup(), stream)
    with setup_path.open("rb") as stream:
        setup = dill.load(stream)
    if args.initial_states > len(setup["initial_states"]):
        parser.error("requested more initial states than setup provides")
    if args.parameters:
        with np.load(args.parameters, allow_pickle=False) as params:
            det, power = params["detunings_hz"], params["intensity_prefactors"]
    else:
        from state_prep import scan_grid
        det, power = scan_grid(n_fields=len(setup["microwave_fields"]), detunings_hz=np.linspace(-2e6, 1e6, args.batch))
    if det.ndim != 2 or det.shape != power.shape or det.shape[1] != len(setup["microwave_fields"]) or not len(det):
        parser.error("parameters must have matching nonempty (B,M) shapes")
    if not np.isfinite(det).all() or not np.isfinite(power).all() or (power < 0).any():
        parser.error("parameters must be finite, with nonnegative intensity prefactors")
    batch = len(det)
    config = dict(steps=args.steps, workers=args.workers, initial_states=args.initial_states,
                  cpu_points=round(batch * args.cpu_fraction), batch=batch)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for data in (setup_path.read_bytes(), det.tobytes(), power.tobytes()):
        digest.update(data)
    for path in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__), ROOT / "benchmarks" / "bench_gpu_spa2.py", ROOT / "benchmarks" / "common.py"]:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    manifest = output / "request.json"
    request = dict(config=config, fingerprint=fingerprint)
    if manifest.exists() and json.loads(manifest.read_text()) != request:
        raise ValueError("Output directory contains a different request; choose a new directory")
    manifest.write_text(json.dumps(request, indent=2) + "\n")
    _save(output / "parameters.npz", detunings_hz=det, intensity_prefactors=power)
    tasks = [("cpu", np.arange(config["cpu_points"])), ("gpu", np.arange(config["cpu_points"], batch))]
    tasks = [(backend, indices) for backend, indices in tasks if len(indices)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_worker, backend, indices, config, fingerprint, str(output), str(setup_path)) for backend, indices in tasks]
        paths = [future.result() for future in futures]
    merged = merge_checkpoints(paths, batch, fingerprint)
    monitors = setup.get("monitor_states")
    if monitors:
        vectors = np.stack([s.state_vector(setup["hamiltonian"].QN) for s in monitors])
        monitor_indices = np.argmax(np.abs(merged["initial_basis"].conj().T @ vectors.T), axis=0)
        merged["monitor_probabilities_final"] = merged["probabilities_final"][..., monitor_indices]
    _save(output / "merged.npz", fingerprint=fingerprint, detunings_hz=det,
          intensity_prefactors=power, **merged)
    elapsed = time.perf_counter()-started
    report = dict(request, wall_s=elapsed, checkpoints=paths, merged=str(output / "merged.npz"))
    (output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
