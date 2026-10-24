"""Paired checks for local CPU changes and production GPU exact blocks.

CPU baseline source is read from an explicit Git revision, never another venv.
GPU dense and block variants both use corrected optimal tracking. Preparation
is excluded for GPU runners; block discovery inside each propagation is included.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import types

import numpy as np

from common import build_spa2_setup, make_payload, write_results, REPO_ROOT
from paired import compare_paired, format_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1000])
    parser.add_argument("--batches", type=int, nargs="+", default=[5, 25])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--baseline-ref", default="315af72")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    setup = build_spa2_setup()
    setup["initial_states"] = setup["initial_states"][:4]
    rows = []
    if args.device == "cpu":
        from state_prep import Simulator

        source = subprocess.check_output(
            ["git", "show", f"{args.baseline_ref}:src/state_prep/simulator.py"],
            cwd=REPO_ROOT, text=True, encoding="utf-8",
        )
        baseline = types.ModuleType("state_prep._performance_baseline")
        sys.modules[baseline.__name__] = baseline
        exec(compile(source, "baseline_simulator.py", "exec"), baseline.__dict__)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from analyze_spa2_bg_feature import make_rc_bg_field

        rc, _ = make_rc_bg_field(setup, bg_fraction=1 / 35, rc_offset_mhz=-7.4)
        cases = [("frozen", setup["microwave_fields"]),
                 ("magnus", [*setup["microwave_fields"], rc])]
        for steps in args.steps:
            for batch in args.batches:
                for propagator, fields in cases:
                    from state_prep import scan_grid

                    det, power = scan_grid(n_fields=len(fields), detunings_hz=np.linspace(-2e6, 1e6, batch))
                    def runner(cls):
                        def run():
                            sim = cls(setup["trajectory"], setup["electric_field"], setup["magnetic_field"], setup["initial_states"], setup["hamiltonian"], fields)
                            return sim.run_microwave_scan(
                                detunings_hz=det, intensity_prefactors=power,
                                N_steps=steps, progress=False, workers=1,
                                allow_multitone_same_manifold=True, propagator=propagator,
                            )
                        return run
                    print(f"CPU {propagator}: steps={steps} batch={batch}", flush=True)
                    pair = compare_paired(
                        {"baseline": runner(baseline.Simulator), "candidate": runner(Simulator)},
                        baseline="baseline", repeat=args.repeat, warmup=args.warmup,
                        check=lambda a, b: float(np.max(np.abs(a.probabilities_final-b.probabilities_final))),
                        tolerance=1e-10, progress=True,
                    )
                    print(format_report(pair), flush=True)
                    rows.extend(dict(row, steps=steps, batch=batch, propagator=propagator) for row in pair.rows())
    else:
        import torch
        from bench_gpu_spa2 import make_torch_runner, run_cpu
        from state_prep import scan_grid

        for steps in args.steps:
            for batch in args.batches:
                det, power = scan_grid(n_fields=2, detunings_hz=np.linspace(-2e6, 1e6, batch))
                dense, sync = make_torch_runner(setup, n_steps=steps, detunings=det, intensity_prefactors=power, block_eigh=False)
                blocks, _ = make_torch_runner(setup, n_steps=steps, detunings=det, intensity_prefactors=power, block_eigh=True)
                print(f"GPU blocks: steps={steps} batch={batch}", flush=True)
                pair = compare_paired(
                    {"dense": dense, "blocks": blocks}, baseline="dense",
                    repeat=args.repeat, warmup=args.warmup, sync=sync,
                    check=lambda a, b: float(torch.max(torch.abs(a.probabilities_final-b.probabilities_final)).item()),
                    tolerance=1e-5, progress=True,
                )
                print(format_report(pair), flush=True)
                # Physical state agreement against the CPU reference as well as
                # labeled probabilities; no global phase sensitivity in fidelity.
                cpu = run_cpu(setup, n_steps=steps, detunings=det, intensity_prefactors=power)
                gpu = pair.results["blocks"]
                vectors = gpu.psi_final.cpu().numpy()
                fidelity = np.abs(np.sum(cpu.psis_final.conj()*vectors, axis=-1)) ** 2
                error = float(np.max(np.abs(cpu.probabilities_final-gpu.probabilities_final.cpu().numpy())))
                rows.append(dict(case="cpu_gpu_agreement", steps=steps, batch=batch,
                                 max_probability_difference=error,
                                 max_infidelity=float(np.max(np.abs(1-fidelity)))))
                if error > 1e-5 or np.max(np.abs(1-fidelity)) > 1e-5:
                    raise RuntimeError(f"CPU/GPU agreement failed: probability difference {error}")
                rows.extend(dict(row, steps=steps, batch=batch) for row in pair.rows())
    payload = make_payload(benchmark="performance_updates", config=vars(args), results=rows,
                           include_gpu_env=args.device == "gpu")
    payload["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    write_results(payload, output=args.output, csv_path=None)


if __name__ == "__main__":
    main()
