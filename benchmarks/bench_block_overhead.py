"""Compare current block solver overhead with an explicit Git baseline."""
from __future__ import annotations

import argparse
import subprocess
import sys
import types

import numpy as np

from common import REPO_ROOT, build_spa2_setup, make_payload, write_results
from paired import compare_paired, format_report


def load_baseline(name, path, ref):
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=REPO_ROOT, text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"baseline/{path}", "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument("--baseline-ref", default="b02b546")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    setup = build_spa2_setup()
    rows = []
    if args.device == "cpu":
        from state_prep import Simulator
        baseline = load_baseline("state_prep._block_baseline", "src/state_prep/simulator.py", args.baseline_ref)
    else:
        import torch
        import state_prep_gpu._blocks as blocks
        from bench_gpu_spa2 import make_torch_runner
        torch.set_num_threads(1)
        original = blocks.eigh_in_blocks
        baseline = load_baseline("_gpu_block_baseline", "src/state_prep_gpu/_blocks.py", args.baseline_ref)
    for steps in args.steps:
        for batch in args.batches:
            det = np.repeat(np.linspace(-2e6, 1e6, batch)[:, None], 2, axis=1)
            power = np.ones_like(det)
            if args.device == "cpu":
                def runner(cls):
                    def run():
                        sim = cls(setup["trajectory"], setup["electric_field"], setup["magnetic_field"],
                                  setup["initial_states"], setup["hamiltonian"], setup["microwave_fields"])
                        return sim.run_microwave_scan(detunings_hz=det, intensity_prefactors=power,
                                                     N_steps=steps, workers=1, progress=False)
                    return run
                variants = dict(baseline=runner(baseline.Simulator), candidate=runner(Simulator))
                sync = None
                check = lambda a, b: float(np.max(np.abs(a.probabilities_final-b.probabilities_final)))
            else:
                run, sync = make_torch_runner(setup, n_steps=steps, detunings=det, intensity_prefactors=power,
                                             slow_eigh_chunk_size=1)
                def runner(helper):
                    def wrapped():
                        blocks.eigh_in_blocks = helper
                        try:
                            return run()
                        finally:
                            blocks.eigh_in_blocks = original
                    return wrapped
                variants = dict(baseline=runner(baseline.eigh_in_blocks), candidate=runner(original))
                check = lambda a, b: float(torch.max(torch.abs(a.probabilities_final-b.probabilities_final)).item())
            print(f"{args.device}: steps={steps} batch={batch}", flush=True)
            pair = compare_paired(variants, warmup=args.warmup, repeat=args.repeat,
                                  sync=sync, check=check, tolerance=1e-6, progress=True)
            print(format_report(pair), flush=True)
            if not all(c.agrees for c in pair.comparisons):
                raise RuntimeError("Block solver changed final probabilities")
            a, b = pair.results["baseline"], pair.results["candidate"]
            av, bv = ((a.psis_final, b.psis_final) if args.device == "cpu" else
                      (a.psi_final.cpu().numpy(), b.psi_final.cpu().numpy()))
            infidelity = float(np.max(np.abs(1-np.abs(np.sum(av.conj()*bv, axis=-1))**2)))
            if infidelity > 1e-6:
                raise RuntimeError(f"Physical final states disagree: {infidelity}")
            rows.extend(dict(row, steps=steps, batch=batch, max_infidelity=infidelity) for row in pair.rows())
            write_results(make_payload(benchmark="block_overhead", config=vars(args), results=rows,
                                       include_gpu_env=args.device == "gpu"),
                          output=args.output, csv_path=None)


if __name__ == "__main__":
    main()
