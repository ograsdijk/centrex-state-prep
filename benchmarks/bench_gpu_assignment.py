"""Compare GPU optimal tracking with exact SciPy matching via host transfers."""
from __future__ import annotations

import argparse
import numpy as np
import torch

from common import build_spa2_setup, make_payload, write_results
from paired import compare_paired, format_report
from bench_gpu_spa2 import make_torch_runner
from state_prep import scan_grid
from state_prep_gpu import _reorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    setup = build_spa2_setup()
    setup["initial_states"] = setup["initial_states"][:4]
    det, power = scan_grid(n_fields=2, detunings_hz=np.linspace(-2e6, 1e6, args.batch))
    run, sync = make_torch_runner(setup, n_steps=args.steps, detunings=det,
                                  intensity_prefactors=power, block_eigh=False)
    on_device = _reorder.linear_assignment
    def via_cpu(cost):
        return on_device(cost.cpu()).to(cost.device)
    def variant(solver):
        def call():
            saved = _reorder.linear_assignment
            try:
                _reorder.linear_assignment = solver
                return run()
            finally:
                _reorder.linear_assignment = saved
        return call
    pair = compare_paired(
        {"cuda_optimal": variant(on_device), "scipy_transfer": variant(via_cpu)},
        baseline="scipy_transfer", repeat=args.repeat, warmup=args.warmup, sync=sync,
        check=lambda a, b: torch.max(torch.abs(a.probabilities_final-b.probabilities_final)).item(),
        tolerance=1e-5, progress=True,
    )
    print(format_report(pair), flush=True)
    if not all(comparison.agrees for comparison in pair.comparisons):
        raise RuntimeError("GPU/CPU optimal tracking disagrees on this trajectory")
    write_results(make_payload(benchmark="gpu_assignment", config=vars(args), results=pair.rows()),
                  output=args.output, csv_path=None)


if __name__ == "__main__":
    main()
