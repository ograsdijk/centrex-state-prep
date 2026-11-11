"""Full per-point eigensolves versus `coupling_tolerance` on the CPU scan.

`coupling_tolerance` restricts each scan point's rotating-frame solve to the
slow eigenstates the microwaves couple appreciably. Baseline and candidate are
the same code path with the option off and on, interleaved, so the comparison
isolates the option. Numerical agreement is recorded alongside every timing.

    python benchmarks/bench_coupling_tolerance.py --output results/coupling_tolerance.json

Cases:
  spa2_aligned, spa2_varying_e, spa2_transverse_b   J=1->2, Js=0..3, n=64
  cascade_aligned, cascade_varying_e                J=0->1->2, Js=0..4, n=100

The non-collinear field variants are the synthetic probes defined in
`profile_transverse_fields.py`; they are not measured field maps.
"""
from __future__ import annotations

import argparse

import numpy as np

from common import build_spa_cascade_setup, make_payload, write_results
from paired import compare_paired, format_report
from profile_transverse_fields import field_variant

CASES = [
    "spa2_aligned",
    "spa2_varying_e",
    "spa2_transverse_b",
    "cascade_aligned",
    "cascade_varying_e",
]


def build_case(case):
    family, variant = case.split("_", 1)
    if family == "spa2":
        return field_variant(variant)
    setup = build_spa_cascade_setup(js=(0, 1, 2, 3, 4))
    if variant == "varying_e":
        original_e = setup["electric_field"].E_R

        # Same probe as `field_variant("varying_e")`.
        def e_r(position):
            e = original_e(position).copy()
            phase = 2 * np.pi * (position[2] + 0.08) / 0.28
            e[0] = 0.02 * e[2] * np.sin(phase)
            e[1] = 0.01 * e[2] * np.cos(phase)
            return e

        setup["electric_field"].E_R = e_r
    elif variant != "aligned":
        raise ValueError(case)
    return setup


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--shapes", nargs="+", default=["300:25", "1000:25", "300:100"],
                        help="steps:batch pairs")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--agreement", type=float, default=1e-7,
                        help="largest accepted change in final probabilities")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from state_prep import Simulator

    rows = []
    for case in args.cases:
        setup = build_case(case)
        sim = Simulator(setup["trajectory"], setup["electric_field"], setup["magnetic_field"],
                        setup["initial_states"], setup["hamiltonian"], setup["microwave_fields"])
        for shape in args.shapes:
            steps, batch = (int(value) for value in shape.split(":"))
            det = np.repeat(np.linspace(-2e6, 1e6, batch)[:, None], 2, axis=1)
            power = np.ones_like(det)

            def runner(tolerance):
                def run():
                    return sim.run_microwave_scan(
                        detunings_hz=det, intensity_prefactors=power, N_steps=steps,
                        workers=1, progress=False, coupling_tolerance=tolerance)
                return run

            variants = dict(baseline=runner(None), candidate=runner(args.tolerance))
            check = lambda a, b: float(np.max(np.abs(a.probabilities_final - b.probabilities_final)))
            print(f"{case}: steps={steps} batch={batch}", flush=True)
            pair = compare_paired(variants, warmup=args.warmup, repeat=args.repeat,
                                  check=check, tolerance=args.agreement, progress=True)
            print(format_report(pair), flush=True)
            if not all(c.agrees for c in pair.comparisons):
                raise RuntimeError("coupling_tolerance changed final probabilities beyond --agreement")
            a, b = pair.results["baseline"], pair.results["candidate"]
            infidelity = float(np.max(np.abs(
                1 - np.abs(np.sum(a.psis_final.conj() * b.psis_final, axis=-1)) ** 2)))
            rows.extend(dict(row, case=case, steps=steps, batch=batch,
                             n=int(a.psis_final.shape[-1]), max_infidelity=infidelity)
                        for row in pair.rows())
            write_results(make_payload(benchmark="coupling_tolerance", config=vars(args),
                                       results=rows, include_gpu_env=False),
                          output=args.output, csv_path=None)


if __name__ == "__main__":
    main()
