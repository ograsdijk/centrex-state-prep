"""Do the saved `results/` need regenerating after the midpoint default?

`time_sampling` defaulted to `"left"` when the analyses in `results/` were run
and defaults to `"mid"` now, so those files no longer match what the code
produces. Regenerating them is expensive; deciding whether it *matters* is not.

This runs the configuration behind `results/spa2_bg_rc_analysis.json` -- 33
detunings, RC background on the same manifold, `N_steps=20000` -- under both
samplings and compares the physical observables the analysis reports:
`depletion` and `transferred`.

The comparison to make is against the convergence the SPA report already quotes
for those quantities, `1.4e-05`. A difference below that is not physical and the
saved files can stand; a difference above it means the numbers in the report
moved and the files need regenerating.

    python benchmarks/bench_sampling_sensitivity.py --n-steps 20000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bench_convergence import build_multitone_setup

# Convergence quoted for depletion/transferred in the SPA report.
REPORT_CONVERGENCE = 1.4e-05


def run(setup, det_b, pref_b, n_steps, time_sampling, monitor_states):
    from state_prep.simulator import Simulator

    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        setup["microwave_fields"],
    )
    return simulator.run_microwave_scan(
        detunings_hz=det_b,
        intensity_prefactors=pref_b,
        N_steps=n_steps,
        monitor_states=monitor_states,
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        workers=1,
        allow_multitone_same_manifold=True,
        time_sampling=time_sampling,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-steps", type=int, default=20000)
    ap.add_argument("--n-detunings", type=int, default=33)
    ap.add_argument("--lo-mhz", type=float, default=-2.0)
    ap.add_argument("--hi-mhz", type=float, default=1.5)
    args = ap.parse_args()

    setup = build_multitone_setup(
        bg_fraction=1 / 35, rc_bg_fraction=1 / 35, rc_offset_mhz=0.0
    )
    det = np.linspace(args.lo_mhz, args.hi_mhz, args.n_detunings) * 1e6
    # Matches the analysis: the RC column stays at zero, since scanning the SPA2
    # frequency does not move a fixed RC source.
    det_b = np.column_stack([det, det, np.zeros_like(det)])
    pref_b = np.ones((det.size, 3))

    out = {}
    for sampling in ("left", "mid"):
        t0 = time.perf_counter()
        out[sampling] = run(
            setup, det_b, pref_b, args.n_steps, sampling, setup["monitor_states"]
        )
        print(f"  {sampling:>5} {time.perf_counter() - t0:7.1f}s", flush=True)

    print()
    print(f"N_steps={args.n_steps}, {args.n_detunings} detunings")
    print(f"{'observable':<16}{'max |left - mid|':>20}{'vs 1.4e-05':>14}")
    verdict_needed = False
    for name in ("depletion", "transferred"):
        curves = []
        for sampling in ("left", "mid"):
            r = out[sampling]
            populations = np.asarray(r.probabilities_final)
            monitors = np.asarray(r.monitor_probabilities_final)
            if name == "depletion":
                curves.append(1.0 - populations[:, 0])
            else:
                curves.append(monitors.sum(axis=1))
        diff = float(np.abs(curves[0] - curves[1]).max())
        ratio = diff / REPORT_CONVERGENCE
        flag = "REGENERATE" if ratio > 1.0 else "within noise"
        verdict_needed |= ratio > 1.0
        print(f"{name:<16}{diff:>20.3e}{ratio:>11.1f}x  {flag}")

    print()
    print("The saved analyses used time_sampling='left' (the old default).")
    print("REGENERATE means the reported numbers moved by more than the")
    print("convergence the report itself quotes, so they are stale in substance,")
    print("not just in provenance.")


if __name__ == "__main__":
    main()
