"""Settle the four SPA2 cells where Magnus looked worse than the frozen propagator.

The regressions -- `det=-2.0 p=1`, `det=-2.0 p=4`, `det=+0.0 p=16`,
`det=+1.5 p=16` -- were reported at absolute differences `<= 2.2e-04`. Ruled out
previously: `magnus_integral` cancellation, near-degenerate spectra, and the
`D_mu` diagonal. No closed form is available, since `V(t) diag(D_mu) V(t)^H`
breaks every analytic solution at production `delta*dt`.

**Do not settle this by comparing against a fine run of either scheme.** Two
independent things break that comparison:

1. `probabilities_final` is indexed by adiabatically tracked eigenstate labels,
   and 56 of 64 labels cross on this setup. Runs at different `N_steps` track
   independently, so the same physical population can land in a different
   column. Measured: comparing `N=4000` against `N=128000` gives differences of
   `0.999` in cells where the physics is nearly identical -- a permutation, not
   an error.
2. A reference drawn from one scheme flatters that scheme, which is how this
   repository reached two opposite wrong conclusions about graded grids.

What works needs no reference at all: compare the two schemes **at the same
N_steps**, where they share one label path, and watch that difference as
`N_steps` grows. A systematic defect in one scheme does not shrink. Measured, it
shrinks in every cell -- `4.69e-03 -> 9.85e-06` from `N=4000` to `N=64000` in the
worst-reported cell, tracking the control cell exactly -- so the four
regressions are differences between two unconverged runs, well below the
`~5e-03` discretisation floor both schemes share at `N=4000`, and not a property
of Magnus.

    python benchmarks/bench_magnus_regressions.py --n-steps 4000 16000 64000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import build_spa2_setup

# The four cells reported as regressions, plus controls that were not.
REGRESSED = [(-2.0, 1.0), (-2.0, 4.0), (0.0, 16.0), (1.5, 16.0)]
CONTROLS = [(-1.0, 2.0), (0.5, 8.0)]


def run(setup, det_b, pref_b, n_steps, propagator):
    from state_prep.simulator import Simulator

    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        setup["microwave_fields"],
    )
    result = simulator.run_microwave_scan(
        detunings_hz=det_b,
        intensity_prefactors=pref_b,
        N_steps=n_steps,
        store_final_probabilities=True,
        progress=False,
        workers=1,
        propagator=propagator,
    )
    return np.asarray(result.probabilities_final)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-steps", type=int, nargs="+", default=[4000, 16000, 64000])
    args = ap.parse_args()

    cells = REGRESSED + CONTROLS
    setup = build_spa2_setup()
    n_fields = len(setup["microwave_fields"])
    det = np.array([d * 1e6 for d, _ in cells])
    pref = np.array([p for _, p in cells])
    det_b = np.column_stack([det] * n_fields)
    pref_b = np.column_stack([np.ones_like(pref)] + [pref] * (n_fields - 1))

    diffs = {}
    for n in args.n_steps:
        t0 = time.perf_counter()
        a = run(setup, det_b, pref_b, n, "frozen")
        b = run(setup, det_b, pref_b, n, "magnus")
        diffs[n] = np.abs(a - b).max(axis=1)
        print(f"  N={n:<8} {time.perf_counter() - t0:7.1f}s", flush=True)

    print()
    print("same-N |frozen - magnus|; a systematic defect would not shrink")
    header = f"{'cell':<18}" + "".join(f"{'N=' + str(n):>13}" for n in args.n_steps)
    print(header)
    for i, (d, p) in enumerate(cells):
        mark = "*" if (d, p) in REGRESSED else " "
        row = "".join(f"{diffs[n][i]:>13.2e}" for n in args.n_steps)
        print(f"{mark}det={d:+.1f} p={p:<6g}{row}")

    print()
    print("* = one of the four originally reported regressions")
    print("Compare each row against the controls. Falling together means the")
    print("cells differ only in how far from converged they are.")


if __name__ == "__main__":
    main()
