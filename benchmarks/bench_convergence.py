"""Timestep convergence over a (detuning x coupling) grid.

**This benchmark's metric is unreliable, and its default grid triggers the
failure.** It compares `probabilities_final` elementwise at *fixed* detunings.
Near a steep part of the lineshape that measures the slope rather than
discretisation error: on SPA2 `d(population)/d(detuning)` reaches `6e-03` per
kHz, so an effective error of `0.2 kHz` -- far smaller than any step count will
remove -- shows up as `1e-03` of population and refuses to fall with `N_steps`.
`det=-1.0 MHz` is in `DEFAULT_DETUNINGS_MHZ` and is exactly such a point: it sits
on a partial Landau-Zener transfer at a crossing `2.5 sigma` out in the beam
flank.

Errors here that stop improving are therefore **not** evidence of a converged or
unconvergeable propagator. Reading them that way cost this repository several
reversed conclusions; see "Performance Priority E" in `IMPROVEMENTS.md`.

Use `align_lineshape` in `benchmarks/common.py` instead: scan a detuning window,
compare whole curves, and report the best-fit effective detuning shift together
with the residual after alignment. That residual converges cleanly where this
metric oscillates. `benchmarks/bench_analytic.py` scores against closed-form
solutions and needs no reference at all.

Convergence depends on both the detuning and the coupling strength, and not
monotonically in either, so a single parameter point cannot certify a step
count. The batch is therefore a grid: `intensity_prefactors` scales intensity,
so Rabi scales as its square root.

The multitone path deserves particular attention because
`scripts/analyze_spa2_bg_feature.py` doubles `N_steps` whenever the RC
background is included, and because its beat frequency varies across the scan
(the RC source is a fixed oscillator, so its detuning column is zeros while the
SPA2 columns carry the scan).

Errors are reported against the largest `N_steps` in the list. Note that this
underestimates the true error as N approaches the reference: if the improvement
ratio between successive N is well below the order of the method, the reference
is itself unconverged and a larger one is needed.

Examples:

    python benchmarks/bench_convergence.py --multitone --n-steps 10000 20000 40000 80000
    python benchmarks/bench_convergence.py --n-steps 2000 4000 8000 --output results/conv.json --csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    REPO_ROOT,
    build_spa2_setup,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)

DEFAULT_DETUNINGS_MHZ = [-2.0, -1.0, 0.0, 0.5, 1.5]
DEFAULT_PREFACTORS = [1.0, 2.0, 4.0, 8.0, 16.0]


def build_multitone_setup(*, bg_fraction: float, rc_bg_fraction: float, rc_offset_mhz: float):
    """SPA2 + SPA background + RC background, all on the same excited-J manifold."""
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import analyze_spa2_bg_feature as bg

    setup = build_spa2_setup()
    spa_bg = bg.make_bg_field(
        setup, region="right", polarization="z", bg_fraction=bg_fraction
    )
    rc_bg, info = bg.make_rc_bg_field(
        setup, bg_fraction=rc_bg_fraction, rc_offset_mhz=rc_offset_mhz
    )
    setup["microwave_fields"] = [setup["microwave_fields"][0], spa_bg, rc_bg]
    setup["rc_info"] = info
    return setup


def make_grid(detunings_mhz, prefactors, n_fields: int, multitone: bool):
    det = np.array([d * 1e6 for d in detunings_mhz for _ in prefactors], dtype=float)
    pref = np.array([p for _ in detunings_mhz for p in prefactors], dtype=float)

    if multitone:
        # Matches analyze_spa2_bg_feature.py: the RC field's detuning column is
        # zeros, since scanning the SPA2 frequency does not move a fixed RC
        # source. This makes the beat frequency vary across the scan, which is
        # what sets the required timestep.
        det_b = np.column_stack([det, det, np.zeros_like(det)])
        pref_b = np.column_stack([np.ones_like(pref), pref, pref])
    else:
        det_b = np.column_stack([det] * n_fields)
        pref_b = np.column_stack([np.ones_like(pref)] + [pref] * (n_fields - 1))

    labels = [
        f"det={d:+.1f}MHz pref={p:g}" for d in detunings_mhz for p in prefactors
    ]
    return det_b, pref_b, labels


def run_one(setup, det_b, pref_b, n_steps: int, multitone: bool, workers: int,
            propagator: str = "frozen", time_sampling: str = "mid"):
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
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        workers=workers,
        allow_multitone_same_manifold=multitone,
        propagator=propagator,
        time_sampling=time_sampling,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-steps", type=int, nargs="+", default=[2000, 4000, 8000])
    parser.add_argument("--multitone", action="store_true")
    parser.add_argument("--detunings-mhz", type=float, nargs="+", default=DEFAULT_DETUNINGS_MHZ)
    parser.add_argument("--prefactors", type=float, nargs="+", default=DEFAULT_PREFACTORS)
    parser.add_argument("--bg-fraction", type=float, default=1 / 35)
    parser.add_argument("--rc-bg-fraction", type=float, default=1 / 35)
    parser.add_argument("--rc-offset-mhz", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    n_steps_list = sorted(set(args.n_steps))
    if len(n_steps_list) < 2:
        parser.error("--n-steps needs at least two values to compare")

    print("building setup ...", flush=True)
    if args.multitone:
        setup = build_multitone_setup(
            bg_fraction=args.bg_fraction,
            rc_bg_fraction=args.rc_bg_fraction,
            rc_offset_mhz=args.rc_offset_mhz,
        )
    else:
        setup = build_spa2_setup()

    n_fields = len(setup["microwave_fields"])
    det_b, pref_b, labels = make_grid(
        args.detunings_mhz, args.prefactors, n_fields, args.multitone
    )
    print(f"grid: {len(labels)} points, {n_fields} microwave fields", flush=True)

    probs: dict[int, np.ndarray] = {}
    times: dict[int, float] = {}
    import time

    for n_steps in n_steps_list:
        start = time.perf_counter()
        res = run_one(setup, det_b, pref_b, n_steps, args.multitone, args.workers)
        times[n_steps] = time.perf_counter() - start
        probs[n_steps] = np.asarray(res.probabilities_final)
        print(f"  N_steps={n_steps:>7}  {times[n_steps]:8.2f}s", flush=True)

    ref_n = n_steps_list[-1]
    ref = probs[ref_n]

    if args.multitone:
        offset = setup["rc_info"]["rc_offset_from_spa2_center_mhz"]
        print()
        print(f"RC offset from SPA2 centre: {offset:+.3f} MHz")
        print("beat frequency per detuning (sets the required dt):")
        for d in args.detunings_mhz:
            print(f"  det={d:+.1f} MHz  ->  beat = {offset - d:+.3f} MHz")

    print()
    print(f"errors against N_steps={ref_n}")
    print(f"{'N_steps':>8}{'max abs err':>14}{'ratio':>9}   worst cell")
    rows: list[dict[str, Any]] = []
    prev_err = None
    for n_steps in n_steps_list[:-1]:
        diff = np.abs(probs[n_steps] - ref)
        per_cell = diff.reshape(diff.shape[0], -1).max(axis=1)
        worst = int(np.argmax(per_cell))
        err = float(diff.max())
        ratio = (prev_err / err) if prev_err else float("nan")
        print(f"{n_steps:>8}{err:>14.3e}{ratio:>9.2f}   {labels[worst]}")
        rows.append(
            result_row(
                "convergence",
                n_steps=n_steps,
                reference_n_steps=ref_n,
                max_abs_error=err,
                improvement_ratio=ratio,
                worst_cell=labels[worst],
                time_s=times[n_steps],
            )
        )
        prev_err = err

    print()
    print("per-cell max error:")
    header = "cell".ljust(24) + "".join(f"{n:>13}" for n in n_steps_list[:-1])
    print(header)
    for idx, label in enumerate(labels):
        cells = "".join(
            f"{np.abs(probs[n][idx] - ref[idx]).max():>13.2e}" for n in n_steps_list[:-1]
        )
        print(f"{label:<24}{cells}")

    print()
    print("A second-order method should improve ~4x per halving of dt. A ratio well")
    print("below that means the reference is itself unconverged and the errors above")
    print("are lower bounds, not estimates.")

    rows.append(result_row("reference", n_steps=ref_n, time_s=times[ref_n]))
    payload = make_payload(
        benchmark="convergence",
        config={
            "n_steps": n_steps_list,
            "multitone": args.multitone,
            "detunings_mhz": args.detunings_mhz,
            "prefactors": args.prefactors,
            "rc_offset_mhz": args.rc_offset_mhz,
            "bg_fraction": args.bg_fraction,
            "rc_bg_fraction": args.rc_bg_fraction,
            "workers": args.workers,
            "grid_points": len(labels),
        },
        results=rows,
        include_gpu_env=False,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "convergence"),
    )


if __name__ == "__main__":
    main()
