"""Observed convergence order of each propagator, measured without a reference.

Every order estimate made so far scored against a fixed reference, and every
reference available here carries an error comparable to what is being measured
once `N_steps` is large. Worse, a reference built from a finer run of the *same*
scheme flatters that scheme through correlated error structure -- that mistake
inverted a conclusion earlier in this work, where `N_steps=128000` on a uniform
grid turned out to sit `2.5e-04` from converged and made a graded grid look
worse than it is.

Self-convergence triples avoid the problem entirely. For solutions at `N`, `2N`
and `4N`,

    p = log2( ||u(N) - u(2N)|| / ||u(2N) - u(4N)|| )

which needs no converged answer at all: `~1` per doubling means first order,
`~2` means second, and a number that wanders means there is no clean order to
extrapolate from.

The observable is a lineshape, not a population at a fixed detuning.
`d(population)/d(detuning)` reaches `6e-03` per kHz on this system, so a
pointwise comparison measures the slope of a resonance rather than
discretisation error and refuses to converge. See `align_lineshape` in
`common.py`.

The standing question this is built to answer: `exp(-i H(t + dt/2) dt)` should
be second order and measures first. Either the earlier estimates were
reference-contaminated, or a first-order term is unaccounted for -- and if it is
the latter, no higher-order propagator can help until it is found.

    python benchmarks/bench_propagator_order.py --base-n 1000 2000
    python benchmarks/bench_propagator_order.py --variants left mid --output results/order.json --csv
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import numpy as np

from bench_magnus_propagator import make_magnus_loop
from common import (
    align_lineshape,
    build_spa2_setup,
    convergence_order,
    make_payload,
    parse_js,
    resolve_csv_path,
    result_row,
    write_results,
)
from state_prep import Simulator

SHIPPED = Simulator._time_evolve_mu_batched_shared_slow

VARIANTS = ("left", "mid", "magnus-left", "magnus-mid")

# The SPA2 initial state, J=1 triplet mF=0.
INITIAL_IDX = 2


def readout_identities(setup) -> tuple[dict, dict]:
    """Quantum numbers of source and target at readout, by propagating DC only."""
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import analyze_spa2_bg_feature as bg

    from state_prep.approximate_states import J1_triplet_0, J2_triplet_0

    return bg.readout_identity(setup, J1_triplet_0), bg.readout_identity(
        setup, J2_triplet_0
    )


def make_variants(time_samplings=("left", "mid")) -> dict[str, tuple[Any, str]]:
    """`{name: (loop_impl_or_None, time_sampling)}`; None means the shipped loop."""
    variants: dict[str, tuple[Any, str]] = {}
    for sampling in time_samplings:
        variants[sampling] = (None, sampling)
        variants[f"magnus-{sampling}"] = (make_magnus_loop(), sampling)
    return variants


def curve_fn(simulator, setup, det, pref, source_id, target_id, args) -> Callable:
    """Return `f(impl, time_sampling, n_steps) -> (lineshape, seconds)`."""
    keys = ("J", "F1", "F", "mF")
    initial = setup["initial_states"][INITIAL_IDX]

    def run(impl, time_sampling: str, n_steps: int):
        if impl is not None:
            Simulator._time_evolve_mu_batched_shared_slow = impl
        try:
            start = time.perf_counter()
            result = simulator.run_microwave_scan(
                detunings_hz=det,
                intensity_prefactors=pref,
                N_steps=n_steps,
                monitor_states=setup["monitor_states"],
                store_final_probabilities=True,
                store_final_monitor_probabilities=True,
                progress=False,
                eig_backend=args.eig_backend,
                time_sampling=time_sampling,
            )
            elapsed = time.perf_counter() - start
        finally:
            Simulator._time_evolve_mu_batched_shared_slow = SHIPPED
        remaining = result.population(initial, **{k: source_id[k] for k in keys})
        return np.asarray(remaining, dtype=float), elapsed

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--js", default="0,1,2,3")
    parser.add_argument(
        "--base-n",
        type=int,
        nargs="+",
        default=[1000, 2000],
        help="each value N yields a triple (N, 2N, 4N)",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--lo-mhz", type=float, default=-1.10)
    parser.add_argument("--hi-mhz", type=float, default=-0.90)
    parser.add_argument("--points", type=int, default=15)
    parser.add_argument("--eig-backend", default="zheevd", choices=["zheevd", "numpy"])
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    js = parse_js(args.js)
    print(f"building setup js={js} ...", flush=True)
    setup = build_spa2_setup(js=js)
    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        setup["microwave_fields"],
    )
    source_id, target_id = readout_identities(setup)

    mhz = np.linspace(args.lo_mhz, args.hi_mhz, args.points)
    khz = mhz * 1e3
    det = np.column_stack([mhz * 1e6] * len(setup["microwave_fields"]))
    pref = np.ones((args.points, len(setup["microwave_fields"])))
    print(
        f"window {args.lo_mhz:+.2f}..{args.hi_mhz:+.2f} MHz, {args.points} points "
        f"(partial-transfer regime: unsaturated, so the observable can resolve error)",
        flush=True,
    )

    run = curve_fn(simulator, setup, det, pref, source_id, target_id, args)
    variants = make_variants()

    rows: list[dict[str, Any]] = []
    print()
    print(f"{'variant':>13}{'N':>8}{'2N':>8}{'4N':>8}{'|u(N)-u(2N)|':>15}"
          f"{'|u(2N)-u(4N)|':>16}{'order p':>10}")

    for base in args.base_n:
        counts = (base, 2 * base, 4 * base)
        for name in args.variants:
            impl, sampling = variants[name]
            curves = {}
            seconds = {}
            for n in counts:
                curves[n], seconds[n] = run(impl, sampling, n)
            first = float(np.linalg.norm(curves[counts[0]] - curves[counts[1]]))
            second = float(np.linalg.norm(curves[counts[1]] - curves[counts[2]]))
            order = convergence_order(*[curves[n] for n in counts])
            # Residual of the coarsest against the finest of this triple, aligned,
            # to separate a bodily shift of the lineshape from a change of shape.
            shift, residual = align_lineshape(curves[counts[0]], curves[counts[2]], khz)
            print(
                f"{name:>13}{counts[0]:>8}{counts[1]:>8}{counts[2]:>8}"
                f"{first:>15.3e}{second:>16.3e}{order:>10.2f}",
                flush=True,
            )
            rows.append(
                result_row(
                    name,
                    base_n=base,
                    n_steps=list(counts),
                    diff_coarse=first,
                    diff_fine=second,
                    order=order,
                    shift_khz=shift,
                    residual=residual,
                    seconds=[seconds[n] for n in counts],
                )
            )

    print()
    print("p ~ 1 => first order, p ~ 2 => second order.")
    print("A variant whose p differs between triples has no clean order, and")
    print("Richardson extrapolation on it will not work.")

    if args.output or args.csv is not None:
        payload = make_payload(
            benchmark="propagator_order",
            config={
                "js": js,
                "base_n": args.base_n,
                "variants": args.variants,
                "window_mhz": [args.lo_mhz, args.hi_mhz],
                "points": args.points,
                "eig_backend": args.eig_backend,
                "source_identity": {k: source_id[k] for k in ("J", "F1", "F", "mF")},
            },
            results=rows,
            include_gpu_env=False,
        )
        write_results(
            payload,
            output=args.output,
            csv_path=resolve_csv_path(args.output, args.csv, "propagator_order"),
        )


if __name__ == "__main__":
    main()
