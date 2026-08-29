"""Timestep study for the SPA singlet-vs-triplet verification example.

The example (`examples/SPA/Experimental verification/`) reports every hyperfine
chain transferring to `>= 0.9991`, with a singlet-vs-triplet **spread** of only
`1.7e-04`. That spread is the physics claim. Every number in it was produced at
`N_steps = 10_000` with the shipped defaults, and nothing in the notebook passes
`step_density`, `time_sampling` or `propagator`.

Three things this measures that the example does not:

1. **The spread's own convergence.** The one committed check gives
   `max |P(10k) - P(20k)| = 6.996e-04`, four times *larger* than the `1.7e-04`
   being reported. The errors are plausibly common-mode across sublevels and
   cancel in the difference -- but that is an assumption. Converging the
   individual transfers does not establish the spread.
2. **AP2.** The committed check covers AP1 only, on the `manifold` observable (a
   sum over the manifold, smoother and converging sooner than the `matched`
   population actually reported). AP2 carries the narrow feature, the hard
   positive cliff and the report's counter-example.
3. **The identification run.** `readout_identities` is upstream of every
   reported number and its convergence has never been checked.

Method notes, all learned the hard way elsewhere in this repo:

- Score `matched` (quantum numbers at readout), never `tracked` (a `t=0` index).
  The notebook records the J=2 singlet moving `32 -> 29` at `80_000`.
- Never compare populations at a fixed detuning across step counts near a steep
  feature: that measures the slope of a resonance. For the AP2 cliff use
  `align_lineshape`.
- Self-convergence triples need no reference at all, and a reference drawn from
  a finer run of the *same* scheme flatters that scheme.
- This cascade drives `J=0->1` and `J=1->2`, two tones on *different* manifolds,
  so there is no beat phase and the `beat*dt` aliasing that invalidated the SPA2
  RC analysis cannot occur here.

    python benchmarks/bench_spa_verification.py --phase0a
    python benchmarks/bench_spa_verification.py --phase0b
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import (
    align_lineshape,
    build_spa_cascade_setup,
    cascade_readout_identities,
    cascade_transfer_observables,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)
from state_prep import (
    build_time_grid,
    field_variation_density,
    grading_ceiling,
    magnus_step_norms,
)

IDENTITY_KEYS = ("J", "F1", "F", "mF")


def phase0a(setup, args) -> list[dict]:
    """Are the readout identities stable in `N_steps` and `time_sampling`?

    Everything downstream matches against these, so a change here is not an error
    that shows up as noise -- it silently redefines what is being reported.
    """
    print("Phase 0a: readout identities vs N_steps and sampling")
    print("(microwaves off; run() -> _time_evolve, where Magnus is a no-op)\n")
    rows: list[dict] = []
    for stage, targets in (("j1", setup["targets_j1"]), ("j2", setup["targets_j2"])):
        print(f"  {stage}:")
        for sampling in args.samplings:
            for n in args.identity_steps:
                t0 = time.perf_counter()
                ids = cascade_readout_identities(
                    setup, targets, n_steps=n, time_sampling=sampling
                )
                # `+ 0.0` normalises negative zero. `mF` comes back as -0.0 from
                # some runs and 0.0 from others, which is the same number but a
                # different string, and comparing the rendered key made a stable
                # identity look like it moved.
                key = tuple(
                    tuple(round(d[k], 4) + 0.0 for k in IDENTITY_KEYS) for d in ids
                )
                rows.append(
                    result_row(
                        "identity",
                        stage=stage,
                        n_steps=n,
                        time_sampling=sampling,
                        identities=str(key),
                        seconds=time.perf_counter() - t0,
                    )
                )
                rendered = "  ".join(
                    "(" + ",".join(f"{v:g}" for v in t) + ")" for t in key
                )
                print(f"    {sampling:>4} N={n:<7} {rendered}")
        seen = {r["identities"] for r in rows if r.get("stage") == stage}
        verdict = "STABLE" if len(seen) == 1 else f"*** MOVES ({len(seen)} distinct) ***"
        print(f"    -> {verdict}\n")
    return rows


def phase0b(setup, args) -> list[dict]:
    """Geometry only: can grading pay here, and does it endanger Magnus?"""
    print("Phase 0b: grading ceiling and Magnus step norms (no propagation)\n")
    hamiltonian = setup["hamiltonian"]
    H_slow_t = hamiltonian.get_H_t_func()
    T = float(setup["trajectory"].get_T())
    trajectory = setup["trajectory"]
    muw = [
        field.get_H_t_func(trajectory.R_t, hamiltonian.QN)
        for field in setup["microwave_fields"]
    ]

    rows: list[dict] = []
    density = field_variation_density(H_slow_t, muw)
    probes = np.linspace(0.0, T, 400)
    d = np.asarray(density(probes), dtype=float)
    dyn_range = float(d.max() / max(d.min(), 1e-300))
    note = "  (below ~10x: a grading comparison is not meaningful)" if dyn_range < 10 else ""
    print(f"  density dynamic range : {dyn_range:.4g}{note}")

    for order in (1, 2):
        ceiling = grading_ceiling(H_slow_t, muw, T=T, order=order)
        print(f"  grading ceiling p={order}  : {ceiling:.3f}x")
        rows.append(
            result_row(
                "ceiling",
                order=order,
                ceiling=ceiling,
                density_dynamic_range=dyn_range,
            )
        )
    print("  SPA2's was 1.58x, too small to pay against the cancellation a uniform")
    print("  grid enjoys. Compare before spending compute on the graded axis.\n")

    for n in args.norm_steps:
        for grid in ("uniform", "graded1", "graded2"):
            if grid == "uniform":
                t_array = np.linspace(0.0, T, n)
            else:
                t_array = build_time_grid(
                    T, n, density=density, order=1 if grid == "graded1" else 2
                )
            max_norm = float(np.max(magnus_step_norms(muw, t_array)))
            flag = "  <-- Taylor guard engages near 0.5" if max_norm > 0.4 else ""
            print(f"  N={n:<7}{grid:<9} max||A|| = {max_norm:.4f}{flag}")
            rows.append(
                result_row("magnus_norm", n_steps=n, grid=grid, max_norm_A=max_norm)
            )
    print()
    return rows


# Optima the notebook and the figures script both use.
D1_OPT, PREF1_OPT = 1.350e6, 16.156
D2_OPT, PREF2_OPT = 0.212e6, 8.254


def _scan(setup, stage, args, *, n_steps, grid, sampling, propagator, cliff=False):
    """One scan of `stage`, returning `(observables, seconds)`."""
    from state_prep import Simulator, scan_grid

    hamiltonian = setup["hamiltonian"]
    trajectory = setup["trajectory"]
    T = float(trajectory.get_T())
    mf01, mf12 = setup["microwave_fields"]

    if stage == "ap1":
        fields = [mf01]
        initial = setup["initial_states"]
        targets = setup["targets_j1"]
        j_target = 1
        det = np.linspace(D1_OPT - 2e6, D1_OPT + 2e6, args.points)
        det_b, pref_b = scan_grid(
            n_fields=1, detunings_hz=det, intensity_prefactors=PREF1_OPT
        )
    else:
        fields = [mf01, mf12]
        initial = setup["initial_states"]
        targets = setup["targets_j2"]
        j_target = 2
        det = (
            np.linspace(250e3, 380e3, args.points)
            if cliff
            else np.linspace(D2_OPT - 1e6, D2_OPT + 0.5e6, args.points)
        )
        # Only field 1 varies; AP1 is parked. Passing a 2-element prefactor array
        # here would build an outer product (`scan_grid` returns
        # `B = n_detunings * n_prefactors`) and silently run AP2 at AP1's power
        # for half the points. Matches the notebook's cells 22/23.
        det_b, pref_b = scan_grid(
            n_fields=2,
            detunings_hz=det,
            intensity_prefactors=PREF2_OPT,
            detuning_fields=[1],
            prefactor_fields=[1],
        )
        det_b[:, 0] = D1_OPT
        pref_b[:, 0] = PREF1_OPT

    density = None
    if grid != "uniform":
        muw = [f.get_H_t_func(trajectory.R_t, hamiltonian.QN) for f in fields]
        density = field_variation_density(hamiltonian.get_H_t_func(), muw)

    simulator = Simulator(
        trajectory,
        setup["electric_field"],
        setup["magnetic_field"],
        initial,
        hamiltonian,
        fields,
    )
    t0 = time.perf_counter()
    result = simulator.run_microwave_scan(
        detunings_hz=det_b,
        intensity_prefactors=pref_b,
        N_steps=n_steps,
        store_final_probabilities=True,
        progress=False,
        workers=args.workers,
        time_sampling=sampling,
        propagator=propagator,
        step_density=density,
        step_density_order=1 if grid == "graded1" else 2,
    )
    seconds = time.perf_counter() - t0
    identities = cascade_readout_identities(setup, targets)
    obs = cascade_transfer_observables(result, setup, targets, j_target, identities)
    return obs, seconds, det


def phase2(setup, args) -> list[dict]:
    """Sweep propagator x sampling x grid x N_steps, scored on matched and spread."""
    print("Phase 2: convergence sweep\n")
    print("Self-convergence triples: p = log2(|u(N)-u(2N)| / |u(2N)-u(4N)|).")
    print("No reference is involved, so no scheme is flattered by its own refinement.\n")

    rows: list[dict] = []
    for stage in args.stages:
        print(f"=== {stage}{' (cliff)' if args.cliff else ''} ===", flush=True)
        for propagator in args.propagators:
            for sampling in args.samplings:
                for grid in args.grids:
                    runs = {}
                    for n in args.n_steps:
                        obs, secs, det = _scan(
                            setup, stage, args, n_steps=n, grid=grid,
                            sampling=sampling, propagator=propagator,
                            cliff=args.cliff,
                        )
                        runs[n] = (obs, secs, det)
                    label = f"{propagator}/{sampling}/{grid}"
                    print(f"  {label}", flush=True)
                    header = f"    {'pair':<18}{'d matched':>13}{'d spread':>13}"
                    header += (f"{'shift kHz':>11}{'residual':>12}" if args.lineshape
                               else f"{'order':>8}")
                    print(header + f"{'seconds':>10}")
                    prev = None
                    for a, b in zip(args.n_steps, args.n_steps[1:]):
                        dm = float(
                            np.abs(runs[a][0]["matched"] - runs[b][0]["matched"]).max()
                        )
                        ds = float(
                            np.abs(runs[a][0]["spread"] - runs[b][0]["spread"]).max()
                        )
                        order = np.log2(prev / dm) if prev else float("nan")
                        shift = residual = float("nan")
                        if args.lineshape:
                            # Pointwise differences are ill-conditioned on AP2:
                            # at the cliff the population falls from 0.998938 to
                            # 0.000035 between adjacent points, so a sub-kHz
                            # effective shift reads as a 4e-03 difference and
                            # never converges. Separate "the curve moved" from
                            # "the curve changed shape".
                            det_khz = runs[a][2] / 1e3
                            shifts, residuals = [], []
                            for k in range(runs[a][0]["matched"].shape[1]):
                                sh, rs = align_lineshape(
                                    runs[b][0]["matched"][:, k],
                                    runs[a][0]["matched"][:, k],
                                    det_khz,
                                    max_shift_khz=args.max_shift_khz,
                                )
                                shifts.append(sh)
                                residuals.append(rs)
                            shift = max(shifts, key=abs)
                            residual = max(residuals)
                        tail = (f"{shift:>11.3f}{residual:>12.3e}" if args.lineshape
                                else f"{order:>8.2f}")
                        print(f"    {str(a)+' vs '+str(b):<18}{dm:>13.3e}{ds:>13.3e}"
                              + tail + f"{runs[b][1]:>10.1f}", flush=True)
                        rows.append(
                            result_row(
                                "converge",
                                stage=stage,
                                propagator=propagator,
                                time_sampling=sampling,
                                grid=grid,
                                n_coarse=a,
                                n_fine=b,
                                d_matched=dm,
                                d_spread=ds,
                                order=None if np.isnan(order) else float(order),
                                shift_khz=None if np.isnan(shift) else float(shift),
                                residual=None if np.isnan(residual) else float(residual),
                                seconds_fine=runs[b][1],
                                cliff=args.cliff,
                            )
                        )
                        prev = dm
                    print()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase0a", action="store_true")
    ap.add_argument("--phase0b", action="store_true")
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--stages", nargs="+", default=["ap1", "ap2"],
                    choices=["ap1", "ap2"])
    ap.add_argument("--cliff", action="store_true",
                    help="AP2 only: scan the steep +250..+380 kHz cutoff region")
    ap.add_argument("--n-steps", type=int, nargs="+",
                    default=[2500, 5000, 10000, 20000, 40000])
    ap.add_argument("--propagators", nargs="+", default=["frozen", "magnus"],
                    choices=["frozen", "magnus"])
    ap.add_argument("--grids", nargs="+", default=["uniform", "graded1"],
                    choices=["uniform", "graded1", "graded2"])
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--lineshape",
        action="store_true",
        help=(
            "score by best-fit detuning shift plus residual instead of pointwise "
            "differences. Required for AP2, whose cliff makes a pointwise "
            "comparison measure the slope of the resonance."
        ),
    )
    ap.add_argument("--max-shift-khz", type=float, default=3.0)
    ap.add_argument(
        "--identity-steps", type=int, nargs="+", default=[2500, 5000, 10000, 20000, 40000]
    )
    ap.add_argument(
        "--samplings", nargs="+", default=["mid", "left"], choices=["mid", "left"]
    )
    ap.add_argument("--norm-steps", type=int, nargs="+", default=[2500, 10000, 40000])
    ap.add_argument("--output")
    ap.add_argument("--csv", nargs="?", const="", default=None)
    args = ap.parse_args()

    if not (args.phase0a or args.phase0b or args.phase2):
        ap.error("choose at least one of --phase0a / --phase0b / --phase2")

    print("building cascade setup ...", flush=True)
    setup = build_spa_cascade_setup()
    rows: list[dict] = []
    if args.phase0a:
        rows += phase0a(setup, args)
    if args.phase0b:
        rows += phase0b(setup, args)
    if args.phase2:
        rows += phase2(setup, args)

    payload = make_payload(
        benchmark="spa_verification",
        config={
            "identity_steps": args.identity_steps,
            "samplings": args.samplings,
            "norm_steps": args.norm_steps,
            "n_steps": args.n_steps,
            "stages": args.stages,
            "propagators": args.propagators,
            "grids": args.grids,
            "points": args.points,
            "workers": args.workers,
            "cliff": args.cliff,
        },
        results=rows,
        include_gpu_env=False,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "spa_verification"),
    )


if __name__ == "__main__":
    main()
