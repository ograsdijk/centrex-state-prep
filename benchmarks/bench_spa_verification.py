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
    build_spa_cascade_setup,
    cascade_readout_identities,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase0a", action="store_true")
    ap.add_argument("--phase0b", action="store_true")
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

    if not (args.phase0a or args.phase0b):
        ap.error("choose at least one of --phase0a / --phase0b")

    print("building cascade setup ...", flush=True)
    setup = build_spa_cascade_setup()
    rows: list[dict] = []
    if args.phase0a:
        rows += phase0a(setup, args)
    if args.phase0b:
        rows += phase0b(setup, args)

    payload = make_payload(
        benchmark="spa_verification",
        config={
            "identity_steps": args.identity_steps,
            "samplings": args.samplings,
            "norm_steps": args.norm_steps,
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
