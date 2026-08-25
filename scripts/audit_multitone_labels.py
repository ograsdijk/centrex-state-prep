"""Close the multitone grid audit left open in IMPROVEMENTS.md.

The three unverified questions there were: do the other 24 grid cells swap
labels, are the published figures affected, and is `transferred` as safe as it
looked. All three were unanswerable because the check available at the time
compared `probabilities_final` elementwise, which is exactly the comparison a
label swap confounds.

With quantum-number selection (`state_prep.utils.eigenstate_quantum_numbers`)
the comparison becomes decisive. This runs the 5x5 grid at several step counts
and reports, side by side:

  * the elementwise error, the old confounded metric, kept for continuity;
  * the observables read through a tracked `V_ini` index, i.e. what the analysis
    used to do;
  * the same observables read by quantum numbers, which a swap cannot affect.

A swap shows up as a large elementwise error, and a change in the tracked-index
reading, while the quantum-number values hold steady.

    .\\.venv\\Scripts\\python.exe scripts/audit_multitone_labels.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "benchmarks", REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bench_convergence import (  # noqa: E402
    DEFAULT_DETUNINGS_MHZ,
    DEFAULT_PREFACTORS,
    build_multitone_setup,
    make_grid,
    run_one,
)
from analyze_spa2_bg_feature import readout_identity  # noqa: E402
from state_prep.approximate_states import J2_triplet_0  # noqa: E402
from state_prep.utils import find_max_overlap_idx, select_eigenstate  # noqa: E402

INITIAL_IDX = 2


def observables(result, setup, identities):
    """Both readings of depletion / target / transferred, plus the indices used."""
    qn_table = result.final_quantum_numbers()
    probs = np.asarray(result.probabilities_final)[:, INITIAL_IDX, :]
    QN = setup["hamiltonian"].QN

    indices = {"selected": {}, "tracked": {}}
    for name, (identity, state) in identities.items():
        indices["selected"][name] = int(select_eigenstate(qn_table, identity))
        indices["tracked"][name] = int(
            find_max_overlap_idx(state.state_vector(QN), result.V_ini)
        )

    def read(kind):
        idx = indices[kind]
        target = probs[:, idx["target"]]
        monitor_keys = [k for k in idx if k.startswith("monitor")]
        monitors = probs[:, [idx[k] for k in monitor_keys]]
        return {
            "depletion": 1.0 - probs[:, idx["initial"]],
            "target": target,
            "transferred": target + monitors.sum(axis=1),
        }

    return indices, read("selected"), read("tracked"), probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-steps", type=int, nargs="+", default=[20000, 40000, 80000])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bg-fraction", type=float, default=1 / 35)
    parser.add_argument("--rc-bg-fraction", type=float, default=1 / 35)
    parser.add_argument("--rc-offset-mhz", type=float, default=0.0)
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "results" / "multitone_label_audit.json")
    )
    args = parser.parse_args()

    setup = build_multitone_setup(
        bg_fraction=args.bg_fraction,
        rc_bg_fraction=args.rc_bg_fraction,
        rc_offset_mhz=args.rc_offset_mhz,
    )
    det_b, pref_b, labels = make_grid(
        DEFAULT_DETUNINGS_MHZ, DEFAULT_PREFACTORS, len(setup["microwave_fields"]), True
    )

    # What each state becomes at readout, from the DC ramp with microwaves off.
    print("resolving readout identities (microwaves off)...", flush=True)
    initial_state = setup["initial_states"][INITIAL_IDX]
    identities = {
        "initial": (readout_identity(setup, initial_state), initial_state),
        "target": (readout_identity(setup, J2_triplet_0), J2_triplet_0),
    }
    for i, state in enumerate(setup["monitor_states"]):
        identities[f"monitor{i}"] = (readout_identity(setup, state), state)
    for name, (identity, _) in identities.items():
        print(
            f"  {name:9s} J={identity['J']:.2f} F1={identity['F1']:.2f} "
            f"F={identity['F']:.2f} mF={identity['mF']:+.2f} "
            f"(pop {identity['population']:.5f})",
            flush=True,
        )
    print(flush=True)

    runs = {}
    for n_steps in args.n_steps:
        start = time.perf_counter()
        result = run_one(setup, det_b, pref_b, n_steps, True, args.workers)
        indices, selected, tracked, probs = observables(result, setup, identities)
        runs[n_steps] = {
            "indices": indices,
            "selected": selected,
            "tracked": tracked,
            "probs": probs,
        }
        disagree = [
            k for k in indices["selected"] if indices["selected"][k] != indices["tracked"][k]
        ]
        print(
            f"N_steps={n_steps:>7}  {time.perf_counter() - start:8.1f}s  "
            f"tracked indices disagreeing with quantum numbers: {disagree or 'none'}",
            flush=True,
        )

    ref = args.n_steps[-1]
    print(f"\nreferenced to N_steps={ref}")
    print(
        f"{'N_steps':>8}{'elementwise':>14}{'depletion(qn)':>16}{'depletion(idx)':>17}"
        f"{'transferred(qn)':>18}{'transferred(idx)':>19}"
    )
    rows = []
    for n_steps in args.n_steps[:-1]:
        a, b = runs[n_steps], runs[ref]
        row = {
            "n_steps": n_steps,
            "elementwise": float(np.abs(a["probs"] - b["probs"]).max()),
            "depletion_qn": float(
                np.abs(a["selected"]["depletion"] - b["selected"]["depletion"]).max()
            ),
            "depletion_idx": float(
                np.abs(a["tracked"]["depletion"] - b["tracked"]["depletion"]).max()
            ),
            "transferred_qn": float(
                np.abs(a["selected"]["transferred"] - b["selected"]["transferred"]).max()
            ),
            "transferred_idx": float(
                np.abs(a["tracked"]["transferred"] - b["tracked"]["transferred"]).max()
            ),
        }
        rows.append(row)
        print(
            f"{n_steps:>8}{row['elementwise']:>14.3e}{row['depletion_qn']:>16.3e}"
            f"{row['depletion_idx']:>17.3e}{row['transferred_qn']:>18.3e}"
            f"{row['transferred_idx']:>19.3e}",
            flush=True,
        )

    print("\nper-cell quantum-number depletion across step counts (should be flat)")
    print("cell".ljust(24) + "".join(f"{n:>13}" for n in args.n_steps))
    worst_qn = 0.0
    worst_tracked = 0.0
    for i, label in enumerate(labels):
        qn_vals = [runs[n]["selected"]["depletion"][i] for n in args.n_steps]
        tr_vals = [runs[n]["tracked"]["depletion"][i] for n in args.n_steps]
        worst_qn = max(worst_qn, max(qn_vals) - min(qn_vals))
        worst_tracked = max(worst_tracked, max(tr_vals) - min(tr_vals))
        print(f"{label:<24}" + "".join(f"{v:>13.6f}" for v in qn_vals))

    print(f"\nlargest spread of depletion across step counts:")
    print(f"  by quantum numbers : {worst_qn:.3e}")
    print(f"  by tracked index   : {worst_tracked:.3e}")
    print(
        "\nA swap gives a large elementwise error and a large tracked-index spread "
        "with a flat quantum-number column. If both spreads are small, no cell in "
        "this grid swapped at these step counts."
    )

    payload = {
        "n_steps": args.n_steps,
        "labels": labels,
        "rows": rows,
        "largest_depletion_spread_qn": worst_qn,
        "largest_depletion_spread_tracked": worst_tracked,
        "indices": {str(n): runs[n]["indices"] for n in args.n_steps},
        "depletion_qn": {
            str(n): runs[n]["selected"]["depletion"].tolist() for n in args.n_steps
        },
        "depletion_tracked": {
            str(n): runs[n]["tracked"]["depletion"].tolist() for n in args.n_steps
        },
        "transferred_qn": {
            str(n): runs[n]["selected"]["transferred"].tolist() for n in args.n_steps
        },
        "transferred_tracked": {
            str(n): runs[n]["tracked"]["transferred"].tolist() for n in args.n_steps
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
