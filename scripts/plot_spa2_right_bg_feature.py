from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarks.common import make_detunings, utc_now_iso  # noqa: E402
from analyze_spa2_bg_feature import scan_case  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


def bg_fraction_for_polarization(polarization: str) -> float:
    if polarization == "z":
        return 1 / 35
    if polarization == "zy":
        return 1 / 30
    raise ValueError("polarization must be 'z' or 'zy'")


def plot_polarization(
    ax: plt.Axes,
    *,
    polarization: str,
    detunings_hz: np.ndarray,
    n_steps: int,
    workers: int,
    parallel_backend: str,
    include_rc_bg: bool,
    rc_offset_mhz: float,
    rc_bg_fraction: float | None,
    time_sampling: str = "mid",
    propagator: str = "frozen",
) -> list[dict[str, Any]]:
    no_bg = scan_case(
        region="none",
        polarization=None,
        bg_fraction=0.0,
        n_steps=n_steps,
        detunings_hz=detunings_hz,
        workers=workers,
        parallel_backend=parallel_backend,
        time_sampling=time_sampling,
        propagator=propagator,
    )
    right_bg = scan_case(
        region="right",
        polarization=polarization,
        bg_fraction=bg_fraction_for_polarization(polarization),
        n_steps=n_steps,
        detunings_hz=detunings_hz,
        workers=workers,
        parallel_backend=parallel_backend,
        time_sampling=time_sampling,
        propagator=propagator,
    )
    cases = [no_bg, right_bg]
    right_bg_rc = None
    rc_only = None
    if include_rc_bg:
        rc_only = scan_case(
            region="right",
            polarization=polarization,
            bg_fraction=0.0,
            n_steps=n_steps,
            detunings_hz=detunings_hz,
            workers=workers,
            parallel_backend=parallel_backend,
            time_sampling=time_sampling,
            propagator=propagator,
            include_rc_bg=True,
            rc_offset_mhz=rc_offset_mhz,
            rc_bg_fraction=(
                bg_fraction_for_polarization(polarization)
                if rc_bg_fraction is None
                else rc_bg_fraction
            ),
            case_label=f"{polarization}_rc_only",
        )
        right_bg_rc = scan_case(
            region="right",
            polarization=polarization,
            bg_fraction=bg_fraction_for_polarization(polarization),
            n_steps=n_steps,
            detunings_hz=detunings_hz,
            workers=workers,
            parallel_backend=parallel_backend,
            time_sampling=time_sampling,
            propagator=propagator,
            include_rc_bg=True,
            rc_offset_mhz=rc_offset_mhz,
            rc_bg_fraction=rc_bg_fraction,
        )
        cases.append(rc_only)
        cases.append(right_bg_rc)

    detunings_mhz = np.asarray(no_bg["detuning_mhz"], dtype=float)
    no_bg_depletion = np.asarray(no_bg["depletion"], dtype=float)
    no_bg_accumulation = np.asarray(no_bg["target"], dtype=float)
    bg_depletion = np.asarray(right_bg["depletion"], dtype=float)
    bg_accumulation = np.asarray(right_bg["target"], dtype=float)
    if right_bg_rc is not None:
        rc_bg_depletion = np.asarray(right_bg_rc["depletion"], dtype=float)
        rc_bg_accumulation = np.asarray(right_bg_rc["target"], dtype=float)

    if right_bg_rc is not None:
        rc_only_depletion = np.asarray(rc_only["depletion"], dtype=float)
        rc_only_accumulation = np.asarray(rc_only["target"], dtype=float)
        ax.plot(
            detunings_mhz,
            rc_only_depletion,
            color="tab:green",
            lw=2,
            label="depletion, RC only",
        )
        ax.plot(
            detunings_mhz,
            rc_only_accumulation,
            color="tab:green",
            lw=2,
            ls="--",
            label="accumulation, RC only",
        )
        ax.plot(
            detunings_mhz,
            rc_bg_depletion,
            color="tab:red",
            lw=2,
            label="depletion, right BG + RC",
        )
        ax.plot(
            detunings_mhz,
            rc_bg_accumulation,
            color="tab:red",
            lw=2,
            ls="--",
            label="accumulation, right BG + RC",
        )
    ax.plot(
        detunings_mhz,
        no_bg_depletion,
        color="tab:blue",
        lw=2,
        label="depletion, no BG",
        zorder=5,
    )
    ax.plot(
        detunings_mhz,
        no_bg_accumulation,
        color="tab:blue",
        lw=2,
        ls="--",
        label="accumulation, no BG",
        zorder=5,
    )
    ax.plot(
        detunings_mhz,
        bg_depletion,
        color="tab:orange",
        lw=2,
        label="depletion, right BG",
        zorder=6,
    )
    ax.plot(
        detunings_mhz,
        bg_accumulation,
        color="tab:orange",
        lw=2,
        ls="--",
        label="accumulation, right BG",
        zorder=6,
    )
    ax.axvline(0.0, color="k", lw=1, alpha=0.35)
    ax.axvspan(0.35, 0.6, color="tab:red", alpha=0.08)
    ax.set_title(f"SPA2 right-side BG, {polarization.upper()} polarization")
    ax.set_xlabel("detuning (MHz)")
    ax.set_ylabel("efficiency")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    return cases


def write_plot_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        for idx, detuning_mhz in enumerate(case["detuning_mhz"]):
            row = {
                "case": case["case"],
                "region": case["region"],
                "polarization": case["polarization"],
                "bg_fraction": case["bg_fraction"],
                "detuning_mhz": detuning_mhz,
                "depletion": case["depletion"][idx],
                "target": case["target"][idx],
                "transferred": case["transferred"][idx],
                "include_rc_bg": case.get("include_rc_bg", False),
                "rc_bg_fraction": case.get("rc_bg_fraction"),
                "rc_zero_e_frequency_hz": case.get("rc_zero_e_frequency_hz"),
                "rc_frequency_hz": case.get("rc_frequency_hz"),
                "rc_offset_mhz": case.get("rc_offset_mhz"),
                "rc_offset_from_spa2_center_mhz": case.get("rc_offset_from_spa2_center_mhz"),
            }
            monitors = case.get("monitor_probabilities")
            if monitors is not None:
                for monitor_idx, probability in enumerate(monitors[idx]):
                    row[f"monitor_{monitor_idx}"] = probability
            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SPA2 depletion/accumulation with and without right-side background field."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        help="Time steps per trajectory. Defaults to 10000 for static scans, 20000 with --include-rc-bg.",
    )
    parser.add_argument("--n-detunings", type=int, default=33)
    parser.add_argument("--detuning-lo-mhz", type=float, default=-2.0)
    parser.add_argument("--detuning-hi-mhz", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", default="loky")
    parser.add_argument("--include-rc-bg", action="store_true")
    parser.add_argument("--rc-offset-mhz", type=float, default=-7.4)
    parser.add_argument("--rc-bg-fraction", type=float)
    # Mirrors analyze_spa2_bg_feature.py; see the note there.
    parser.add_argument("--time-sampling", default="mid", choices=("mid", "left"))
    parser.add_argument(
        "--propagator", default="frozen", choices=("frozen", "magnus"),
        help="With --include-rc-bg the beat rotates within a step and 'frozen' "
             "does not resolve it.",
    )
    parser.add_argument(
        "--polarization",
        choices=("z", "zy", "both"),
        default="both",
        help="Background polarization to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/spa2_right_bg_feature.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--data-json",
        type=Path,
        help="Output JSON path for the scan data used to make the plot.",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        help="Output CSV path for the scan data used to make the plot.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.n_steps is None:
        args.n_steps = 20000 if args.include_rc_bg else 10000

    detunings_hz = make_detunings(
        args.n_detunings,
        lo_mhz=args.detuning_lo_mhz,
        hi_mhz=args.detuning_hi_mhz,
    )
    polarizations = ["z", "zy"] if args.polarization == "both" else [args.polarization]

    fig, axes = plt.subplots(
        len(polarizations),
        1,
        figsize=(8, 4.8 * len(polarizations)),
        squeeze=False,
        sharex=True,
    )
    plot_cases: list[dict[str, Any]] = []
    for ax, polarization in zip(axes[:, 0], polarizations):
        plot_cases.extend(
            plot_polarization(
                ax,
                polarization=polarization,
                detunings_hz=detunings_hz,
                n_steps=args.n_steps,
                workers=args.workers,
                parallel_backend=args.parallel_backend,
        time_sampling=args.time_sampling,
        propagator=args.propagator,
                include_rc_bg=args.include_rc_bg,
                rc_offset_mhz=args.rc_offset_mhz,
                rc_bg_fraction=args.rc_bg_fraction,
            )
        )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")

    data_json = args.data_json or args.output.with_suffix(".json")
    data_csv = args.data_csv or args.output.with_suffix(".csv")
    payload = {
        "generated_at": utc_now_iso(),
        "plot": str(args.output),
        "config": {
            "n_steps": args.n_steps,
            "n_detunings": args.n_detunings,
            "detuning_lo_mhz": args.detuning_lo_mhz,
            "detuning_hi_mhz": args.detuning_hi_mhz,
            "polarization": args.polarization,
            "workers": args.workers,
            "parallel_backend": args.parallel_backend,
            "time_sampling": args.time_sampling,
            "propagator": args.propagator,
            "include_rc_bg": args.include_rc_bg,
            "rc_offset_mhz": args.rc_offset_mhz,
            "rc_bg_fraction": args.rc_bg_fraction,
        },
        "scan_results": plot_cases,
    }
    data_json.parent.mkdir(parents=True, exist_ok=True)
    data_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_plot_csv(data_csv, plot_cases)
    print(f"wrote {data_json}")
    print(f"wrote {data_csv}")


if __name__ == "__main__":
    main()
