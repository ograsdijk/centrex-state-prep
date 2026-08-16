from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarks.common import build_spa2_setup, make_detunings, utc_now_iso  # noqa: E402
from centrex_tlf import states  # noqa: E402
from state_prep.approximate_states import (  # noqa: E402
    J1_singlet,
    J1_triplet_0,
    J2_singlet,
    J2_triplet_0,
    J2_triplet_m,
    J2_triplet_p,
)
from state_prep.intensity_profiles import BackgroundField  # noqa: E402
from state_prep.microwaves import MicrowaveField, Polarization  # noqa: E402
from state_prep.simulator import Simulator  # noqa: E402
from state_prep.utils import calculate_transition_frequency, find_max_overlap_idx, vector_to_state  # noqa: E402


def make_bg_field(setup: dict[str, Any], *, polarization: str, region: str, bg_fraction: float):
    p_y = np.array([0.0, 1.0, 0.0])
    p_z = np.array([0.0, 0.0, 1.0])
    if polarization == "z":
        pol_vec = p_z
    elif polarization == "zy":
        pol_vec = p_z + 0.5 * p_y
    else:
        raise ValueError("polarization must be 'z' or 'zy'")

    def p_r(_position):
        return pol_vec / np.sqrt(np.sum(pol_vec**2))

    if region == "constant":
        lims = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    elif region == "left":
        lims = [(-1.0, 1e-5), (-1.0, 1e-5), (-1.0, 1e-5)]
    elif region == "right":
        lims = [(-1e-5, 1.0), (-1e-5, 1.0), (-1e-5, 1.0)]
    else:
        raise ValueError("region must be constant, left, or right")

    mf12 = setup["microwave_fields"][0]
    intensity = BackgroundField(lims, intensity=mf12.intensity.I_R(setup["r0"]) * bg_fraction)
    return MicrowaveField(
        1,
        2,
        intensity,
        Polarization(p_r),
        setup["frequency"],
        setup["hamiltonian"].QN,
        background_field=True,
    )


def rc_frequency_info(setup: dict[str, Any], *, rc_offset_mhz: float) -> dict[str, float]:
    hamiltonian = setup["hamiltonian"]
    H_zero_E = hamiltonian.H_EB(
        np.zeros(3),
        setup["magnetic_field"].B_R(np.zeros(3)),
    )
    f_zero = calculate_transition_frequency(
        J1_singlet,
        J2_singlet,
        H_zero_E,
        hamiltonian.QN,
    )
    f_rc = f_zero + rc_offset_mhz * 1e6
    return {
        "rc_zero_e_frequency_hz": float(f_zero),
        "rc_frequency_hz": float(f_rc),
        "rc_offset_mhz": float(rc_offset_mhz),
        "rc_offset_from_spa2_center_mhz": float((f_rc - setup["frequency"]) / 1e6),
    }


def make_rc_bg_field(setup: dict[str, Any], *, bg_fraction: float, rc_offset_mhz: float) -> tuple[MicrowaveField, dict[str, float]]:
    p_y = np.array([0.0, 1.0, 0.0])
    p_z = np.array([0.0, 0.0, 1.0])
    pol_vec = p_z + 0.4 * p_y

    def p_r(_position):
        return pol_vec / np.sqrt(np.sum(pol_vec**2))

    lims = [(-1.0, 1.0), (-1.0, 1.0), (-1e-5, 1.0)]
    mf12 = setup["microwave_fields"][0]
    info = rc_frequency_info(setup, rc_offset_mhz=rc_offset_mhz)
    intensity = BackgroundField(lims, intensity=mf12.intensity.I_R(setup["r0"]) * bg_fraction)
    field = MicrowaveField(
        1,
        2,
        intensity,
        Polarization(p_r),
        info["rc_frequency_hz"],
        setup["hamiltonian"].QN,
        background_field=True,
    )
    return field, info


def j2_states() -> list[tuple[str, Any]]:
    return [
        ("J2_triplet_0", J2_triplet_0),
        ("J2_singlet", J2_singlet),
        (
            "J2_F1=3/2_F=1_mF=0",
            states.CoupledBasisState(
                J=2,
                F1=3 / 2,
                F=1,
                mF=0,
                electronic_state=states.ElectronicState.X,
                I1=1 / 2,
                I2=1 / 2,
                Omega=0,
                P=1,
            ).transform_to_uncoupled(),
        ),
        ("J2_triplet_m", J2_triplet_m),
        ("J2_triplet_p", J2_triplet_p),
    ]


def crossing_positions(z_cm: np.ndarray, values_mhz: np.ndarray, detuning_mhz: float) -> list[float]:
    y = values_mhz - detuning_mhz
    idx = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
    out = []
    for i in idx:
        z1, z2 = z_cm[i], z_cm[i + 1]
        y1, y2 = y[i], y[i + 1]
        out.append(float(z1 - y1 * (z2 - z1) / (y2 - y1)))
    return out


def resonance_table(
    setup: dict[str, Any],
    *,
    detunings_mhz: Sequence[float],
    z_min_cm: float,
    z_max_cm: float,
    z_points: int,
) -> list[dict[str, Any]]:
    hamiltonian = setup["hamiltonian"]
    base = setup["frequency"]
    z_cm = np.linspace(z_min_cm, z_max_cm, z_points)
    rows = []
    for state_name, state in j2_states():
        freqs_mhz = []
        for z in z_cm:
            r = np.array([0.0, 0.0, z / 100])
            freq = calculate_transition_frequency(J1_triplet_0, state, hamiltonian.H_R(r), hamiltonian.QN)
            freqs_mhz.append((freq - base) / 1e6)
        freqs_mhz = np.asarray(freqs_mhz)
        for det in detunings_mhz:
            crosses = crossing_positions(z_cm, freqs_mhz, float(det))
            rows.append(
                {
                    "state": state_name,
                    "detuning_mhz": float(det),
                    "crossing_z_cm": crosses,
                    "crossing_z_cm_text": ";".join(f"{z:.3f}" for z in crosses),
                    "min_detuning_mhz": float(freqs_mhz.min()),
                    "max_detuning_mhz": float(freqs_mhz.max()),
                    "z_at_min_cm": float(z_cm[freqs_mhz.argmin()]),
                    "z_at_max_cm": float(z_cm[freqs_mhz.argmax()]),
                }
            )
    return rows


def scan_case(
    *,
    region: str,
    polarization: str | None,
    bg_fraction: float,
    n_steps: int,
    detunings_hz: np.ndarray,
    workers: int = 1,
    parallel_backend: str = "loky",
    include_rc_bg: bool = False,
    rc_offset_mhz: float = -7.4,
    rc_bg_fraction: float | None = None,
    top_n: int = 0,
    case_label: str | None = None,
) -> dict[str, Any]:
    setup = build_spa2_setup()
    setup["trajectory"].Vini[-1] = 184.0
    mf12 = setup["microwave_fields"][0]
    mf12.set_power(5e-5)
    fields = [mf12]
    detuning_columns = [detunings_hz]
    rc_info: dict[str, float] | None = None
    if polarization is not None:
        fields.append(
            make_bg_field(
                setup,
                polarization=polarization,
                region=region,
                bg_fraction=bg_fraction,
            )
        )
        detuning_columns.append(detunings_hz)
    if include_rc_bg:
        if polarization is None:
            raise ValueError("include_rc_bg requires a SPA background polarization/case")
        rc_fraction = bg_fraction if rc_bg_fraction is None else rc_bg_fraction
        rc_field, rc_info = make_rc_bg_field(
            setup,
            bg_fraction=rc_fraction,
            rc_offset_mhz=rc_offset_mhz,
        )
        fields.append(rc_field)
        detuning_columns.append(np.zeros_like(detunings_hz))

    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        fields,
    )
    detunings_batched = np.column_stack(detuning_columns)
    intensity_prefactors = np.ones((detunings_hz.size, len(fields)), dtype=float)
    result = simulator.run_microwave_scan(
        detunings_hz=detunings_batched,
        intensity_prefactors=intensity_prefactors,
        N_steps=n_steps,
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        workers=workers,
        parallel_backend=parallel_backend,
        allow_multitone_same_manifold=include_rc_bg,
    )

    qn = setup["hamiltonian"].QN
    initial_idx = 2
    init_eigen_idx = find_max_overlap_idx(setup["initial_states"][initial_idx].state_vector(qn), result.V_ini)
    target_idx = find_max_overlap_idx(J2_triplet_0.state_vector(qn), result.V_ini)
    depletion = 1 - result.probabilities_final[:, initial_idx, init_eigen_idx]
    target = result.probabilities_final[:, initial_idx, target_idx]
    monitors = result.monitor_probabilities_final[:, initial_idx, :]
    transferred = target + monitors.sum(axis=1)

    peak_idx = int(np.argmax(np.where(detunings_hz > 0, depletion, -np.inf)))
    monitor_indices = [
        find_max_overlap_idx(state.state_vector(qn), result.V_ini)
        for state in setup["monitor_states"]
    ]

    case_name = "no_bg" if polarization is None else f"{polarization}_{region}"
    if include_rc_bg:
        case_name = f"{case_name}_rc_offset"
    if case_label is not None:
        case_name = case_label

    top_population_rows = []
    if top_n > 0:
        selected_detunings = [-1.0, -0.75, -0.5, 0.5, float(detunings_hz[peak_idx] / 1e6)]
        selected_indices = sorted(
            {
                int(np.argmin(np.abs(detunings_hz / 1e6 - detuning)))
                for detuning in selected_detunings
            }
        )
        for detuning_idx in selected_indices:
            probabilities = result.probabilities_final[detuning_idx, initial_idx]
            for rank, eigen_idx in enumerate(np.argsort(probabilities)[::-1][:top_n], start=1):
                classification = "other"
                if int(eigen_idx) == int(target_idx):
                    classification = "target"
                elif int(eigen_idx) in monitor_indices:
                    classification = "monitor"
                try:
                    label = (
                        vector_to_state(result.V_fin[:, eigen_idx], qn)
                        .transform_to_coupled()
                        .largest
                        .state_string_custom(["J", "F1", "F", "mF"])
                    )
                except Exception as exc:
                    label = f"label_error: {exc!r}"
                top_population_rows.append(
                    {
                        "case": case_name,
                        "detuning_mhz": float(detunings_hz[detuning_idx] / 1e6),
                        "rank": rank,
                        "final_eigenstate_index": int(eigen_idx),
                        "population": float(probabilities[eigen_idx]),
                        "classification": classification,
                        "largest_coupled_label": label,
                    }
                )

    payload = {
        "case": case_name,
        "region": "none" if polarization is None else region,
        "polarization": "none" if polarization is None else polarization,
        "bg_fraction": 0.0 if polarization is None else bg_fraction,
        "include_rc_bg": include_rc_bg,
        "rc_bg_fraction": None if not include_rc_bg else (bg_fraction if rc_bg_fraction is None else rc_bg_fraction),
        "detuning_mhz": (detunings_hz / 1e6).tolist(),
        "depletion": depletion.tolist(),
        "target": target.tolist(),
        "transferred": transferred.tolist(),
        "monitor_probabilities": monitors.tolist(),
        "positive_peak_detuning_mhz": float(detunings_hz[peak_idx] / 1e6),
        "positive_peak_depletion": float(depletion[peak_idx]),
        "positive_peak_target": float(target[peak_idx]),
        "positive_peak_transferred": float(transferred[peak_idx]),
        "positive_peak_monitor_probabilities": monitors[peak_idx].tolist(),
        "top_population_rows": top_population_rows,
    }
    if rc_info is not None:
        payload.update(rc_info)
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SPA2 background-field positive-detuning feature.")
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
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/spa2_bg_feature_analysis.json"),
        help="Output JSON path for the full simulated analysis.",
    )
    parser.add_argument(
        "--resonance-csv",
        type=Path,
        default=Path("results/spa2_bg_resonances.csv"),
        help="Output CSV path for the resonance-crossing table.",
    )
    parser.add_argument(
        "--population-csv",
        type=Path,
        default=Path("results/spa2_bg_top_populations.csv"),
        help="Output CSV path for Top-N final population rows.",
    )
    args = parser.parse_args()
    if args.n_steps is None:
        args.n_steps = 20000 if args.include_rc_bg else 10000

    detunings_hz = make_detunings(
        args.n_detunings,
        lo_mhz=args.detuning_lo_mhz,
        hi_mhz=args.detuning_hi_mhz,
    )
    setup = build_spa2_setup()
    resonance_rows = resonance_table(
        setup,
        detunings_mhz=[-1.0, -0.75, -0.5, 0.188, 0.406, 0.5, 0.625],
        z_min_cm=-8.0,
        z_max_cm=20.0,
        z_points=561,
    )

    scan_results = [
        scan_case(region="none", polarization=None, bg_fraction=0.0, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, top_n=args.top_n),
        scan_case(region="constant", polarization="z", bg_fraction=1 / 35, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, top_n=args.top_n),
        scan_case(region="left", polarization="z", bg_fraction=1 / 35, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, top_n=args.top_n),
        scan_case(region="right", polarization="z", bg_fraction=1 / 35, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, top_n=args.top_n),
        scan_case(region="constant", polarization="zy", bg_fraction=1 / 30, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, top_n=args.top_n),
        scan_case(region="left", polarization="zy", bg_fraction=1 / 30, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, top_n=args.top_n),
        scan_case(region="right", polarization="zy", bg_fraction=1 / 30, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, top_n=args.top_n),
    ]
    if args.include_rc_bg:
        scan_results.extend(
            [
                scan_case(region="right", polarization="z", bg_fraction=0.0, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, include_rc_bg=True, rc_offset_mhz=args.rc_offset_mhz, rc_bg_fraction=1 / 35 if args.rc_bg_fraction is None else args.rc_bg_fraction, top_n=args.top_n, case_label="z_rc_only"),
                scan_case(region="right", polarization="z", bg_fraction=1 / 35, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, include_rc_bg=True, rc_offset_mhz=args.rc_offset_mhz, rc_bg_fraction=args.rc_bg_fraction, top_n=args.top_n),
                scan_case(region="right", polarization="zy", bg_fraction=0.0, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, include_rc_bg=True, rc_offset_mhz=args.rc_offset_mhz, rc_bg_fraction=1 / 30 if args.rc_bg_fraction is None else args.rc_bg_fraction, top_n=args.top_n, case_label="zy_rc_only"),
                scan_case(region="right", polarization="zy", bg_fraction=1 / 30, n_steps=args.n_steps, detunings_hz=detunings_hz, workers=args.workers, parallel_backend=args.parallel_backend, include_rc_bg=True, rc_offset_mhz=args.rc_offset_mhz, rc_bg_fraction=args.rc_bg_fraction, top_n=args.top_n),
            ]
        )

    payload = {
        "generated_at": utc_now_iso(),
        "config": {
            "n_steps": args.n_steps,
            "n_detunings": args.n_detunings,
            "detuning_lo_mhz": args.detuning_lo_mhz,
            "detuning_hi_mhz": args.detuning_hi_mhz,
            "workers": args.workers,
            "parallel_backend": args.parallel_backend,
            "include_rc_bg": args.include_rc_bg,
            "rc_offset_mhz": args.rc_offset_mhz,
            "rc_bg_fraction": args.rc_bg_fraction,
            "top_n": args.top_n,
        },
        "base_frequency_ghz": setup["frequency"] / 1e9,
        "spa2_center_z_cm": float(setup["r0"][2] * 100),
        "notes": [
            "left/right background regions are split at z ~= 0 cm in the notebook, not at the SPA2 center.",
            "positive-detuning crossings near 10-12 cm are inside the right-side background region and far outside the main Gaussian beam center.",
        ],
        "resonance_rows": resonance_rows,
        "scan_results": scan_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")

    csv_rows = []
    for row in resonance_rows:
        flat = dict(row)
        flat.pop("crossing_z_cm", None)
        csv_rows.append(flat)
    write_csv(args.resonance_csv, csv_rows)
    print(f"wrote {args.resonance_csv}")

    population_rows = [
        row
        for result in scan_results
        for row in result.get("top_population_rows", [])
    ]
    write_csv(args.population_csv, population_rows)
    print(f"wrote {args.population_csv}")


if __name__ == "__main__":
    main()
