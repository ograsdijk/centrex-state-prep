"""Figures and numbers for the SPA singlet-vs-triplet report.

Runs dedicated 1D detuning and power scans for both adiabatic passages, starting from
all four J=0 hyperfine sublevels, and writes the figures the report embeds plus a JSON
file holding every number the report quotes in prose. Nothing in the report should be a
hand-typed number that can drift away from the code.

The companion notebook is `examples/SPA/Experimental verification/SPA - singlet vs
triplet, no background.ipynb`, which found the optima this script holds fixed. The 2D
grids come from the CSV that notebook wrote; only the 1D lineshapes are recomputed,
because the notebook's grid samples the narrow AP2 feature too coarsely to plot.

    .\\.venv\\Scripts\\python.exe scripts\\make_singlet_triplet_report_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "benchmarks"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from benchmarks.common import build_spa_cascade_setup, utc_now_iso  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

DEFAULT_OUT = REPO_ROOT / "examples" / "SPA" / "Experimental verification"

STATE_NAMES = ["singlet", "triplet_m", "triplet_0", "triplet_p"]
STATE_LABELS = [
    "singlet  (F=0, mF=0)",
    "triplet  (F=1, mF=-1)",
    "triplet  (F=1, mF=0)",
    "triplet  (F=1, mF=+1)",
]
STATE_COLORS = ["#1b6ca8", "#d1495b", "#2e933c", "#e08a1e"]
STATE_STYLES = ["-", "--", "-.", ":"]

# Optima located by the notebook's 2D grids.
D1_OPT, PREF1_OPT = 1.350e6, 16.156
D2_OPT, PREF2_OPT = 0.212e6, 8.254

N_STEPS = 10_000
WORKERS = 8


# --------------------------------------------------------------------------- helpers


class Observables:
    """Matched-partner populations, with the group sum kept as a label-swap check.

    The reported transfer is a single tracked eigenstate per chain: the target that
    carries the same nuclear-spin label as the initial state. The near-degenerate group
    sum exists only to detect an adiabatic label permuting, the failure mode
    `IMPROVEMENTS.md` documents; the manifold sum says where lost population went.
    """

    def __init__(self, hamiltonian: Any) -> None:
        from state_prep.utils import find_max_overlap_idx

        self._find = find_max_overlap_idx
        self.qn = hamiltonian.QN
        self.j_of_basis = np.array([q.J for q in self.qn])
        energies, _ = np.linalg.eigh(hamiltonian.get_H_t_func()(0.0))
        self.energies = np.sort(energies)

    def _manifold_indices(self, evecs: np.ndarray, j: int) -> np.ndarray:
        weight = np.abs(evecs) ** 2
        js = np.unique(self.j_of_basis)
        per_j = np.array([weight[self.j_of_basis == k].sum(axis=0) for k in js])
        return np.flatnonzero(js[per_j.argmax(axis=0)] == j)

    def _degenerate_indices(self, idx: int, tol_hz: float = 5e3) -> np.ndarray:
        delta = np.abs(self.energies - self.energies[idx]) / (2 * np.pi)
        return np.flatnonzero(delta < tol_hz)

    def __call__(self, result: Any, targets: list[Any], j_target: int):
        probs = result.probabilities_final
        evecs = result.V_ini
        manifold = self._manifold_indices(evecs, j_target)
        matched, group = [], []
        for s, target in enumerate(targets):
            idx = self._find(target.state_vector(self.qn), evecs)
            matched.append(probs[:, s, idx])
            group.append(probs[:, s, self._degenerate_indices(idx)].sum(axis=-1))
        return (
            np.stack(matched, axis=-1),
            np.stack(group, axis=-1),
            probs[:, :, manifold].sum(axis=-1),
        )


def run_scan(sim, *, detunings, prefactors, n_fields, n_steps, park=None):
    """One `run_microwave_scan` over a 1D axis, with the other rung optionally parked."""
    import state_prep as sp

    det, pref = sp.scan_grid(
        n_fields=n_fields,
        detunings_hz=detunings,
        intensity_prefactors=prefactors,
        detuning_fields=[n_fields - 1],
        prefactor_fields=[n_fields - 1],
    )
    if park is not None:
        det[:, 0], pref[:, 0] = park
    started = time.perf_counter()
    result = sim.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=n_steps,
        store_final_probabilities=True,
        progress=False,
        workers=WORKERS,
    )
    return result, time.perf_counter() - started


# --------------------------------------------------------------------------- figures


def _decorate(ax, ylabel=True):
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25, lw=0.6)
    if ylabel:
        ax.set_ylabel("transfer efficiency")


def figure_vs_detuning(data, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, key, title in (
        (axes[0], "ap1_detuning", "AP1   J = 0 $\\rightarrow$ 1"),
        (axes[1], "ap2_detuning", "AP2   J = 1 $\\rightarrow$ 2  (end to end from J = 0)"),
    ):
        x = np.asarray(data[key]["detuning_hz"]) / 1e6
        y = np.asarray(data[key]["matched"])
        for s in range(4):
            ax.plot(x, y[:, s], STATE_STYLES[s], color=STATE_COLORS[s],
                    label=STATE_LABELS[s], lw=1.8)
        ax.set_xlabel("detuning (MHz)")
        ax.set_title(title)
        _decorate(ax, ylabel=ax is axes[0])
    axes[0].legend(frameon=False, loc="lower center")
    fig.suptitle("Transfer efficiency vs microwave detuning, per hyperfine sublevel",
                 fontsize=13)
    path = out / "transfer_vs_detuning.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_vs_power(data, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, key, title in (
        (axes[0], "ap1_power", "AP1   J = 0 $\\rightarrow$ 1"),
        (axes[1], "ap2_power", "AP2   J = 1 $\\rightarrow$ 2  (end to end from J = 0)"),
    ):
        watts = np.asarray(data[key]["power_w"])
        y = np.asarray(data[key]["matched"])
        for s in range(4):
            ax.plot(watts, y[:, s], STATE_STYLES[s], color=STATE_COLORS[s],
                    label=STATE_LABELS[s], lw=1.8)
        ax.set_xscale("log")
        ax.set_xlim(watts.min(), watts.max())
        ax.set_xlabel("microwave power (W)")
        _decorate(ax, ylabel=ax is axes[0])

        # dBm is affine in log10(W), so a plain linear twin lines up exactly. Building
        # it with secondary_xaxis instead inherits the log scale and mislabels itself.
        dbm = ax.twiny()
        dbm.set_xlim(*(10 * np.log10(np.asarray(ax.get_xlim()) * 1e3)))
        dbm.set_xlabel("microwave power (dBm)", labelpad=6)
        dbm.set_title(title, pad=28)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Transfer efficiency vs microwave power, per hyperfine sublevel",
                 fontsize=13)
    path = out / "transfer_vs_power.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_spread(data, out: Path) -> Path:
    """The claim in quantitative form: the sublevel-to-sublevel difference."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)

    for key, colour, label in (
        ("ap1_detuning", "#1b6ca8", "AP1"),
        ("ap2_detuning", "#d1495b", "AP2 (end to end)"),
    ):
        y = np.asarray(data[key]["matched"])
        spread = y.max(axis=1) - y.min(axis=1)
        keep = y.max(axis=1) > 0.01  # a spread of zero where nothing transfers is not informative
        axes[0].semilogy(np.asarray(data[key]["detuning_hz"])[keep] / 1e6,
                         np.maximum(spread[keep], 1e-12), color=colour, label=label, lw=1.8)
    axes[0].set_xlabel("detuning (MHz)")

    for key, colour, label in (
        ("ap1_power", "#1b6ca8", "AP1"),
        ("ap2_power", "#d1495b", "AP2 (end to end)"),
    ):
        y = np.asarray(data[key]["matched"])
        spread = y.max(axis=1) - y.min(axis=1)
        axes[1].loglog(np.asarray(data[key]["power_w"]),
                       np.maximum(spread, 1e-12), color=colour, label=label, lw=1.8)
    axes[1].set_xlabel("microwave power (W)")

    for ax in axes:
        ax.axhline(1e-3, color="0.55", ls=":", lw=1.2)
        ax.text(0.02, 1.3e-3, "0.1 %", transform=ax.get_yaxis_transform(),
                color="0.4", fontsize=9)
        ax.set_ylim(1e-8, 1.5)
        ax.grid(alpha=0.25, lw=0.6, which="both")
        ax.legend(frameon=False)
    axes[0].set_ylabel("spread across the four sublevels")
    fig.suptitle("Difference between the best and worst hyperfine sublevel", fontsize=13)
    path = out / "sublevel_spread.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_2d_grids(csv_path: Path, out: Path) -> Path | None:
    """The notebook's 2D detuning x power grids, one panel per sublevel."""
    if not csv_path.exists():
        print(f"  ! {csv_path.name} not found, skipping the 2D grid figure")
        return None
    import polars as pl

    frame = pl.read_csv(csv_path)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True,
                             sharex="row", sharey=True)
    for row, (stage, title) in enumerate(
        [("AP1", "AP1  J = 0 $\\rightarrow$ 1"),
         ("AP2_end_to_end", "AP2  end to end J = 0 $\\rightarrow$ 2")]
    ):
        sub = frame.filter(pl.col("stage") == stage)
        det = np.unique(sub["detuning_hz"].to_numpy())
        pref = np.unique(sub["intensity_prefactor"].to_numpy())
        for s, name in enumerate(STATE_NAMES):
            grid = sub[name].to_numpy().reshape(det.size, pref.size)
            mesh = axes[row, s].pcolormesh(det / 1e6, pref, grid.T, vmin=0, vmax=1,
                                           shading="nearest", cmap="viridis")
            axes[row, s].set_yscale("log")
            axes[row, s].set_title(f"{title}\n{STATE_LABELS[s]}" if s == 0
                                   else STATE_LABELS[s], fontsize=10)
            axes[row, s].set_xlabel("detuning (MHz)")
        axes[row, 0].set_ylabel("intensity prefactor")
    fig.colorbar(mesh, ax=axes, label="transfer efficiency", shrink=0.8)
    path = out / "transfer_2d_grids.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- numbers


def structural_numbers(setup: dict) -> dict:
    """The inputs to the report's theory section.

    The angular matrix elements and the transition frequencies per chain are what make
    the spectator argument concrete: identical couplings, near-identical frequencies.
    """
    from state_prep.utils import calculate_transition_frequency

    hamiltonian = setup["hamiltonian"]
    qn = hamiltonian.QN
    mf01, mf12 = setup["microwave_fields"]
    r0_01, r0_12 = setup["positions"]
    ini, t1, t2 = setup["initial_states"], setup["targets_j1"], setup["targets_j2"]

    out: dict[str, Any] = {}

    # Stark field and shift along the trajectory.
    out["stark"] = {}
    zero_field = np.linalg.eigvalsh(hamiltonian.H_EB(np.zeros(3), np.zeros(3)))
    for label, position in (
        ("t0", np.array([0.0, 0.0, -80e-3])),
        ("ap1_centre", np.asarray(r0_01)),
        ("ap2_centre", np.asarray(r0_12)),
    ):
        e_vec = setup["electric_field"].E_R(position)
        shifted = np.linalg.eigvalsh(hamiltonian.H_EB(e_vec, np.zeros(3)))
        out["stark"][label] = {
            "field_v_per_cm": float(np.linalg.norm(e_vec)),
            "j0_shift_hz": float((shifted[:4].mean() - zero_field[:4].mean()) / (2 * np.pi)),
        }

    # Zero-field hyperfine spread within each rotational manifold.
    energies, evecs = np.linalg.eigh(hamiltonian.H_EB(np.zeros(3), np.zeros(3)))
    j_of_basis = np.array([q.J for q in qn])
    weight = np.array([[(np.abs(evecs[j_of_basis == j, k]) ** 2).sum()
                        for k in range(len(qn))] for j in (0, 1, 2, 3)])
    labels = weight.argmax(axis=0)
    out["hyperfine_spread_hz"] = {}
    for j in (0, 1, 2):
        levels = np.sort(energies[labels == j]) / (2 * np.pi)
        out["hyperfine_spread_hz"][f"J{j}"] = float(levels.max() - levels.min())

    # How cleanly each nominal state maps onto an eigenstate, and its coupled-basis
    # content. F1 and F are not good quantum numbers for the J=1 and J=2 targets at
    # these fields, so the report must not present them as if they were. mF is exact.
    from state_prep.utils import find_max_overlap_idx

    _, evecs_t0 = np.linalg.eigh(hamiltonian.get_H_t_func()(0.0))
    order = np.argsort(np.linalg.eigvalsh(hamiltonian.get_H_t_func()(0.0)))
    evecs_t0 = evecs_t0[:, order]
    mf_of_basis = np.array([q.mJ + q.m1 + q.m2 for q in qn])

    out["target_identification"] = {}
    for label, states in (("J0", ini), ("J1", t1), ("J2", t2)):
        entry = {}
        for name, state in zip(STATE_NAMES, states):
            vec = state.state_vector(qn)
            vec = vec / np.linalg.norm(vec)
            overlaps = np.abs(vec.conj() @ evecs_t0) ** 2
            ranked = np.argsort(overlaps)[::-1]
            idx = find_max_overlap_idx(vec, evecs_t0)
            weights = np.abs(evecs_t0[:, idx]) ** 2
            coupled = [(float(abs(a) ** 2), f"F1={c.F1},F={c.F},mF={c.mF}")
                       for a, c in state.transform_to_coupled().data if abs(a) > 1e-6]
            coupled.sort(reverse=True)
            entry[name] = {
                "eigenstate_index": int(idx),
                "overlap": float(overlaps[ranked[0]]),
                "runner_up_overlap": float(overlaps[ranked[1]]),
                "mF": float((mf_of_basis * weights).sum()),
                "dominant_coupled": coupled[0][1],
                "dominant_coupled_weight": coupled[0][0],
                "n_coupled_components": len(coupled),
            }
        out["target_identification"][label] = entry

    # Per-chain transition frequency and Rabi rate.
    out["chains"] = {}
    for stage, field, lower, upper, r0 in (
        ("AP1", mf01, ini, t1, r0_01),
        ("AP2", mf12, t1, t2, r0_12),
    ):
        matrix = hamiltonian.H_R(np.asarray(r0))
        freqs, rabis = [], []
        for a, b in zip(lower, upper):
            freqs.append(calculate_transition_frequency(a, b, matrix, qn))
            rabis.append(field.calculate_rabi_rate(a, b, 5e-5, np.asarray(r0)) / (2 * np.pi))
        freqs, rabis = np.array(freqs), np.array(rabis)
        out["chains"][stage] = {
            "frequency_hz": {n: float(v) for n, v in zip(STATE_NAMES, freqs)},
            "frequency_spread_hz": float(freqs.max() - freqs.min()),
            "rabi_hz_at_5e-5W": {n: float(v) for n, v in zip(STATE_NAMES, rabis)},
            "rabi_relative_spread": float((rabis.max() - rabis.min()) / rabis.mean()),
        }
    return out


def coupled_basis_terms(setup: dict) -> dict:
    """Expand each chain's dipole matrix element term by term in the coupled basis.

    The spectator argument is a one-liner in the decoupled basis and looks like a
    coincidence in the coupled one, where every `F -> F'` carries its own 3j and 6j
    factors. This shows the reconciliation explicitly: the individual terms differ in
    number, magnitude and sign between chains, and every total is the same bare
    rotational matrix element.
    """
    from centrex_tlf import states as tlf_states
    from centrex_tlf.hamiltonian.matrix_elements_electric_dipole import (
        generate_ED_ME_mixed_state as dipole_me,
    )

    pol = np.array([0.0, 0.0, 1.0])  # pi / z
    out: dict[str, Any] = {}
    for stage, lower_states, upper_states in (
        ("AP1", setup["initial_states"], setup["targets_j1"]),
        ("AP2", setup["targets_j1"], setup["targets_j2"]),
    ):
        stage_out = {}
        for name, lower, upper in zip(STATE_NAMES, lower_states, upper_states):
            low_c, up_c = lower.transform_to_coupled(), upper.transform_to_coupled()
            terms, total = [], 0.0
            for amp_lo, basis_lo in low_c.data:
                for amp_up, basis_up in up_c.data:
                    if abs(amp_lo) < 1e-9 or abs(amp_up) < 1e-9:
                        continue
                    element = dipole_me(
                        tlf_states.State([(1.0, basis_up)]),
                        tlf_states.State([(1.0, basis_lo)]),
                        pol_vec=pol,
                    )
                    if abs(element) < 1e-12:
                        continue
                    contribution = np.conj(amp_up) * amp_lo * element
                    total += contribution
                    terms.append({
                        "from": f"F1={basis_lo.F1},F={basis_lo.F},mF={basis_lo.mF}",
                        "to": f"F1={basis_up.F1},F={basis_up.F},mF={basis_up.mF}",
                        "element": float(np.real(element)),
                        "amplitude_product": float(np.real(np.conj(amp_up) * amp_lo)),
                        "contribution": float(np.real(contribution)),
                    })
            stage_out[name] = {
                "terms": terms,
                "n_terms": len(terms),
                "total": float(np.real(total)),
                "whole_state": float(np.real(dipole_me(up_c, low_c, pol_vec=pol))),
            }
        totals = [stage_out[n]["total"] for n in STATE_NAMES]
        stage_out["total_spread"] = float(max(totals) - min(totals))
        out[stage] = stage_out
    return out


def cliff_shift_test(det: np.ndarray, transfer: np.ndarray, frequencies: dict) -> dict:
    """Compare each chain's measured cutoff position with its resonance offset.

    The residual state dependence is a resonance-position offset, so each chain's cliff
    should sit at a carrier detuning displaced by exactly that offset. This is the
    sharpest available test of the mechanism.
    """
    def crossing(values: np.ndarray) -> float:
        below = np.where((values[:-1] >= 0.5) & (values[1:] < 0.5))[0]
        if not below.size:
            return float("nan")
        i = below[-1]
        return float(det[i] + (values[i] - 0.5) / (values[i] - values[i + 1])
                     * (det[i + 1] - det[i]))

    edges = {n: crossing(transfer[:, s]) for s, n in enumerate(STATE_NAMES)}
    reference = "triplet_0"
    out = {"resolution_hz": float(det[1] - det[0]), "edge_hz": edges, "reference": reference}
    out["shift_hz"] = {
        n: {
            "predicted": float(frequencies[n] - frequencies[reference]),
            "measured": float(edges[n] - edges[reference]),
            "difference": float((edges[n] - edges[reference])
                                - (frequencies[n] - frequencies[reference])),
        }
        for n in STATE_NAMES
    }
    return out


def feature_width(detunings: np.ndarray, transfer: np.ndarray, level: float = 0.5) -> float:
    """Full width of the region where the worst sublevel clears `level`."""
    worst = transfer.min(axis=1)
    above = detunings[worst > level]
    return float(above.max() - above.min()) if above.size else float("nan")


def windows(axis: np.ndarray, transfer: np.ndarray, thresholds) -> dict:
    """Range of `axis` over which every sublevel clears each threshold.

    Reported at several thresholds because a single one is misleading: the end-to-end
    plateau sits just below 0.999, so that threshold alone makes a broad plateau look
    like a knife edge.
    """
    worst = transfer.min(axis=1)
    out = {}
    for level in thresholds:
        inside = axis[worst > level]
        out[str(level)] = (
            [float(inside.min()), float(inside.max()), int(inside.size)]
            if inside.size else None
        )
    return out


def collect_numbers(data: dict, setup: dict, *, n_steps: int, swaps: dict) -> dict:
    mf01, mf12 = setup["microwave_fields"]
    numbers: dict[str, Any] = {
        "generated": utc_now_iso(),
        "n_steps": n_steps,
        "workers": WORKERS,
        "state_names": STATE_NAMES,
        "optima": {
            "AP1": {"detuning_hz": D1_OPT, "intensity_prefactor": PREF1_OPT,
                    "power_w": float(mf01.intensity.power * PREF1_OPT)},
            "AP2": {"detuning_hz": D2_OPT, "intensity_prefactor": PREF2_OPT,
                    "power_w": float(mf12.intensity.power * PREF2_OPT)},
        },
        "nominal_power_w": {"AP1": float(mf01.intensity.power),
                            "AP2": float(mf12.intensity.power)},
        "label_swap_check": swaps,
        "structure": structural_numbers(setup),
        "coupled_basis_terms": coupled_basis_terms(setup),
    }
    if "ap2_cliff" in data:
        numbers["cliff_shift_test"] = cliff_shift_test(
            np.asarray(data["ap2_cliff"]["detuning_hz"]),
            np.asarray(data["ap2_cliff"]["matched"]),
            numbers["structure"]["chains"]["AP2"]["frequency_hz"],
        )

    thresholds = (0.999, 0.998, 0.995, 0.99)
    for stage, det_key, pow_key in (("AP1", "ap1_detuning", "ap1_power"),
                                    ("AP2", "ap2_detuning", "ap2_power")):
        det = np.asarray(data[det_key]["detuning_hz"])
        det_y = np.asarray(data[det_key]["matched"])
        powers = np.asarray(data[pow_key]["power_w"])
        pow_y = np.asarray(data[pow_key]["matched"])
        best = det_y.min(axis=1).argmax()
        at_opt = det_y[best]
        spread_det = det_y.max(axis=1) - det_y.min(axis=1)
        worst_i = spread_det.argmax()
        numbers[stage] = {
            "best_detuning_hz": float(det[best]),
            "at_optimum": {n: float(v) for n, v in zip(STATE_NAMES, at_opt)},
            "spread_at_optimum": float(at_opt.max() - at_opt.min()),
            "worst_sublevel_at_optimum": float(at_opt.min()),
            "feature_fwhm_hz": feature_width(det, det_y),
            "max_spread_over_detuning": float(spread_det.max()),
            "max_spread_detuning_hz": float(det[worst_i]),
            "at_max_spread": {n: float(v) for n, v in zip(STATE_NAMES, det_y[worst_i])},
            "max_spread_over_power": float((pow_y.max(axis=1) - pow_y.min(axis=1)).max()),
            "wrong_substate_at_optimum": float(
                np.asarray(data[det_key]["manifold_minus_matched"])[best].max()),
            "detuning_windows_hz": windows(det, det_y, thresholds),
            "power_windows_w": windows(powers, pow_y, thresholds),
        }
    return numbers


# ------------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help="directory holding figures/ and results/")
    parser.add_argument("--n-detunings", type=int, default=81)
    parser.add_argument("--n-powers", type=int, default=25)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--from-cache", action="store_true",
                        help="re-render figures from report_curves.json without rescanning")
    args = parser.parse_args()

    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    from state_prep import Simulator

    print("building the SPA cascade setup ...")
    setup = build_spa_cascade_setup()
    mf01, mf12 = setup["microwave_fields"]
    observables = Observables(setup["hamiltonian"])

    if args.from_cache:
        cached = json.loads((figures_dir / "report_curves.json").read_text(encoding="utf-8"))
        cached_numbers = json.loads(
            (figures_dir / "report_numbers.json").read_text(encoding="utf-8"))
        write_outputs(cached, setup, figures_dir, args,
                      n_steps=cached_numbers["n_steps"],
                      swaps=cached_numbers["label_swap_check"])
        return

    common = (setup["trajectory"], setup["electric_field"], setup["magnetic_field"],
              setup["initial_states"], setup["hamiltonian"])
    sim_ap1 = Simulator(*common, [mf01])
    sim_ap2 = Simulator(*common, [mf01, mf12])

    pref_axis = np.logspace(-2, 1.5, args.n_powers)
    data: dict[str, Any] = {}
    swaps: dict[str, float] = {}

    scans = [
        ("ap1_detuning", sim_ap1, 1, dict(
            detunings=np.linspace(-6e6, 8e6, args.n_detunings),
            prefactors=PREF1_OPT), None, setup["targets_j1"], 1, mf01),
        ("ap1_power", sim_ap1, 1, dict(
            detunings=D1_OPT, prefactors=pref_axis), None, setup["targets_j1"], 1, mf01),
        ("ap2_detuning", sim_ap2, 2, dict(
            detunings=np.linspace(-2.0e6, 0.6e6, args.n_detunings),
            prefactors=PREF2_OPT), (D1_OPT, PREF1_OPT), setup["targets_j2"], 2, mf12),
        ("ap2_power", sim_ap2, 2, dict(
            detunings=D2_OPT, prefactors=pref_axis),
            (D1_OPT, PREF1_OPT), setup["targets_j2"], 2, mf12),
        # Finely resolved AP2 cutoff, for the cliff-shift test.
        ("ap2_cliff", sim_ap2, 2, dict(
            detunings=np.linspace(250e3, 380e3, 53), prefactors=PREF2_OPT),
            (D1_OPT, PREF1_OPT), setup["targets_j2"], 2, mf12),
    ]

    for key, sim, n_fields, axes_kwargs, park, targets, j_target, field in scans:
        result, elapsed = run_scan(sim, n_fields=n_fields, park=park,
                                   n_steps=args.n_steps, **axes_kwargs)
        matched, group, manifold = observables(result, targets, j_target)
        swaps[key] = float(np.abs(matched - group).max())
        norm_err = float(np.abs(result.probabilities_final.sum(axis=-1) - 1.0).max())
        print(f"  {key:14} {result.batch_size:3d} points in {elapsed:6.1f} s   "
              f"|matched-group| = {swaps[key]:.2e}   |sum P - 1| = {norm_err:.1e}")

        entry: dict[str, Any] = {
            "matched": matched.tolist(),
            "manifold_minus_matched": (manifold - matched).tolist(),
            "normalisation_error": norm_err,
        }
        if np.ndim(axes_kwargs["detunings"]) > 0:
            entry["detuning_hz"] = np.asarray(axes_kwargs["detunings"]).tolist()
        else:
            powers = field.intensity.power * pref_axis
            entry["power_w"] = powers.tolist()
            entry["intensity_prefactor"] = pref_axis.tolist()
        data[key] = entry

    write_outputs(data, setup, figures_dir, args, n_steps=args.n_steps, swaps=swaps)


def write_outputs(data, setup, figures_dir: Path, args, *, n_steps: int, swaps: dict) -> None:
    numbers = collect_numbers(data, setup, n_steps=n_steps, swaps=swaps)

    print("\nwriting figures ...")
    written = [
        figure_vs_detuning(data, figures_dir),
        figure_vs_power(data, figures_dir),
        figure_spread(data, figures_dir),
        figure_2d_grids(
            args.output / "results" / "SPA_singlet_vs_triplet_no_background.csv",
            figures_dir),
    ]
    for path in written:
        if path is not None:
            print(f"  {path.relative_to(REPO_ROOT)}  ({path.stat().st_size / 1024:.0f} kB)")

    numbers_path = figures_dir / "report_numbers.json"
    numbers_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(f"  {numbers_path.relative_to(REPO_ROOT)}")

    curves_path = figures_dir / "report_curves.json"
    curves_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"  {curves_path.relative_to(REPO_ROOT)}")

    print("\nheadline:")
    for stage in ("AP1", "AP2"):
        entry = numbers[stage]
        print(f"  {stage}: worst sublevel {entry['worst_sublevel_at_optimum']:.6f}, "
              f"spread {entry['spread_at_optimum']:.2e}, "
              f"max spread across the scan {entry['max_spread_over_detuning']:.2e}")


if __name__ == "__main__":
    main()
