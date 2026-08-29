from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, cast

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_nvidia_dll_paths() -> list[str]:
    """Add pip-provided NVIDIA CUDA DLL directories to Windows search paths.

    Delegates to the package implementation so benchmarks and the GPU import
    path discover CUDA the same way. `prepend_path` is on here because NVRTC has
    been observed to need PATH in addition to the DLL directories.
    """
    from state_prep_gpu._cupy import add_cuda_dll_directories

    return add_cuda_dll_directories(prepend_path=True)


def environment_info(include_gpu: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception as exc:
        info["numpy_error"] = repr(exc)

    try:
        import scipy

        info["scipy"] = scipy.__version__
    except Exception as exc:
        info["scipy_error"] = repr(exc)

    if include_gpu:
        try:
            add_nvidia_dll_paths()
            import cupy

            info["cupy"] = cupy.__version__
            count = int(cupy.cuda.runtime.getDeviceCount())
            info["cuda_device_count"] = count
            devices = []
            for idx in range(count):
                props = cupy.cuda.runtime.getDeviceProperties(idx)
                name = props.get("name", b"")
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                devices.append(
                    {
                        "index": idx,
                        "name": name,
                        "total_global_mem": int(props.get("totalGlobalMem", 0)),
                    }
                )
            info["cuda_devices"] = devices
        except Exception as exc:
            info["cupy_error"] = repr(exc)

    return info


def time_call(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    sync: Optional[Callable[[], None]] = None,
) -> tuple[dict[str, Any], Any]:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    times: list[float] = []
    last_result: Any = None
    for _ in range(repeat):
        start = time.perf_counter()
        last_result = fn()
        if sync is not None:
            sync()
        times.append(time.perf_counter() - start)

    return (
        {
            "time_best_s": min(times),
            "time_mean_s": mean(times),
            "time_std_s": pstdev(times) if len(times) > 1 else 0.0,
            "repeat": repeat,
            "warmup": warmup,
        },
        last_result,
    )


def stored_result_bytes(result: Any) -> int:
    total = 0
    for name in (
        "psis",
        "energies",
        "probabilities",
        "probabilities_final",
        "monitor_probabilities",
        "monitor_probabilities_final",
        "psis_final",
    ):
        arr = getattr(result, name, None)
        if arr is not None and hasattr(arr, "nbytes"):
            total += int(arr.nbytes)
    return total


def make_payload(
    *,
    benchmark: str,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    include_gpu_env: bool = True,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "generated_at": utc_now_iso(),
        "environment": environment_info(include_gpu=include_gpu_env),
        "config": config,
        "results": results,
    }


def resolve_csv_path(
    output: Optional[str],
    csv_arg: Optional[str],
    benchmark: str,
) -> Optional[Path]:
    if csv_arg is None:
        return None
    if csv_arg:
        return Path(csv_arg)
    if output:
        return Path(output).with_suffix(".csv")
    return Path(f"{benchmark}.csv")


def write_results(
    payload: dict[str, Any],
    *,
    output: Optional[str],
    csv_path: Optional[Path],
) -> None:
    jsonable = to_jsonable(payload)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(jsonable, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(jsonable, indent=2))

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [flatten_dict(row) for row in jsonable.get("results", [])]
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def flatten_dict(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, val in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            flat.update(flatten_dict(val, name))
        elif isinstance(val, (list, tuple)):
            flat[name] = json.dumps(to_jsonable(val))
        else:
            flat[name] = to_jsonable(val)
    return flat


def parse_js(value: str) -> list[int]:
    try:
        js = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--js must be a comma-separated list of integers") from exc
    if not js:
        raise ValueError("--js must contain at least one J value")
    return js


def make_detunings(batch: int, *, lo_mhz: float = -2.0, hi_mhz: float = 1.5) -> np.ndarray:
    if batch < 1:
        raise ValueError("batch must be >= 1")
    return np.linspace(float(lo_mhz), float(hi_mhz), int(batch)) * 1e6


def build_fields_and_hamiltonian(*, js: Sequence[int], v_forward: float = 184.0):
    from state_prep.electric_fields import ElectricField, Ez_from_csv
    from state_prep.hamiltonians import SlowHamiltonian
    from state_prep.magnetic_fields import MagneticField
    from state_prep.trajectory import Trajectory

    trajectory = Trajectory(
        Rini=np.array((0.0, 0.0, -80e-3)),
        Vini=np.array((0.0, 0.0, float(v_forward))),
        zfin=200e-3,
    )

    ez_interp = Ez_from_csv()

    def e_r(position):
        return np.array([0.0, 0.0, float(ez_interp(position[2]))])

    electric_field = ElectricField(e_r, trajectory.R_t)

    b0 = np.array((0.0, 0.0, 1e-3))

    def b_r(position):
        if len(np.shape(position)) == 1:
            return b0
        return b0.reshape((3, 1)) * np.ones(np.shape(position))

    magnetic_field = MagneticField(b_r, R_t=trajectory.R_t)
    hamiltonian = SlowHamiltonian(
        list(js),
        trajectory,
        cast(Any, electric_field),
        cast(Any, magnetic_field),
    )
    return trajectory, electric_field, magnetic_field, hamiltonian


def build_microwaves(hamiltonian: Any, trajectory: Any):
    from state_prep.approximate_states import J1_triplet_0, J2_triplet_0
    from state_prep.intensity_profiles import BackgroundField, GaussianBeam
    from state_prep.microwaves import MicrowaveField, Polarization
    from state_prep.utils import calculate_transition_frequency

    r0 = np.array((0.0, 0.0, 0.0254 * 1.125))
    p_z = np.array([0.0, 0.0, 1.0])

    def p_r(_position):
        return p_z / np.sqrt(np.sum(p_z**2))

    k_vec = np.array((1.0, 0.0, 0.0))
    polarization = Polarization(p_r, k_vec, f_long=1)
    frequency = calculate_transition_frequency(
        J1_triplet_0,
        J2_triplet_0,
        hamiltonian.H_R(r0),
        hamiltonian.QN,
    )

    intensity = GaussianBeam(
        power=0.5e-3,
        sigma=25.4e-3 / (2 * np.sqrt(2 * np.log(2))),
        R0=r0,
        k=k_vec,
        freq=frequency,
    )
    main = MicrowaveField(1, 2, intensity, polarization, frequency, QN=hamiltonian.QN)

    limits = [
        (-1.0, 1.0),
        (-1.0, 1.0),
        (+25.4e-3 / (2 * np.sqrt(2 * np.log(2))), 1.0),
    ]
    bg_intensity = BackgroundField(cast(Any, limits), intensity=cast(Any, main.intensity).I_R(r0) / 20)
    bg = MicrowaveField(
        1,
        2,
        bg_intensity,
        Polarization(p_r),
        frequency,
        hamiltonian.QN,
        background_field=True,
    )
    return main, bg, frequency, r0


def configure_microwaves(
    microwave_fields: Sequence[Any],
    *,
    frequency: float,
    r0: np.ndarray,
    power_w: float = 5e-5,
    bg_fraction: float = 1 / 35,
) -> None:
    main, bg = microwave_fields
    main.set_frequency(frequency)
    bg.set_frequency(frequency)
    main.set_power(power_w)
    cast(Any, bg.intensity).intensity = cast(Any, main.intensity).I_R(r0) * float(bg_fraction)


def spa2_initial_states() -> list[Any]:
    from state_prep.approximate_states import (
        J1_singlet,
        J1_triplet_0,
        J1_triplet_m,
        J1_triplet_p,
    )

    return [J1_singlet, J1_triplet_m, J1_triplet_0, J1_triplet_p]


def spa2_monitor_states() -> list[Any]:
    from centrex_tlf import states
    from state_prep.approximate_states import (
        J2_singlet,
        J2_triplet_m,
        J2_triplet_p,
    )

    return [
        J2_singlet,
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
        J2_triplet_m,
        J2_triplet_p,
    ]


def build_spa2_setup(*, js: Sequence[int] = (0, 1, 2, 3), v_forward: float = 184.0):
    trajectory, electric_field, magnetic_field, hamiltonian = build_fields_and_hamiltonian(
        js=js,
        v_forward=v_forward,
    )
    main, bg, frequency, r0 = build_microwaves(hamiltonian, trajectory)
    microwave_fields = [main, bg]
    configure_microwaves(microwave_fields, frequency=frequency, r0=r0)
    return {
        "trajectory": trajectory,
        "electric_field": electric_field,
        "magnetic_field": magnetic_field,
        "hamiltonian": hamiltonian,
        "microwave_fields": microwave_fields,
        "frequency": frequency,
        "r0": r0,
        "initial_states": spa2_initial_states(),
        "monitor_states": spa2_monitor_states(),
    }


def build_cascade_microwaves(hamiltonian: Any, *, omega_nominal: Optional[float] = None):
    """SPA1 (J=0->1) and SPA2 (J=1->2) as a two-rung cascade, no background fields.

    Same optics as `build_microwaves` -- pure z polarization, k along x, the 1 inch
    FWHM Gaussian -- but two rungs and nothing stray. Each field's carrier is the
    transition frequency evaluated at *its own* beam centre.

    Powers are anchored to a Rabi rate rather than set in Watts, so the two rungs
    are driven equivalently despite their different matrix elements.
    `omega_nominal` defaults to what the established SPA2 operating point of
    `power_w=5e-5` produces (see `configure_microwaves`).

    Returns `(mf01, mf12, frequencies, positions)` with the fields ordered low rung
    first, which `build_rotating_frame_shift` requires for a cascade.
    """
    from state_prep.approximate_states import (
        J0_triplet_0,
        J1_triplet_0,
        J2_triplet_0,
    )
    from state_prep.intensity_profiles import GaussianBeam
    from state_prep.microwaves import MicrowaveField, Polarization
    from state_prep.utils import calculate_transition_frequency

    sigma = 25.4e-3 / (2 * np.sqrt(2 * np.log(2)))
    k_vec = np.array((1.0, 0.0, 0.0))
    p_z = np.array([0.0, 0.0, 1.0])

    def p_r(_position):
        return p_z / np.sqrt(np.sum(p_z**2))

    polarization = Polarization(p_r, k_vec, f_long=1)

    rungs = [
        (0, 1, J0_triplet_0, J1_triplet_0, np.array((0.0, 0.0, 0.0))),
        (1, 2, J1_triplet_0, J2_triplet_0, np.array((0.0, 0.0, 0.0254 * 1.125))),
    ]

    fields, frequencies, positions = [], [], []
    for jg, je, lower, upper, r0 in rungs:
        frequency = calculate_transition_frequency(
            lower, upper, hamiltonian.H_R(r0), hamiltonian.QN
        )
        intensity = GaussianBeam(
            power=5e-5, sigma=sigma, R0=r0, k=k_vec, freq=frequency
        )
        field = MicrowaveField(
            jg, je, intensity, polarization, frequency, QN=hamiltonian.QN
        )
        fields.append(field)
        frequencies.append(frequency)
        positions.append(r0)

    if omega_nominal is None:
        omega_nominal = fields[1].calculate_rabi_rate(
            J1_triplet_0, J2_triplet_0, 5e-5, positions[1]
        )
    for field, (_, _, lower, upper, r0) in zip(fields, rungs):
        # Sets intensity.power in place; it does not return the power.
        field.calculate_microwave_power(lower, upper, omega_nominal, r0)

    return fields[0], fields[1], frequencies, positions


def spa_cascade_initial_states() -> list[Any]:
    """The four J=0 hyperfine sublevels: F=0 mF=0, then F=1 mF=-1, 0, +1."""
    from state_prep.approximate_states import (
        J0_singlet,
        J0_triplet_0,
        J0_triplet_m,
        J0_triplet_p,
    )

    return [J0_singlet, J0_triplet_m, J0_triplet_0, J0_triplet_p]


def spa_cascade_targets(j: int) -> list[Any]:
    """Matched nuclear-spin partners in manifold `j`, ordered as the initial states."""
    import state_prep.approximate_states as approx

    return [
        getattr(approx, f"J{j}_singlet"),
        getattr(approx, f"J{j}_triplet_m"),
        getattr(approx, f"J{j}_triplet_0"),
        getattr(approx, f"J{j}_triplet_p"),
    ]


def build_spa_cascade_setup(
    *,
    js: Sequence[int] = (0, 1, 2, 3),
    v_forward: float = 184.0,
    omega_nominal: Optional[float] = None,
):
    """Full SPA1+SPA2 cascade setup with no stray-microwave background.

    The companion to `build_spa2_setup` for work that needs both rungs. J=0 and J=3
    stay in the manifold because the DC electric field couples to them, and the
    nominal 1e-3 magnetic field is kept so the mF sublevels are non-degenerate and
    adiabatic labelling is well defined.
    """
    trajectory, electric_field, magnetic_field, hamiltonian = build_fields_and_hamiltonian(
        js=js,
        v_forward=v_forward,
    )
    mf01, mf12, frequencies, positions = build_cascade_microwaves(
        hamiltonian, omega_nominal=omega_nominal
    )
    return {
        "trajectory": trajectory,
        "electric_field": electric_field,
        "magnetic_field": magnetic_field,
        "hamiltonian": hamiltonian,
        "microwave_fields": [mf01, mf12],
        "frequencies": frequencies,
        "positions": positions,
        "initial_states": spa_cascade_initial_states(),
        "targets_j1": spa_cascade_targets(1),
        "targets_j2": spa_cascade_targets(2),
    }


def compute_eb_t(*, trajectory: Any, electric_field: Any, magnetic_field: Any, t_array: np.ndarray):
    e_t = np.empty((t_array.size, 3), dtype=float)
    b_t = np.empty((t_array.size, 3), dtype=float)
    for idx, t in enumerate(t_array):
        position = trajectory.R_t(float(t))
        e_t[idx] = electric_field.E_R(position)
        b_t[idx] = magnetic_field.B_R(position)
    return e_t, b_t


def make_d_mu_diag_batch(
    *,
    microwave_fields: Sequence[Any],
    qn: Sequence[Any],
    detunings_hz_batched: np.ndarray,
) -> np.ndarray:
    """Rotating-frame diagonal shift, delegating to the package implementation.

    This used to be a third independent copy of the construction. Benchmarks must
    exercise the same rotating-frame convention as the code under test, or a
    change to one silently invalidates the other.
    """
    from state_prep.microwaves import build_rotating_frame_shift

    d_mu_diag_batch, _, _ = build_rotating_frame_shift(
        list(microwave_fields), list(qn), detunings_hz_batched
    )
    return d_mu_diag_batch


def microwave_base_matrices_np(
    microwave_fields: Sequence[Any],
    *,
    trajectory: Any,
    t0: float,
    n: int,
) -> np.ndarray:
    h_base = np.empty((len(microwave_fields), n, n), dtype=np.complex128)
    for field_idx, field in enumerate(microwave_fields):
        field.get_H_t_func(trajectory.R_t, field.QN)
        polarization = np.asarray(field.p_t(float(t0)), dtype=np.complex128)
        h_sum = np.zeros((n, n), dtype=np.complex128)
        for axis in range(3):
            h_axis = np.asarray(field.H_list[axis], dtype=np.complex128)
            h_sum += polarization[axis] * np.triu(h_axis)
            h_sum += np.conj(polarization[axis]) * np.tril(h_axis)
        h_base[field_idx] = h_sum
    return h_base


def microwave_amp_t_np(
    microwave_fields: Sequence[Any],
    *,
    trajectory: Any,
    t_array: np.ndarray,
) -> np.ndarray:
    from state_prep.microwaves import XConstants

    for field in microwave_fields:
        field.get_H_t_func(trajectory.R_t, field.QN)

    amp_t = np.empty((t_array.size - 1, len(microwave_fields)), dtype=np.complex128)
    for time_idx, t in enumerate(t_array[:-1]):
        for field_idx, field in enumerate(microwave_fields):
            e_field = float(field.E_t(float(t)))
            amp_t[time_idx, field_idx] = 2 * np.pi * XConstants.D_TlF * e_field / 2.0
    return amp_t


def result_row(case: str, status: str = "ok", **values: Any) -> dict[str, Any]:
    row = {"case": case, "status": status}
    row.update(values)
    return row


# ---------------------------------------------------------------------------
# Lineshape scoring
# ---------------------------------------------------------------------------


def align_lineshape(
    curve: np.ndarray,
    reference: np.ndarray,
    detunings_khz: np.ndarray,
    *,
    max_shift_khz: float = 3.0,
    samples: int = 12001,
) -> tuple[float, float]:
    """Best-fit detuning shift of `curve` onto `reference`, and the residual.

    Comparing populations at a *fixed* detuning is ill-conditioned wherever a
    scan point sits on a steep part of the lineshape. On the SPA2 setup
    `d(population)/d(detuning)` reaches `6e-03` per kHz, so a sub-kHz effective
    error reads as a `1e-03` population difference and no amount of refinement
    makes it converge. Comparing whole lineshapes and reporting the shift
    separates "the curve moved" from "the curve changed shape", and the residual
    then converges cleanly where the pointwise metric oscillates.

    Returns `(shift_khz, residual)`. The residual is RMS over the window after
    alignment, so it is the part that a detuning offset cannot explain.
    """
    curve = np.asarray(curve, dtype=float)
    reference = np.asarray(reference, dtype=float)
    detunings_khz = np.asarray(detunings_khz, dtype=float)

    grid = np.linspace(-max_shift_khz, max_shift_khz, int(samples))
    errors = [
        np.sum((np.interp(detunings_khz + shift, detunings_khz, reference) - curve) ** 2)
        for shift in grid
    ]
    index = int(np.argmin(errors))
    shift = float(grid[index])
    if index in (0, len(grid) - 1):
        # The optimum sat on the boundary, so the true shift is at least this
        # large and the residual below is an overestimate. Silently returning a
        # clipped fit would look like a converged alignment.
        raise ValueError(
            f"best-fit shift hit the +/-{max_shift_khz} kHz search boundary; "
            "raise max_shift_khz"
        )
    aligned = np.interp(detunings_khz + shift, detunings_khz, reference)
    return shift, float(np.sqrt(np.mean((aligned - curve) ** 2)))


def convergence_order(coarse: np.ndarray, medium: np.ndarray, fine: np.ndarray) -> float:
    """Observed order from a self-convergence triple at `N`, `2N`, `4N`.

    `p = log2(||u(N) - u(2N)|| / ||u(2N) - u(4N)||)`. Uses no reference at all,
    which is the point: every reference available here carries an error
    comparable to what is being measured at large `N_steps`, and a reference
    built from a finer run of the *same* scheme flatters that scheme through
    correlated error structure.
    """
    first = float(np.linalg.norm(np.asarray(coarse) - np.asarray(medium)))
    second = float(np.linalg.norm(np.asarray(medium) - np.asarray(fine)))
    if second <= 0.0:
        return float("nan")
    return float(np.log2(first / second))


def magnus_step_norms(muw_hams, t_array, *, coupling_scale: float = 1.0) -> np.ndarray:
    """`||A|| = ||H_mu(t_mid)|| * dt` on each step of `t_array`.

    This is the quantity that governs the interaction-picture Magnus propagator:
    it Taylor-expands `expm(-i A)`, so its accuracy and its cost both depend on
    `||A||`. At the SPA2 operating point `||A||` is about `0.038` rad, where a
    short series is ample. Measured against exact solutions, a fixed eight-term
    series loses `10x` accuracy **silently** near `||A|| ~ 1.9` and overflows to
    NaN near `19`; scaling and squaring now handles that, at the cost of `2**k`
    applications per step.

    **Why this needs checking when a graded grid is used.** Grading stretches
    steps -- up to `50x` uniform on SPA2 at `order=1` -- and `||A||` scales with
    `dt`, so a long step landing where the coupling is strong drives `||A||` up.
    Whether that happens is pure geometry: if the density peaks where the beam
    is, the grid puts *short* steps there and `||A||` falls (measured `0.038 ->
    0.0068` on the `engineered` model); if the density peaks elsewhere, as on
    SPA2 where the Stark ramp dominates, the long steps land away from the beam
    but `max ||A||` still rose from `0.044` to `0.32`.

    Needs no propagation, so it is cheap enough to assert before a run rather
    than discover afterwards.
    """
    t_array = np.asarray(t_array, dtype=float)
    steps = np.diff(t_array)
    mids = 0.5 * (t_array[:-1] + t_array[1:])
    # Sample the coupling on a coarse probe and interpolate: it is smooth, and
    # evaluating it per step would cost as much as the run being guarded.
    probe = np.linspace(t_array[0], t_array[-1], min(600, mids.size))
    norms = np.array(
        [np.linalg.norm(sum(H(float(t)) for H in muw_hams), 2) for t in probe]
    )
    return np.interp(mids, probe, norms) * steps * float(coupling_scale)


# --- SPA singlet-vs-triplet verification observables -------------------------
#
# Lifted from `examples/SPA/Experimental verification/SPA - singlet vs triplet,
# no background.ipynb` (cell 7) so that a benchmark scores exactly what the
# report quotes, rather than a reimplementation that drifts from it.
#
# The notebook identifies states by their quantum numbers **at readout**, which
# is label-path independent and therefore safe to compare across `N_steps`.
# `scripts/make_singlet_triplet_report_figures.py` still uses a t=0 tracked
# index, which is not; adopting this helper there is a separate change.


def cascade_manifold_of_eigenstates(V: np.ndarray, j_of_basis: np.ndarray) -> np.ndarray:
    """Rotational manifold each eigenvector of `V` predominantly belongs to."""
    weight = np.abs(V) ** 2
    js = np.unique(j_of_basis)
    per_j = np.array([weight[j_of_basis == j].sum(axis=0) for j in js])
    return js[per_j.argmax(axis=0)]


def cascade_readout_identities(
    setup: dict,
    targets: Sequence[Any],
    *,
    n_steps: int = 10_000,
    time_sampling: str = "mid",
):
    """`(J, F1, F, mF)` each target has become once the Stark field has ramped off.

    Measured, not assumed: at `t=0` the Stark field mixes `F` badly, so a target's
    `F` is only a good quantum number at readout. Propagating the DC fields with
    the microwaves off answers it in one run for all targets.

    This runs `Simulator.run` with `microwave_fields=None`, which dispatches to
    `_time_evolve`. That loop propagates in `H_slow`'s own instantaneous
    eigenbasis, where `H_slow` is diagonal, so there is no off-diagonal part for a
    Magnus step to integrate and no `propagator` argument to pass -- `Magnus`
    would reduce to exactly the phases already applied.

    **These identities sit upstream of every reported transfer.** If they move
    with `n_steps`, every `matched` population silently changes meaning, so
    `bench_spa_verification.py --phase0a` asserts they do not.
    """
    from state_prep import Simulator

    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        list(targets),
        setup["hamiltonian"],
        None,
    )
    ref = simulator.run(
        N_steps=n_steps,
        store_probabilities=False,
        store_final_probabilities=True,
        progress=False,
        time_sampling=time_sampling,
    )
    table = ref.final_quantum_numbers()
    identities = []
    for s in range(len(targets)):
        idx = int(np.argmax(ref.probabilities_final[s]))
        identities.append({k: float(table[k][idx]) for k in ("J", "F1", "F", "mF")})
    return identities


def cascade_transfer_observables(
    result: Any,
    setup: dict,
    targets: Sequence[Any],
    j_target: int,
    identities: Sequence[dict],
    *,
    tolerance: float = 0.05,
) -> dict[str, np.ndarray]:
    """`matched` / `tracked` / `manifold` / `spread` final populations.

    `matched` is the reported number: population of the eigenstate carrying the
    target's readout quantum numbers, identified in the basis the populations are
    actually expressed in. `tracked` reads the same population through a `t=0`
    index, i.e. the adiabatically continued label. That is a correct and useful
    quantity within a single run -- it is what makes population moving *between*
    labels visible, and the two agree to `0.000e+00` here at `N_steps = 10_000`.
    What it does not support is comparison **across** step counts: near a
    crossing the tracker does not resolve, the label's meaning shifts with `N`,
    and the notebook records the J=2 singlet moving `32 -> 29` at `80_000`. Use
    `matched` for anything that varies `N_steps`.

    `spread` is `max - min` across the target sublevels. It is the report's actual
    physics claim (`1.7e-04`), it is a difference of nearly equal numbers, and it
    is the quantity whose convergence matters -- converging the individual
    transfers does not imply it, because the errors may or may not be common-mode.

    `tolerance` matches the notebook's `IDENTITY_TOL`: `F1` stays partly mixed at
    readout (singlets at `1.4922 +- 0.0883`), while the competing level sits a
    full `1.0` away, so `0.05` is far above the drift and far below the gap.
    """
    from state_prep.utils import find_max_overlap_idx, select_eigenstate

    qn = setup["hamiltonian"].QN
    j_of_basis = np.array([q.J for q in qn])
    probs = result.probabilities_final
    V = result.V_ini
    table = result.final_quantum_numbers()
    manifold = np.flatnonzero(cascade_manifold_of_eigenstates(V, j_of_basis) == j_target)

    matched, tracked = [], []
    for s, (target, identity) in enumerate(zip(targets, identities)):
        matched.append(probs[:, s, select_eigenstate(table, identity, tolerance=tolerance)])
        tracked.append(probs[:, s, find_max_overlap_idx(target.state_vector(qn), V)])
    matched_arr = np.stack(matched, axis=-1)
    return {
        "matched": matched_arr,
        "tracked": np.stack(tracked, axis=-1),
        "manifold": probs[:, :, manifold].sum(axis=-1),
        "spread": matched_arr.max(axis=-1) - matched_arr.min(axis=-1),
    }
