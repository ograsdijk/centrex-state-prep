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
