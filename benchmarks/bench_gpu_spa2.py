from __future__ import annotations

import argparse
import time
from typing import Any, Optional

import numpy as np

from common import (
    build_spa2_setup,
    compute_eb_t,
    make_d_mu_diag_batch,
    make_detunings,
    make_payload,
    microwave_amp_t_np,
    microwave_base_matrices_np,
    resolve_csv_path,
    result_row,
    time_call,
    write_results,
)
from paired import compare_paired


def run_cpu(
    setup: dict[str, Any],
    *,
    n_steps: int,
    detunings: np.ndarray,
    intensity_prefactors: np.ndarray,
    workers: int = 1,
):
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
        detunings_hz=detunings,
        intensity_prefactors=intensity_prefactors,
        N_steps=n_steps,
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        workers=workers,
    )


def make_torch_runner(
    setup: dict[str, Any],
    *,
    n_steps: int,
    detunings: np.ndarray,
    intensity_prefactors: np.ndarray,
    complex_dtype: str = "complex128",
    block_eigh: bool | str = "auto",
    slow_eigh_chunk_size: Optional[int] = None,
):
    """Prepare a full PyTorch GPU propagation runner for the SPA2 scan."""
    import centrex_tlf
    import torch
    from state_prep.simulator import Simulator
    from state_prep_gpu.simulator import simulate_batched_scalar_microwaves_shared_slow
    from state_prep_gpu.slow_hamiltonian import terms_from_centrex_tlf

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is unavailable")

    hamiltonian = setup["hamiltonian"]
    trajectory = setup["trajectory"]
    microwave_fields = setup["microwave_fields"]
    device = torch.device("cuda")
    dtype = getattr(torch, complex_dtype)
    batch = int(detunings.shape[0])
    terms = terms_from_centrex_tlf(
        centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(
            cast_any(hamiltonian.QN)
        ),
        dtype=dtype,
        device=device,
    )

    t_array = np.linspace(0.0, float(trajectory.get_T()), int(n_steps))
    dt = float(t_array[1] - t_array[0])
    t_mid = 0.5 * (t_array[:-1] + t_array[1:])
    sample_times = np.concatenate((t_mid, t_array[-1:]))
    e_t, b_t = compute_eb_t(
        trajectory=trajectory,
        electric_field=setup["electric_field"],
        magnetic_field=setup["magnetic_field"],
        t_array=sample_times,
    )
    e0, b0 = compute_eb_t(
        trajectory=trajectory,
        electric_field=setup["electric_field"],
        magnetic_field=setup["magnetic_field"],
        t_array=np.array([0.0]),
    )

    simulator = Simulator(
        trajectory,
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        hamiltonian,
        microwave_fields,
    )
    simulator.init_state_vecs(hamiltonian.get_H_t_func()(0.0))
    psi0 = np.repeat(simulator.psis[None, :, :], batch, axis=0)
    monitor_vectors = np.stack(
        [state.state_vector(hamiltonian.QN) for state in setup["monitor_states"]],
        axis=0,
    )
    d_mu_diag = make_d_mu_diag_batch(
        microwave_fields=microwave_fields,
        qn=hamiltonian.QN,
        detunings_hz_batched=detunings,
    )
    h_mu_fields = microwave_base_matrices_np(
        microwave_fields, trajectory=trajectory, t0=float(t_mid[0]), n=len(hamiltonian.QN)
    )
    amp_t = microwave_amp_t_np(
        microwave_fields, trajectory=trajectory, t_array=sample_times
    )
    inputs = {
        "E_t": torch.as_tensor(e_t, dtype=torch.float64, device=device),
        "B_t": torch.as_tensor(b_t, dtype=torch.float64, device=device),
        "psi0": torch.as_tensor(psi0, dtype=dtype, device=device),
        "H_mu_base_fields": torch.as_tensor(h_mu_fields, dtype=dtype, device=device),
        "amp_t": torch.as_tensor(amp_t, dtype=dtype, device=device),
        "coupling_scales": torch.as_tensor(
            np.sqrt(intensity_prefactors), dtype=torch.float64, device=device
        ),
        "D_mu_diag_batch": torch.as_tensor(d_mu_diag, dtype=dtype, device=device),
        "E0": torch.as_tensor(e0[0], dtype=torch.float64, device=device),
        "B0": torch.as_tensor(b0[0], dtype=torch.float64, device=device),
        "monitor_state_vectors": torch.as_tensor(monitor_vectors, dtype=dtype, device=device),
    }

    def run(
        store_every: Optional[int] = None,
        profile_timings: Optional[dict[str, list[float]]] = None,
    ):
        return simulate_batched_scalar_microwaves_shared_slow(
            terms=terms,
            dt=dt,
            return_final_probabilities=True,
            store_every=store_every,
            profile_timings=profile_timings,
            block_eigh=block_eigh,
            slow_eigh_chunk_size=slow_eigh_chunk_size,
            **inputs,
        )

    return run, lambda: torch.cuda.synchronize(device)


def make_gpu_runner(
    setup: dict[str, Any],
    *,
    n_steps: int,
    detunings: np.ndarray,
    intensity_prefactors: np.ndarray,
    complex_dtype: str = "complex128",
    eigh_backend: str = "torch",
):
    if eigh_backend != "torch":
        raise ValueError(
            "The production GPU backend now uses PyTorch throughout; "
            "the CuPy propagation backend has been removed."
        )
    import torch

    run, sync = make_torch_runner(
        setup,
        n_steps=n_steps,
        detunings=detunings,
        intensity_prefactors=intensity_prefactors,
        complex_dtype=complex_dtype,
    )
    return run, sync, torch

def make_scan_parameters(batch: int, mode: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build two-field detuning/power rows with exactly `batch` points."""
    if mode == "detuning":
        n_detuning, n_power = batch, 1
    elif mode == "power":
        n_detuning, n_power = 1, batch
    elif mode == "combined":
        n_detuning = int(np.sqrt(batch))
        while batch % n_detuning:
            n_detuning -= 1
        n_power = batch // n_detuning
    else:
        raise ValueError(f"Unknown scan mode {mode!r}")

    detuning_axis = make_detunings(n_detuning) if n_detuning > 1 else np.zeros(1)
    power_axis = np.linspace(0.25, 4.0, n_power) if n_power > 1 else np.ones(1)
    detuning_values = np.repeat(detuning_axis, n_power)
    power_values = np.tile(power_axis, n_detuning)
    detunings = np.repeat(detuning_values[:, None], 2, axis=1)
    intensity_prefactors = np.repeat(power_values[:, None], 2, axis=1)
    return detunings, intensity_prefactors, n_detuning, n_power


def cast_any(value: Any) -> Any:
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SPA2-like CPU/GPU scans.")
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--batch-sweep", type=int, nargs="*")
    parser.add_argument(
        "--scan-modes",
        choices=("detuning", "power", "combined"),
        nargs="+",
        default=["detuning", "power", "combined"],
        help="Parameter axes to compare at each batch size.",
    )
    parser.add_argument(
        "--initial-state-sweep",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Number of initial states propagated together in each scan point.",
    )
    parser.add_argument(
        "--store-every",
        type=int,
        nargs="+",
        default=[0],
        help="GPU state-history strides to benchmark; 0 means final-only, 1 stores every state.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    run_cpu_flag = args.cpu or not args.gpu
    run_gpu_flag = args.gpu or not args.cpu
    batches = args.batch_sweep if args.batch_sweep else [args.batch]
    if any(batch < 1 for batch in batches):
        parser.error("batch sizes must be >= 1")
    if any(count < 1 for count in args.initial_state_sweep):
        parser.error("initial-state counts must be >= 1")
    if any(stride < 0 for stride in args.store_every):
        parser.error("--store-every values must be >= 0 (0 means final-only)")
    storage_strides = list(dict.fromkeys(args.store_every))
    results = []
    # Building the SPA2 field/Hamiltonian setup is CPU-heavy and is identical
    # across scan modes and batch sizes. Reuse it so GPU-only runs benchmark
    # propagation rather than repeating the setup for every grid cell.
    print("[setup] building the real SPA2 system on CPU", flush=True)
    setup_started = time.perf_counter()
    setup = build_spa2_setup()
    all_initial_states = setup["initial_states"]
    print(
        f"[setup] ready in {time.perf_counter() - setup_started:.1f}s; "
        f"Hilbert dimension n={len(setup['hamiltonian'].QN)}",
        flush=True,
    )

    for batch in batches:
        for scan_mode in args.scan_modes:
            detunings, intensity_prefactors, n_detuning, n_power = make_scan_parameters(
                batch, scan_mode
            )
            for n_initial in args.initial_state_sweep:
                if not 1 <= n_initial <= len(all_initial_states):
                    parser.error(
                        f"initial-state count {n_initial} is outside "
                        f"[1, {len(all_initial_states)}]"
                    )
                setup["initial_states"] = all_initial_states[:n_initial]
                cpu_result: Optional[Any] = None
                cpu_timing: Optional[dict[str, Any]] = None
                cpu_run = lambda: run_cpu(
                    setup,
                    n_steps=args.n_steps,
                    detunings=detunings,
                    intensity_prefactors=intensity_prefactors,
                    workers=args.cpu_workers,
                )

                if run_gpu_flag:
                    try:
                        print(
                            f"[gpu] batch={batch} scan={scan_mode} "
                            f"initial_states={n_initial}",
                            flush=True,
                        )
                        gpu_run, sync = make_torch_runner(
                            setup,
                            n_steps=args.n_steps,
                            detunings=detunings,
                            intensity_prefactors=intensity_prefactors,
                        )
                        for stride in storage_strides:
                            store_every = None if stride == 0 else stride
                            if run_cpu_flag:
                                paired = compare_paired(
                                    {
                                        "cpu": cpu_run,
                                        "gpu": lambda store_every=store_every: gpu_run(
                                            store_every
                                        ),
                                    },
                                    repeat=args.repeat,
                                    warmup=args.warmup,
                                    baseline="cpu",
                                    sync=sync,
                                    progress=True,
                                )
                                cpu_result = paired.results["cpu"]
                                gpu_output = paired.results["gpu"]
                                cpu_timing = {
                                    "time_best_s": paired.timings["cpu"].best_s,
                                    "time_mean_s": paired.timings["cpu"].mean_s,
                                    "time_std_s": paired.timings["cpu"].std_s,
                                    "repeat": args.repeat,
                                    "warmup": args.warmup,
                                }
                                timing = {
                                    "time_best_s": paired.timings["gpu"].best_s,
                                    "time_mean_s": paired.timings["gpu"].mean_s,
                                    "time_std_s": paired.timings["gpu"].std_s,
                                    "repeat": args.repeat,
                                    "warmup": args.warmup,
                                }
                                cpu_row = result_row(
                                    "cpu_shared_slow",
                                    **cpu_timing,
                                    batch=batch,
                                    scan_mode=scan_mode,
                                    n_detuning=n_detuning,
                                    n_power=n_power,
                                    initial_states=n_initial,
                                    n=len(setup["hamiltonian"].QN),
                                    workers=args.cpu_workers,
                                )
                                results.append(cpu_row)
                            else:
                                timing, gpu_output = time_call(
                                    lambda store_every=store_every: gpu_run(store_every),
                                    warmup=args.warmup,
                                    repeat=args.repeat,
                                    sync=sync,
                                )
                            history = gpu_output.psi_history
                            tensor_bytes = lambda value: int(value.numel() * value.element_size())
                            final_output_bytes = sum(
                                tensor_bytes(value)
                                for value in (
                                    gpu_output.psi_final,
                                    gpu_output.probabilities_final,
                                    gpu_output.monitor_probabilities_final,
                                    gpu_output.last_evecs,
                                )
                                if value is not None
                            )
                            row = result_row(
                                "pytorch_shared_slow",
                                **timing,
                                batch=batch,
                                scan_mode=scan_mode,
                                n_detuning=n_detuning,
                                n_power=n_power,
                                initial_states=n_initial,
                                n=len(setup["hamiltonian"].QN),
                                store_every=stride,
                                history_samples=(
                                    0 if history is None else int(history.shape[0])
                                ),
                                final_output_bytes=final_output_bytes,
                                psi_history_bytes=(
                                    0 if history is None else tensor_bytes(history)
                                ),
                            )
                            if cpu_result is not None:
                                gpu_probs = gpu_output.probabilities_final.detach().cpu().numpy()
                                row["max_abs_tracked_probability_diff"] = float(
                                    np.max(
                                        np.abs(
                                            cpu_result.probabilities_final - gpu_probs
                                        )
                                    )
                                )
                                cpu_psi = np.asarray(cpu_result.psis_final)
                                gpu_psi = gpu_output.psi_final.detach().cpu().numpy()
                                overlap = np.sum(np.conj(cpu_psi) * gpu_psi, axis=-1)
                                norm_product = np.sum(np.abs(cpu_psi) ** 2, axis=-1) * np.sum(
                                    np.abs(gpu_psi) ** 2, axis=-1
                                )
                                fidelity = np.abs(overlap) ** 2 / norm_product
                                row["max_state_infidelity"] = float(
                                    np.max(np.maximum(0.0, 1.0 - fidelity))
                                )
                            if (
                                stride == 0
                                and cpu_timing is not None
                                and timing["time_mean_s"] > 0
                            ):
                                row["speedup_vs_cpu_mean"] = (
                                    cpu_timing["time_mean_s"] / timing["time_mean_s"]
                                )
                            results.append(row)
                            print(
                                f"[gpu] store_every={stride}: "
                                f"{timing['time_mean_s']:.4f}s mean "
                                f"({timing['time_std_s']:.4f}s std)",
                                flush=True,
                            )
                    except Exception as exc:
                        results.append(
                            result_row(
                                "pytorch_shared_slow",
                                status="skipped",
                                batch=batch,
                                scan_mode=scan_mode,
                                initial_states=n_initial,
                                n_steps=args.n_steps,
                                error=repr(exc),
                            )
                        )

                # Keep the CPU baseline available if GPU setup or propagation
                # failed before a paired result could be collected.
                if run_cpu_flag and cpu_timing is None:
                    timing, cpu_result = time_call(
                        cpu_run,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    results.append(
                        result_row(
                            "cpu_shared_slow",
                            **timing,
                            batch=batch,
                            scan_mode=scan_mode,
                            n_detuning=n_detuning,
                            n_power=n_power,
                            initial_states=n_initial,
                            n=len(setup["hamiltonian"].QN),
                            workers=args.cpu_workers,
                        )
                    )

    payload = make_payload(
        benchmark="gpu_spa2",
        config={
            "n_steps": args.n_steps,
            "batch": args.batch,
            "batch_sweep": batches,
            "scan_modes": args.scan_modes,
            "initial_state_sweep": args.initial_state_sweep,
            "store_every": storage_strides,
            "cpu": run_cpu_flag,
            "gpu": run_gpu_flag,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "cpu_workers": args.cpu_workers,
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_spa2"),
    )


if __name__ == "__main__":
    main()

