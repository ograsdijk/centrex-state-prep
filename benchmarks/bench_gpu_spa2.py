from __future__ import annotations

import argparse
from typing import Any, Optional

import numpy as np

from common import (
    add_nvidia_dll_paths,
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


def run_cpu(setup: dict[str, Any], *, n_steps: int, detunings: np.ndarray):
    from state_prep.simulator import Simulator

    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        setup["microwave_fields"],
    )
    detunings_batched = np.column_stack([detunings, detunings])
    intensity_prefactors = np.ones((detunings.size, 2), dtype=float)
    return simulator.run_microwave_scan(
        detunings_hz=detunings_batched,
        intensity_prefactors=intensity_prefactors,
        N_steps=n_steps,
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
    )


def make_gpu_runner(setup: dict[str, Any], *, n_steps: int, detunings: np.ndarray):
    add_nvidia_dll_paths()
    import centrex_tlf
    from state_prep.simulator import Simulator
    from state_prep_gpu._cupy import cupy
    from state_prep_gpu.simulator import simulate_batched_scalar_microwaves_shared_slow
    from state_prep_gpu.slow_hamiltonian import terms_from_centrex_tlf

    cp = cupy()
    hamiltonian = setup["hamiltonian"]
    trajectory = setup["trajectory"]
    microwave_fields = setup["microwave_fields"]
    batch = int(detunings.size)

    terms = terms_from_centrex_tlf(
        centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(cast_any(hamiltonian.QN)),
        dtype=cp.complex128,
    )
    t_array = np.linspace(0.0, float(trajectory.get_T()), int(n_steps))
    dt = float(t_array[1] - t_array[0])
    e_t, b_t = compute_eb_t(
        trajectory=trajectory,
        electric_field=setup["electric_field"],
        magnetic_field=setup["magnetic_field"],
        t_array=t_array,
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
    detunings_batched = np.column_stack([detunings, detunings])
    d_mu_diag_batch = make_d_mu_diag_batch(
        microwave_fields=microwave_fields,
        qn=hamiltonian.QN,
        detunings_hz_batched=detunings_batched,
    )
    h_mu_base_fields = microwave_base_matrices_np(
        microwave_fields,
        trajectory=trajectory,
        t0=float(t_array[0]),
        n=len(hamiltonian.QN),
    )
    amp_t = microwave_amp_t_np(microwave_fields, trajectory=trajectory, t_array=t_array)
    coupling_scales = np.ones((batch, 2), dtype=float)

    def run():
        return simulate_batched_scalar_microwaves_shared_slow(
            terms=terms,
            E_t=e_t,
            B_t=b_t,
            psi0=psi0,
            dt=dt,
            H_mu_base_fields=h_mu_base_fields,
            amp_t=amp_t,
            coupling_scales=coupling_scales,
            D_mu_diag_batch=d_mu_diag_batch,
            monitor_state_vectors=monitor_vectors,
            return_final_probabilities=True,
        )

    return run, lambda: cp.cuda.Stream.null.synchronize(), cp


def cast_any(value: Any) -> Any:
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SPA2-like CPU/GPU scans.")
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--batch-sweep", type=int, nargs="*")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    run_cpu_flag = args.cpu or not args.gpu
    run_gpu_flag = args.gpu or not args.cpu
    batches = args.batch_sweep if args.batch_sweep else [args.batch]
    results = []

    for batch in batches:
        detunings = make_detunings(batch)
        setup = build_spa2_setup()
        cpu_result: Optional[Any] = None
        cpu_best: Optional[float] = None

        if run_cpu_flag:
            timing, cpu_result = time_call(
                lambda: run_cpu(setup, n_steps=args.n_steps, detunings=detunings),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            cpu_best = timing["time_best_s"]
            results.append(
                result_row(
                    "cpu_shared_slow",
                    **timing,
                    batch=batch,
                    n=len(setup["hamiltonian"].QN),
                )
            )

        if run_gpu_flag:
            try:
                gpu_run, sync, cp = make_gpu_runner(setup, n_steps=args.n_steps, detunings=detunings)
                timing, gpu_output = time_call(
                    gpu_run,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    sync=sync,
                )
                row = result_row(
                    "gpu_shared_slow",
                    **timing,
                    batch=batch,
                    n=len(setup["hamiltonian"].QN),
                )
                if cpu_result is not None:
                    gpu_probs = cp.asnumpy(gpu_output.probabilities_final)
                    row["max_abs_probability_diff"] = float(
                        np.max(np.abs(cpu_result.probabilities_final - gpu_probs))
                    )
                if cpu_best is not None and timing["time_best_s"] > 0:
                    row["speedup_vs_cpu"] = cpu_best / timing["time_best_s"]
                results.append(row)
            except Exception as exc:
                results.append(
                    result_row(
                        "gpu_shared_slow",
                        status="skipped",
                        batch=batch,
                        n_steps=args.n_steps,
                        error=repr(exc),
                    )
                )

    payload = make_payload(
        benchmark="gpu_spa2",
        config={
            "n_steps": args.n_steps,
            "batch": args.batch,
            "batch_sweep": batches,
            "cpu": run_cpu_flag,
            "gpu": run_gpu_flag,
            "repeat": args.repeat,
            "warmup": args.warmup,
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

