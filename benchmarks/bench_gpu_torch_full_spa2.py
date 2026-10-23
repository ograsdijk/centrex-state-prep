from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from bench_gpu_spa2 import make_scan_parameters, run_cpu
from common import (
    build_spa2_setup,
    compute_eb_t,
    make_d_mu_diag_batch,
    make_payload,
    microwave_amp_t_np,
    microwave_base_matrices_np,
    resolve_csv_path,
    result_row,
    write_results,
)
from paired import compare_paired, format_report
from torch_simulator import simulate_torch_shared_slow


def make_torch_runner(
    setup: dict[str, Any],
    *,
    n_steps: int,
    detunings: np.ndarray,
    intensity_prefactors: np.ndarray,
    time_chunk_size: int = 1,
    scan_chunk_size: int | None = None,
    propagator: str = "eigh",
    all_states: bool = False,
    eigen_blocks: list[Any] | None = None,
):
    import centrex_tlf
    from state_prep.simulator import Simulator

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")

    device = torch.device("cuda")
    hamiltonian = setup["hamiltonian"]
    trajectory = setup["trajectory"]
    microwave_fields = setup["microwave_fields"]
    batch = int(detunings.shape[0])
    n = len(hamiltonian.QN)

    print("[torch setup] generating and transferring Hamiltonian terms", flush=True)
    h_uncoupled = centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(
        list(hamiltonian.QN)
    )
    terms = tuple(
        torch.as_tensor(
            np.asarray(getattr(h_uncoupled, name)),
            dtype=torch.complex128,
            device=device,
        )
        for name in ("Hff", "HSx", "HSy", "HSz", "HZx", "HZy", "HZz")
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
    if all_states:
        psi0_np = np.broadcast_to(
            np.eye(n, dtype=np.complex128), (batch, n, n)
        ).copy()
    else:
        psi0_np = np.repeat(simulator.psis[None, :, :], batch, axis=0)
    monitor_np = np.stack(
        [state.state_vector(hamiltonian.QN) for state in setup["monitor_states"]],
        axis=0,
    )
    d_mu_diag_np = make_d_mu_diag_batch(
        microwave_fields=microwave_fields,
        qn=hamiltonian.QN,
        detunings_hz_batched=detunings,
    )
    h_mu_np = microwave_base_matrices_np(
        microwave_fields,
        trajectory=trajectory,
        t0=float(t_mid[0]),
        n=n,
    )
    amp_np = microwave_amp_t_np(
        microwave_fields, trajectory=trajectory, t_array=sample_times
    )
    coupling_np = np.sqrt(intensity_prefactors)
    print("[torch setup] inputs ready; evolution remains in PyTorch", flush=True)

    inputs = {
        "E_t": torch.as_tensor(e_t, dtype=torch.float64, device=device),
        "B_t": torch.as_tensor(b_t, dtype=torch.float64, device=device),
        "psi0": torch.as_tensor(psi0_np, dtype=torch.complex128, device=device),
        "H_mu_base_fields": torch.as_tensor(
            h_mu_np, dtype=torch.complex128, device=device
        ),
        "amp_t": torch.as_tensor(amp_np, dtype=torch.complex128, device=device),
        "coupling_scales": torch.as_tensor(
            coupling_np, dtype=torch.float64, device=device
        ),
        "D_mu_diag_batch": torch.as_tensor(
            d_mu_diag_np, dtype=torch.complex128, device=device
        ),
        "E0": torch.as_tensor(e0[0], dtype=torch.float64, device=device),
        "B0": torch.as_tensor(b0[0], dtype=torch.float64, device=device),
        "monitor_state_vectors": torch.as_tensor(
            monitor_np, dtype=torch.complex128, device=device
        ),
    }

    def run():
        return simulate_torch_shared_slow(
            terms=terms,
            dt=dt,
            time_chunk_size=time_chunk_size,
            scan_chunk_size=scan_chunk_size,
            propagator=propagator,
            eigen_blocks=eigen_blocks,
            **inputs,
        )

    return run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired CPU versus full-PyTorch SPA2 benchmark."
    )
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument(
        "--scan-mode", choices=("detuning", "power", "combined"), default="detuning"
    )
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    print("[setup] building real SPA2 setup", flush=True)
    setup = build_spa2_setup()
    initial_states = setup["initial_states"]
    if not 1 <= args.initial_states <= len(initial_states):
        parser.error(f"--initial-states must be between 1 and {len(initial_states)}")
    setup["initial_states"] = initial_states[: args.initial_states]
    detunings, powers, n_detuning, n_power = make_scan_parameters(
        args.batch, args.scan_mode
    )

    baseline_name = "cpu"
    baseline_run = lambda: run_cpu(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
    )
    torch_run = make_torch_runner(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
    )

    def sync_both() -> None:
        torch.cuda.synchronize()

    def max_probability_difference(reference, candidate) -> float:
        candidate_np = candidate.probabilities_final.detach().cpu().numpy()
        return float(np.max(np.abs(reference.probabilities_final - candidate_np)))

    paired = compare_paired(
        {baseline_name: baseline_run, "torch_all": torch_run},
        repeat=args.repeat,
        warmup=args.warmup,
        baseline=baseline_name,
        sync=sync_both,
        check=max_probability_difference,
        tolerance=1e-5,
        progress=True,
    )
    print(format_report(paired), flush=True)

    cpu_out = paired.results[baseline_name]
    torch_out = paired.results["torch_all"]
    cpu_psi = np.asarray(cpu_out.psis_final)
    torch_psi = torch_out.psi_final.detach().cpu().numpy()
    overlap = np.sum(np.conj(cpu_psi) * torch_psi, axis=-1)
    norm_cpu = np.sum(np.abs(cpu_psi) ** 2, axis=-1)
    norm_torch = np.sum(np.abs(torch_psi) ** 2, axis=-1)
    fidelity = np.abs(overlap) ** 2 / (norm_cpu * norm_torch)
    max_infidelity = float(np.max(np.maximum(0.0, 1.0 - fidelity)))
    max_probability_diff = max_probability_difference(cpu_out, torch_out)
    print(
        f"max final probability difference={max_probability_diff:.3e}; "
        f"max state infidelity={max_infidelity:.3e}",
        flush=True,
    )

    results = []
    for name, timing in paired.timings.items():
        results.append(
            result_row(
                name,
                time_best_s=timing.best_s,
                time_mean_s=timing.mean_s,
                time_std_s=timing.std_s,
                repeat=args.repeat,
                warmup=args.warmup,
                batch=args.batch,
                scan_mode=args.scan_mode,
                n_detuning=n_detuning,
                n_power=n_power,
                initial_states=args.initial_states,
                n=len(setup["hamiltonian"].QN),
                n_steps=args.n_steps,
                max_probability_difference=max_probability_diff,
                max_state_infidelity=max_infidelity,
            )
        )
    results[1]["speedup_vs_baseline_mean"] = (
        paired.timings[baseline_name].mean_s / paired.timings["torch_all"].mean_s
    )
    payload = make_payload(
        benchmark="gpu_torch_full_spa2",
        config={
            "n_steps": args.n_steps,
            "batch": args.batch,
            "scan_mode": args.scan_mode,
            "initial_states": args.initial_states,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_torch_full_spa2"),
    )


if __name__ == "__main__":
    main()
