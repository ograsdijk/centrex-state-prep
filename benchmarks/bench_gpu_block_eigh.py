from __future__ import annotations

import argparse

import numpy as np
import torch

from bench_gpu_spa2 import make_scan_parameters
from common import (
    add_nvidia_dll_paths,
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


def find_components(matrices: list[np.ndarray]) -> list[np.ndarray]:
    n = matrices[0].shape[0]
    adjacency = np.zeros((n, n), dtype=bool)
    for matrix in matrices:
        adjacency |= np.abs(matrix) > 0
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, False)
    seen: set[int] = set()
    components = []
    for start in range(n):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return sorted(components, key=lambda block: int(block[0]))


def make_rotating_frame_batch(batch: int, n_steps: int):
    import centrex_tlf

    setup = build_spa2_setup()
    qn = list(setup["hamiltonian"].QN)
    n = len(qn)
    slow = centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(qn)
    all_terms = [
        np.asarray(getattr(slow, name), dtype=np.complex128)
        for name in ("Hff", "HSx", "HSy", "HSz", "HZx", "HZy", "HZz")
    ]
    detunings, powers, _, _ = make_scan_parameters(batch, "detuning")

    t_array = np.linspace(0.0, float(setup["trajectory"].get_T()), n_steps)
    t_mid = 0.5 * (t_array[:-1] + t_array[1:])
    sample_times = np.concatenate((t_mid, t_array[-1:]))
    e_t, b_t = compute_eb_t(
        trajectory=setup["trajectory"],
        electric_field=setup["electric_field"],
        magnetic_field=setup["magnetic_field"],
        t_array=sample_times,
    )
    sample_idx = len(t_mid) // 2
    E = e_t[sample_idx]
    B = b_t[sample_idx]
    if np.any(E[:2] != 0) or np.any(B[:2] != 0):
        raise RuntimeError("The selected SPA2 trajectory has transverse fields")

    H_slow = 2 * np.pi * (
        all_terms[0] + E[2] * all_terms[3] + B[2] * all_terms[6]
    )
    h_mu = microwave_base_matrices_np(
        setup["microwave_fields"],
        trajectory=setup["trajectory"],
        t0=float(t_mid[sample_idx]),
        n=n,
    )
    amp_t = microwave_amp_t_np(
        setup["microwave_fields"],
        trajectory=setup["trajectory"],
        t_array=sample_times,
    )
    d_mu = make_d_mu_diag_batch(
        microwave_fields=setup["microwave_fields"],
        qn=qn,
        detunings_hz_batched=detunings,
    )
    coupling_scales = np.sqrt(powers)

    # Only terms active for the z-only SPA2 fields can connect sectors.
    components = find_components([all_terms[0], all_terms[3], all_terms[6], *h_mu])
    d_parts = []
    v_grouped = np.zeros((n, n), dtype=np.complex128)
    labels = []
    col = 0
    for block_id, indices in enumerate(components):
        values, vectors = np.linalg.eigh(H_slow[np.ix_(indices, indices)])
        width = len(indices)
        v_grouped[np.ix_(indices, np.arange(col, col + width))] = vectors
        d_parts.append(values)
        labels.extend([block_id] * width)
        col += width

    values_grouped = np.concatenate(d_parts)
    labels = np.asarray(labels, dtype=np.int64)
    sorted_idx = np.argsort(values_grouped)
    values_sorted = values_grouped[sorted_idx]
    v_sorted = v_grouped[:, sorted_idx]
    labels_sorted = labels[sorted_idx]
    grouped_perm = np.argsort(labels_sorted, kind="stable")
    values_grouped = values_sorted[grouped_perm]
    v_grouped = v_sorted[:, grouped_perm]
    d_mu_grouped = d_mu[:, grouped_perm]

    rotated_fields = np.stack(
        [v_grouped.conj().T @ matrix @ v_grouped for matrix in h_mu], axis=0
    )
    weights = coupling_scales * amp_t[sample_idx][None, :]
    h_rot = np.einsum("bt,tij->bij", weights, rotated_fields)
    h_rot[:, np.arange(n), np.arange(n)] += values_grouped
    h_rot[:, np.arange(n), np.arange(n)] += d_mu_grouped

    offsets = np.cumsum([0, *[len(block) for block in components]])
    offblock_mask = labels_sorted[grouped_perm][:, None] != labels_sorted[grouped_perm][None, :]
    max_offblock = float(np.max(np.abs(h_rot[:, offblock_mask])))
    max_matrix_entry = float(np.max(np.abs(h_rot)))
    return h_rot, offsets, [len(block) for block in components], max_offblock / max_matrix_entry


def dense_np(matrices: np.ndarray) -> np.ndarray:
    values, _ = np.linalg.eigh(matrices)
    return values


def block_np(matrices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    values = [
        np.linalg.eigh(
            np.ascontiguousarray(matrices[:, start:end, start:end])
        )[0]
        for start, end in zip(offsets[:-1], offsets[1:])
    ]
    return np.sort(np.concatenate(values, axis=-1), axis=-1)


def dense_torch(matrices: torch.Tensor) -> torch.Tensor:
    values, _ = torch.linalg.eigh(matrices)
    return values


def block_torch(matrices: torch.Tensor, offsets: np.ndarray) -> torch.Tensor:
    values = [
        torch.linalg.eigh(matrices[:, start:end, start:end])[0]
        for start, end in zip(offsets[:-1], offsets[1:])
    ]
    return torch.sort(torch.cat(values, dim=-1), dim=-1).values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense versus exact blockwise SPA2 rotating-frame eigensolves."
    )
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    add_nvidia_dll_paths()
    h_np, offsets, block_sizes, leakage_ratio = make_rotating_frame_batch(
        args.batch, args.n_steps
    )
    print(
        f"real SPA2 matrices: batch={args.batch}, n=64, blocks={block_sizes}, "
        f"max relative off-block entry={leakage_ratio:.3e}",
        flush=True,
    )

    cpu_matrices = np.ascontiguousarray(h_np)
    cpu_variants = {
        "cpu_dense": lambda: dense_np(cpu_matrices),
        "cpu_blocks": lambda: block_np(cpu_matrices, offsets),
    }
    cpu_pair = compare_paired(
        cpu_variants,
        repeat=args.repeat,
        warmup=args.warmup,
        baseline="cpu_dense",
        check=lambda a, b: float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(a)))),
        tolerance=1e-8,
        progress=True,
    )
    print(format_report(cpu_pair), flush=True)

    gpu_matrices = torch.as_tensor(h_np, dtype=torch.complex128, device="cuda")
    gpu_variants = {
        "gpu_dense": lambda: dense_torch(gpu_matrices),
        "gpu_blocks": lambda: block_torch(gpu_matrices, offsets),
    }
    gpu_pair = compare_paired(
        gpu_variants,
        repeat=args.repeat,
        warmup=args.warmup,
        baseline="gpu_dense",
        sync=torch.cuda.synchronize,
        check=lambda a, b: float(
            torch.max(torch.abs(a - b) / torch.clamp(torch.abs(a), min=1.0)).item()
        ),
        tolerance=1e-8,
        progress=True,
    )
    print(format_report(gpu_pair), flush=True)

    results = []
    for pair in (cpu_pair, gpu_pair):
        comparison = pair.comparisons[0]
        for name, timing in pair.timings.items():
            row = result_row(
                name,
                time_best_s=timing.best_s,
                time_mean_s=timing.mean_s,
                time_std_s=timing.std_s,
                repeat=args.repeat,
                warmup=args.warmup,
                batch=args.batch,
                n=64,
                blocks=block_sizes,
                max_relative_offblock_entry=leakage_ratio,
            )
            if name.endswith("blocks"):
                row.update(
                max_relative_eigenvalue_difference=comparison.max_abs_difference,
                    speedup_vs_dense=comparison.speedup,
                    paired_delta_s=comparison.delta_mean_s,
                    paired_ci95_s=list(comparison.ci95_s),
                    verdict=comparison.verdict,
                )
            results.append(row)

    payload = make_payload(
        benchmark="gpu_block_eigh",
        config={
            "batch": args.batch,
            "n_steps": args.n_steps,
            "block_sizes": block_sizes,
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
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_block_eigh"),
    )


if __name__ == "__main__":
    main()
