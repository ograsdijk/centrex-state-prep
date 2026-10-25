"""Source-checked benchmark variants of the current production velocity scan.

These experiments deliberately do not change public APIs or production defaults.
"""
from pathlib import Path
import sys
import types


def replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Prototype source anchor changed: {old[:90]!r}')
    return source.replace(old, new, 1)


def clone(module, suffix, source):
    name = module.__package__ + '._velocity_' + suffix
    result = types.ModuleType(name)
    sys.modules[name] = result
    exec(compile(source, name, 'exec'), result.__dict__)
    result.__source__ = source
    return result


def cpu_variant(*, grouped=False, partitioned=False):
    import state_prep.simulator as module
    source = Path(module.__file__).read_text()
    if grouped:
        start = source.index('    def _time_evolve_mu_batched_shared_slow(')
        end = source.index('    def _time_evolve_mu_batched_shared_slow_multitone(', start)
        method = source[start:end]
        method = replace(method, '        # Calculate Hamiltonian at tini', '''        if propagator != "frozen":
            raise ValueError("Grouped prototype supports frozen propagation only")
        propagation_groups = [np.flatnonzero(solve_groups == group)
                              for group in np.unique(solve_groups)]
        # Calculate Hamiltonian at tini''')
        a = method.index('            solved = {}')
        b = method.index('                # Assemble rotating-frame Hamiltonian', a)
        method = method[:a] + '''            for members in propagation_groups:
                b = int(members[0])
''' + method[b:]
        a = method.index('                if time_scales is not None:\n                    solved[group]')
        b = method.index('            # Rotate back out', a)
        method = method[:a] + '''                states = psis_slow[members]
                tmp = states.reshape(-1, n) @ V_rot.conj()
                tmp = tmp.reshape(states.shape)
                tmp *= np.exp(-1j * D_rot[None, None, :] *
                              (dt * velocity_scales[members])[:, None, None])
                psis_slow[members] = (tmp.reshape(-1, n) @ V_rot.T).reshape(states.shape)

''' + method[b:]
        source = source[:start] + method + source[end:]
    if partitioned:
        start = source.index('    def _run_microwave_scan_parallel_loky(')
        end = source.index('    def _time_evolve_mu_batched_shared_slow(', start)
        method = source[start:end]
        method = replace(method, '        chunk_indices = np.array_split(np.arange(batch), n_jobs)', '''        _, groups = np.unique(np.concatenate((detunings_hz, intensity_prefactors), axis=1),
                              axis=0, return_inverse=True)
        n_jobs = min(n_jobs, int(groups.max()) + 1)
        bins = [[] for _ in range(n_jobs)]
        sizes = np.zeros(n_jobs, dtype=int)
        for group in np.unique(groups):
            members = np.flatnonzero(groups == group)
            worker = int(np.argmin(sizes))
            bins[worker].extend(members.tolist())
            sizes[worker] += len(members)
        chunk_indices = [np.asarray(indices) for indices in bins]
        restore_order = np.argsort(np.concatenate(chunk_indices))''')
        for name in ('probabilities_final', 'monitor_probabilities_final', 'psis_final'):
            old = f'np.concatenate(\n                [result.{name} for result in results], axis=0\n            )'
            if old in method:
                method = replace(method, old, old + '[restore_order]')
            else:
                old = f'np.concatenate([result.{name} for result in results], axis=0)'
                method = replace(method, old, old + '[restore_order]')
        source = source[:start] + method + source[end:]
    result = clone(module, f'cpu_{grouped}_{partitioned}', source)
    # Worker closures annotate/construct this container. Keep the importable
    # production class so loky does not need an in-memory prototype module.
    result.MicrowaveScanResult = module.MicrowaveScanResult
    return result.Simulator


def gpu_variant(*, grouped=False, full_chunk=0):
    import state_prep_gpu.simulator as module
    source = Path(module.__file__).read_text()
    start = source.index('def simulate_batched_scalar_microwaves_shared_slow(')
    end = source.index('\ndef _as_torch_complex(', start)
    method = source[start:end]
    method = replace(method, '    torch = get_torch()', '''    if profile_timings is not None:
        raise ValueError("Use the production route for stage profiling")
    torch = get_torch()''')
    if full_chunk:
        if 128 % full_chunk:
            raise ValueError('Full chunk must divide the 128-step slow tile')
        a = method.index('        # weights: (B,M) complex')
        b = method.index('        # Factor A = V @ V_rot', a)
        method = method[:a] + f'''        if blocks is not None or slow_eigh_chunk_size != 128:
            raise ValueError("Full-time prototype requires dense solves and slow tiles of 128")
        if offset % {full_chunk} == 0:
            if i:
                del D_rot, V_rot
            rot_values = rot_vectors = None
            tile_end = min(offset + {full_chunk}, len(slow_values))
            vc = slow_vectors[offset:tile_end]
            rotated = torch.matmul(vc.mH[:, None], torch.matmul(H_mu_base_fields[None], vc[:, None]))
            amplitude = amp_t[i:i + tile_end-offset]
            weights = coupling_scales[None] * (amplitude[:, None] if amp_t.ndim == 2 else amplitude)
            shifts = D_mu_diag_batch
            if solve_indices is not None:
                weights, shifts = weights[:, solve_indices], shifts[solve_indices]
            matrices = torch.einsum('tkm,tmij->tkij', weights.to(psi.dtype), rotated)
            matrices[:, :, diag_idx, diag_idx] += slow_values[offset:tile_end, None]
            matrices[:, :, diag_idx, diag_idx] += shifts[None]
            width, groups = matrices.shape[:2]
            rot_values, rot_vectors = torch.linalg.eigh(matrices.reshape(-1, n, n))
            rot_values = rot_values.reshape(width, groups, n)
            rot_vectors = rot_vectors.reshape(width, groups, n, n)
            del matrices, rotated, weights, amplitude, vc
        D_rot, V_rot = rot_values[offset % {full_chunk}], rot_vectors[offset % {full_chunk}]
        if solve_inverse is not None:
            D_rot, V_rot = D_rot[solve_inverse], V_rot[solve_inverse]

''' + method[b:]
    if grouped:
        method = replace(method, '    # Initial slow Hamiltonian and reference eigenvectors', '''    if solve_inverse is not None:
        counts = torch.bincount(solve_inverse)
        if not bool((counts == counts[0]).all()):
            raise ValueError("Grouped GPU prototype requires equal group multiplicities")
        group_order = torch.argsort(solve_inverse, stable=True)
        restore_order = torch.argsort(group_order)
        group_count = len(solve_indices)
        multiplicity = batch // group_count
    # Initial slow Hamiltonian and reference eigenvectors''')
        method = replace(method, '''        if solve_inverse is not None:
            D_rot, V_rot = D_rot[solve_inverse], V_rot[solve_inverse]
''', '')
        a = method.index('        phases_rot = torch.exp(')
        b = method.index('        psi = torch.matmul(psi_slow, V.T)', a)
        original = method[a:b]
        method = method[:a] + '''        if solve_inverse is not None:
            state_count = psi.shape[1]
            tmp = torch.matmul(psi_slow[group_order].reshape(group_count, -1, n), V_rot.conj())
            step_dt = _dt_for_step(dt_arr, i)
            if step_dt.ndim == 0:
                step_dt = step_dt.expand(batch, 1)
            phases = torch.exp(-1j * D_rot[:, None, None, :] *
                               step_dt[group_order].reshape(group_count, multiplicity, 1, 1))
            tmp = (tmp.reshape(group_count, multiplicity, state_count, n) * phases).reshape(group_count, -1, n)
            psi_slow = torch.matmul(tmp, V_rot.mT).reshape(batch, state_count, n)[restore_order]
        else:
''' + ''.join('    ' + line + '\n' for line in original.splitlines()) + method[b:]
    source = source[:start] + method + source[end:]
    return clone(module, f'gpu_{grouped}_{full_chunk}', source).simulate_batched_scalar_microwaves_shared_slow
