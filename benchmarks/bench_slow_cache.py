"""Benchmark-only cross-call slow-eigensystem reuse and GPU time batching.

Alternate two independent power scans over identical fields/timesteps. The
public scan already shares each slow solve across points; this tests separate
calls. GPU prebatch_each_call includes preparation in every timed invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path
import time

import numpy as np
from scipy.linalg.lapack import zheevd

from bench_dense_investigation import patched, parameters
from bench_gpu_spa2 import make_torch_runner, run_cpu
from common import make_payload, write_results
from paired import compare_paired, format_report
from profile_transverse_fields import field_variant


def digest_arrays(arrays, extra):
    digest = hashlib.sha256(extra.encode())
    for a in arrays:
        a = np.asarray(a)
        digest.update(str((a.dtype.str, a.shape)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def gpu_inputs(run):
    # This is a benchmark factory's closure, not a production cache API.
    closure = inspect.getclosurevars(run).nonlocals
    return closure['terms'], closure['inputs'], closure['dt']


def gpu_key(run, perturb=False):
    terms, inputs, dt = gpu_inputs(run)
    arrays = [getattr(terms, name).cpu().numpy() for name in ('Hff','HSx','HSy','HSz','HZx','HZy','HZz')]
    arrays += [inputs[name].cpu().numpy() for name in ('E_t','B_t','E0','B0')]
    if perturb:
        arrays[-4] = arrays[-4].copy()
        arrays[-4][0,0] += 1e-6
    return digest_arrays(arrays, str((dt, terms.Hff.device, terms.Hff.dtype)))


def prepare_gpu(run):
    import torch
    from state_prep_gpu.slow_hamiltonian import slow_hamiltonian_batch
    terms, inputs, _ = gpu_inputs(run)
    torch.cuda.synchronize()
    started = time.perf_counter()
    h0 = slow_hamiltonian_batch(terms, inputs['E0'], inputs['B0'])
    initial = torch.linalg.eigh(h0)
    # Exact independent eigensystems at all propagation times, before tracking.
    # Tracking remains sequential in the unmodified propagation loop.
    matrices = slow_hamiltonian_batch(terms, inputs['E_t'][:-1], inputs['B_t'][:-1])
    w, v = torch.linalg.eigh(matrices)
    entries = [initial, *zip(w.unbind(), v.unbind())]
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    retained = sum(a.numel()*a.element_size() for pair in entries for a in pair)
    return entries, elapsed, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','gpu'], required=True)
    parser.add_argument('--field-case', default='transverse_b', choices=['transverse_b','varying_e'])
    parser.add_argument('--shapes', nargs='+', required=True, help='steps:batch pairs')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--modes', nargs='+')
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('chunk-size must be positive')
    if args.output.exists():
        parser.error('choose a fresh output file')
    setup = field_variant(args.field_case)
    _, basis = np.linalg.eigh(setup['hamiltonian'].get_H_t_func()(setup['trajectory'].get_T()))
    rows = []
    for shape in args.shapes:
        steps,batch = map(int,shape.split(':'))
        det,power = parameters(batch)
        counts = {}
        print(f'{args.device}: steps={steps}, batch={batch}', flush=True)
        if args.device == 'cpu':
            import state_prep.simulator as module
            from state_prep import limit_blas_threads
            times = np.linspace(0, setup['trajectory'].get_T(), steps)
            mids = 0.5*(times[:-1]+times[1:])
            start = time.perf_counter()
            hams = [setup['hamiltonian'].get_H_t_func()(t) for t in mids]
            key = digest_arrays(hams, str(times))
            with limit_blas_threads(1):
                cached = [zheevd(h) for h in hams]
            if any(info for _,_,info in cached):
                raise RuntimeError('Slow LAPACK cache preparation failed')
            build_s = time.perf_counter()-start
            retained = sum(w.nbytes+v.nbytes for w,v,_ in cached)
            assert key != digest_arrays(hams, str(times*1.001))
            changed = hams[0].copy(); changed[0,0] += 1
            assert key != digest_arrays([changed,*hams[1:]],str(times))
            def call(mode, factor):
                count = 0
                def solve(matrix):
                    nonlocal count
                    step,point = divmod(count,batch+1)
                    count += 1
                    return cached[step] if point == 0 else zheevd(matrix)
                with patched(module,'zheevd',solve if mode == 'cached' else zheevd):
                    result = run_cpu(setup,n_steps=steps,detunings=det,intensity_prefactors=power*factor)
                if mode == 'cached' and count != (steps-1)*(batch+1):
                    raise RuntimeError('Dense call pattern changed')
                return result.psis_final
            modes = ['baseline','cached']
            sync = None
        else:
            import torch
            torch.set_num_threads(1)
            runs = [make_torch_runner(setup,n_steps=steps,detunings=det,
                    intensity_prefactors=power*factor,block_eigh=False,slow_eigh_chunk_size=1)[0] for factor in (0.5,1.)]
            key = gpu_key(runs[0])
            assert key == gpu_key(runs[1])
            assert key != gpu_key(runs[0],perturb=True)
            cached, build_s, retained = prepare_gpu(runs[0])
            original = torch.linalg.eigh
            def call(mode, factor):
                run = runs[0 if factor == 0.5 else 1]
                entries = prepare_gpu(run)[0] if mode == 'prebatch_each_call' else cached
                cursor = 0
                chunk = None
                terms, inputs, _ = gpu_inputs(run)
                def solve(matrix,*a,**kw):
                    nonlocal cursor, chunk
                    if matrix.ndim != 2:
                        return original(matrix,*a,**kw)
                    if mode == 'prebatch_chunks':
                        if cursor == 0:
                            result = original(matrix,*a,**kw)
                        else:
                            from state_prep_gpu.slow_hamiltonian import slow_hamiltonian_batch
                            step = cursor-1
                            if step % args.chunk_size == 0:
                                end = min(step+args.chunk_size,steps-1)
                                h = slow_hamiltonian_batch(terms,inputs['E_t'][step:end],inputs['B_t'][step:end])
                                chunk = original(h)
                            offset = step % args.chunk_size
                            result = chunk[0][offset],chunk[1][offset]
                    else:
                        result = entries[cursor]
                    cursor += 1
                    return result
                with patched(torch.linalg,'eigh',solve if mode != 'baseline' else original):
                    result = run()
                if mode != 'baseline' and cursor != steps:
                    raise RuntimeError('Shared slow solve count changed')
                return result.psi_final.cpu().numpy()
            modes = ['baseline','prebatch_each_call','cached','prebatch_chunks']
            sync = torch.cuda.synchronize
        if args.modes:
            if any(mode not in modes for mode in args.modes) or 'baseline' not in args.modes:
                parser.error('modes must include baseline and supported device modes')
            modes = args.modes
        references = {}
        validation = dict(max_infidelity=0.,max_norm_error=0.)
        def runner(mode):
            counts[mode] = 0
            def run():
                factor = (0.5,1.)[counts[mode] % 2]
                counts[mode] += 1
                psi = call(mode, factor)
                norm_error = np.max(np.abs(np.sum(np.abs(psi)**2,axis=-1)-1))
                validation['max_norm_error'] = max(validation['max_norm_error'],float(norm_error))
                if norm_error > 1e-7:
                    raise RuntimeError(f'Invalid norm: {norm_error}')
                if factor not in references:
                    references[factor] = psi
                fidelity = np.abs(np.sum(references[factor].conj()*psi,axis=-1))**2
                validation['max_infidelity'] = max(validation['max_infidelity'],float(np.max(np.abs(1-fidelity))))
                if np.max(np.abs(1-fidelity)) > 1e-5:
                    raise RuntimeError('Physical cached/uncached states differ')
                return np.abs(psi @ basis.conj())**2
            return run
        paired = compare_paired({mode:runner(mode) for mode in modes}, warmup=args.warmup,
            repeat=args.repeat,sync=sync,progress=True,
            check=lambda a,b: float(np.max(np.abs(a-b))),tolerance=1e-5)
        print(format_report(paired),flush=True)
        if not all(c.agrees for c in paired.comparisons):
            raise RuntimeError('Cached probabilities disagree')
        rows.extend(dict(row,steps=steps,batch=batch) for row in paired.rows())
        rows.append(dict(case='state_validation',steps=steps,batch=batch,**validation))
        rows.append(dict(case='cache_build',steps=steps,batch=batch,build_s=build_s,
                         retained_bytes=retained,input_key=key,
                         one_call_bytes_at_10000_steps=10000*(64*64*16+64*8),
                         chunk_bytes=min(steps-1,args.chunk_size)*(64*64*16+64*8),
                         alternating_power_scans=[0.5,1.0],invalidation_checks_passed=True))
        write_results(make_payload(benchmark='slow_eigensystem_cache',config=vars(args),results=rows),
                      output=str(args.output),csv_path=None)


if __name__ == '__main__':
    main()
