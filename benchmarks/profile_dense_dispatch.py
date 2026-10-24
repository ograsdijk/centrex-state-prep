"""Diagnostic CUDA trace: device kernels versus host dispatch/check overhead."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from dense_solver_prototypes import ReusableCusolver


def trace_kernels(path):
    # GPU user annotations also have DeviceType.CUDA in profiler.events();
    # counting them would count parent regions and their kernels twice.
    events = [event for event in json.loads(path.read_text())['traceEvents'] if event.get('cat') == 'kernel']
    kernels = defaultdict(lambda: dict(count=0,total_us=0.))
    for event in events:
        kernels[event['name']]['count'] += 1
        kernels[event['name']]['total_us'] += event['dur']
    intervals = sorted((e['ts'],e['ts']+e['dur']) for e in events)
    union = []
    for start,end in intervals:
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1],end)
        else:
            union.append([start,end])
    return dict(kernel_count=len(events),summed_kernel_s=sum(e['dur'] for e in events)/1e6,
                kernel_union_s=sum(end-start for start,end in union)/1e6,
                kernels=dict(sorted(kernels.items(),key=lambda kv:kv[1]['total_us'],reverse=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrices', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('choose a fresh output directory')
    args.output_dir.mkdir(parents=True)
    import torch
    torch.set_num_threads(1)
    corpus = np.load(args.matrices)['rotating']
    rows = []
    for batch in (1,25,250):
        a = torch.tensor(corpus[:batch],device='cuda')
        native = ReusableCusolver(True)
        for name, solve in (('torch',torch.linalg.eigh),('native_deferred',native)):
            solve(a); native.flush(); torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                started = time.perf_counter()
                for _ in range(3):
                    with torch.profiler.record_function(name):
                        solve(a)
                native.flush()
                torch.cuda.synchronize()
                wall = time.perf_counter()-started
            prefix = args.output_dir / f'{name}_b{batch}'
            profile.export_chrome_trace(str(prefix.with_suffix('.json')))
            kernels = trace_kernels(prefix.with_suffix('.json'))
            if not kernels['kernel_count']:
                raise RuntimeError('CUDA profiler collected no kernels; do not interpret CPU-only trace')
            row = dict(backend=name,batch=batch,solves=3,profile_wall_s=wall,
                       **kernels)
            rows.append(row)
            prefix.with_suffix('.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=25))
            print({k:v for k,v in row.items() if k != 'kernels'},flush=True)
        native.close()
    (args.output_dir/'summary.json').write_text(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
