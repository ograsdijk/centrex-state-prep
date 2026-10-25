"""Physical-state and ordering checks for benchmark-only velocity experiments."""
import numpy as np
import pytest

from bench_dense_investigation import patched
from bench_velocity_scan import make_simulator
from velocity_prototypes import cpu_variant, gpu_variant


@pytest.mark.parametrize('blocks', [False, True])
@pytest.mark.parametrize('partitioned', [False, True])
def test_cpu_prototypes_preserve_ragged_groups_and_output_order(spa2_setup, blocks, partitioned):
    from state_prep import Simulator
    sim = make_simulator(spa2_setup, 184.)
    det = np.repeat(np.array([0., 1e5, 0., 2e5, 1e5])[:, None], 2, axis=1)
    opts = dict(detunings_hz=det, intensity_prefactors=np.ones_like(det),
                forward_velocities=[150., 180., 210., 240., 170.],
                N_steps=24, progress=False, block_eigh=blocks,
                monitor_states=spa2_setup['monitor_states'])
    reference = sim.run_microwave_scan(**opts)
    variant = cpu_variant(grouped=not partitioned, partitioned=partitioned)
    name = '_run_microwave_scan_parallel_loky' if partitioned else '_time_evolve_mu_batched_shared_slow'
    with patched(Simulator, name, getattr(variant, name)):
        actual = sim.run_microwave_scan(**opts, workers=2 if partitioned else 1)
    for field in ('psis_final', 'probabilities_final', 'monitor_probabilities_final', 'times_per_point'):
        np.testing.assert_allclose(getattr(actual, field), getattr(reference, field), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('variant', ['grouped', 'full4', 'full8', 'combined8'])
def test_gpu_prototypes_preserve_histories_monitors_and_tile_boundaries(device, variant):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_gpu_slow_tiles import options, assert_outputs
    from state_prep_gpu.simulator import simulate_batched_scalar_microwaves_shared_slow
    opts = options(device, dense=True)
    opts['block_eigh'] = False
    for key in ('coupling_scales', 'D_mu_diag_batch'):
        opts[key] = opts[key][:2].repeat(2, 1)
    opts['amp_t'] = opts['amp_t'][:, :2].repeat(1, 2, 1)
    opts['psi0'] = torch.cat((opts['psi0'], opts['psi0'][:1]))
    opts.update(dt=torch.linspace(0.001, 0.003, 136, device=device, dtype=torch.float64),
                forward_velocities=[150., 180., 210., 240.], reference_forward_velocity=184.,
                slow_eigh_chunk_size=128)
    reference = simulate_batched_scalar_microwaves_shared_slow(**opts)
    function = gpu_variant(grouped=variant in ('grouped', 'combined8'),
                           full_chunk=4 if variant == 'full4' else (8 if variant in ('full8', 'combined8') else 0))
    actual = function(**opts)
    assert_outputs(actual, reference, 1e-11)
