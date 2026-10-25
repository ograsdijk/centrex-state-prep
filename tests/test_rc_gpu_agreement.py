"""Real RC J12 fields retain fixed carrier detuning and per-velocity beats."""
import numpy as np
import pytest


@pytest.fixture(scope='module',params=[0.,-7.4])
def rc_setup(request):
    from bench_convergence import build_multitone_setup
    return build_multitone_setup(bg_fraction=1/35,rc_bg_fraction=1/35,rc_offset_mhz=request.param)


@pytest.mark.parametrize('velocities',[False,True])
def test_real_rc_cpu_gpu_agreement(rc_setup,velocities):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from bench_convergence import make_grid
    from bench_gpu_spa2 import make_torch_runner,run_cpu
    det,power,_ = make_grid([-1.,0.,0.5],[1.,4.],3,True)
    speeds = np.array([150.,184.,230.,150.,184.,230.]) if velocities else None
    cpu = run_cpu(rc_setup,n_steps=160,detunings=det,intensity_prefactors=power,forward_velocities=speeds)
    runner,sync = make_torch_runner(rc_setup,n_steps=160,detunings=det,
        intensity_prefactors=power,forward_velocities=speeds,max_eigh_matrices=16)
    gpu = runner(store_every=37)
    sync()
    vectors = gpu.psi_final.cpu().numpy()
    np.testing.assert_allclose(np.abs(vectors@cpu.V_fin.conj())**2,
                               np.abs(cpu.psis_final@cpu.V_fin.conj())**2,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(np.abs(np.sum(cpu.psis_final.conj()*vectors,axis=-1))**2,1,atol=1e-5)
    np.testing.assert_allclose(gpu.monitor_probabilities_final.cpu(),cpu.monitor_probabilities_final,atol=1e-5,rtol=1e-5)
    assert gpu.history_indices[-1].item() == 159
