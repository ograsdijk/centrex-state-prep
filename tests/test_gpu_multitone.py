"""Beat-correct tiled propagation versus direct independent CPU matrices."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from state_prep_gpu import simulate_batched_multitone_shared_slow as simulate
from state_prep_gpu.slow_hamiltonian import BatchedSlowHamiltonianTerms

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
SINGLE_PRECISION_NORM_TOL = 2e-4  # CUDA float32 eigensystem/update roundoff over 137 steps.


def options(device,dtype=torch.complex128):
    rng = np.random.default_rng(902)
    n,batch,M,T = 5,3,2,138
    def tensor(a,kind=dtype):
        return torch.as_tensor(a,dtype=kind,device=device)
    hff = np.diag(np.arange(n)*0.7).astype(complex)
    hsx = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    hsx = 0.02*(hsx+hsx.conj().T)
    hsz = np.diag(np.linspace(-0.1,0.1,n)).astype(complex)
    zero = np.zeros_like(hff)
    terms = BatchedSlowHamiltonianTerms(*map(tensor,(hff,hsx,zero,hsz,zero,zero,zero)))
    upper = 0.03*np.triu(rng.normal(size=(M,n,n))+1j*rng.normal(size=(M,n,n)),1)
    psi = rng.normal(size=(batch,2,n))+1j*rng.normal(size=(batch,2,n))
    psi /= np.linalg.norm(psi,axis=-1,keepdims=True)
    dt = np.linspace(0.003,0.005,T-1)
    grid = np.r_[0.,np.cumsum(dt)]
    e = np.zeros((T,3))
    e[:,2] = np.linspace(-0.2,0.3,T)
    e[129:-1,0] = 0.25  # Direction changes after first slow tile.
    e[-1] = [8.,9.,10.]  # Must not be propagated.
    amp = np.ones((T-1,batch,M))
    amp[31:,1,1] = 0.4
    return dict(terms=terms,E_t=tensor(e,torch.float64),B_t=tensor(np.zeros_like(e),torch.float64),
        E0=tensor([0.,0.,-0.4],torch.float64),B0=tensor(np.zeros(3),torch.float64),
        psi0=tensor(psi),dt=tensor(dt,torch.float64),H_mu_upper_fields=tensor(upper),amp_t=tensor(amp),
        coupling_scales=tensor([[1.,0.5],[0.8,1.2],[1.4,0.7]],torch.float64),
        D_mu_diag_batch=tensor(np.arange(batch)[:,None]*np.linspace(0.,0.03,n)[None,:]),
        sample_times=tensor((grid[:-1]+grid[1:])/2,torch.float64),
        beat_angular_frequencies=tensor([[0.,7.],[0.,11.],[0.,-9.]],torch.float64),
        forward_velocities=[150.,184.,230.],reference_forward_velocity=184.,
        monitor_state_vectors=tensor(np.eye(n)[[0,3]]),store_every=19)


def cpu_reference(opts):
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    terms = opts['terms']
    hff,hsx,hsz = map(array,(terms.Hff,terms.HSx,terms.HSz))
    e,psi,upper,amp,shifts,scales,times,freq,dt = map(array,(
        opts['E_t'],opts['psi0'],opts['H_mu_upper_fields'],opts['amp_t'],opts['D_mu_diag_batch'],
        opts['coupling_scales'],opts['sample_times'],opts['beat_angular_frequencies'],opts['dt']))
    psi = psi.astype(complex).copy()
    traversal = 184./np.asarray(opts['forward_velocities'])
    history = [psi.copy()]
    for i in range(len(e)-1):
        # Slow term matrices are in Hz; propagation uses angular frequency.
        d,v = np.linalg.eigh(2*np.pi*(hff+e[i,0]*hsx+e[i,2]*hsz))
        for b in range(len(psi)):
            hu = sum(scales[b,m]*amp[i,b,m]*np.exp(-1j*freq[b,m]*times[i]*traversal[b])*upper[m]
                     for m in range(len(upper)))
            h = v.conj().T @ (hu+hu.conj().T) @ v + np.diag(d+shifts[b])
            values,vectors = np.linalg.eigh(h)
            basis = v @ vectors
            psi[b] = ((psi[b] @ basis.conj())*np.exp(-1j*values*dt[i]*traversal[b])) @ basis.T
        history.append(psi.copy())
    return np.asarray(history)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
@pytest.mark.parametrize('time_chunk,scan_cap',[(1,256),(8,256),(8,9)])
def test_beats_histories_velocities_and_bounded_scan_tiles(device,dtype,time_chunk,scan_cap,monkeypatch):
    opts = options(device,dtype)
    expected = cpu_reference(opts)
    original = torch.linalg.eigh
    calls = []
    def record(matrix,*a,**kw):
        calls.append(int(np.prod(matrix.shape[:-2])))
        return original(matrix,*a,**kw)
    monkeypatch.setattr(torch.linalg,'eigh',record)
    actual = simulate(**opts,rotating_eigh_chunk_size=time_chunk,max_eigh_matrices=scan_cap)
    tolerance = 5e-5 if dtype == torch.complex64 else 1e-10
    np.testing.assert_allclose(actual.psi_final.cpu(),expected[-1],atol=tolerance,rtol=tolerance)
    np.testing.assert_allclose(actual.psi_history.cpu(),expected[actual.history_indices.cpu()],atol=tolerance,rtol=tolerance)
    assert max(calls) <= scan_cap
    norm_tolerance = SINGLE_PRECISION_NORM_TOL if dtype == torch.complex64 else tolerance
    np.testing.assert_allclose(actual.probabilities_final.sum(-1).cpu(),1,atol=norm_tolerance)
    torch.testing.assert_close(actual.monitor_probabilities_final,actual.probabilities_final[:,:,[0,3]],atol=tolerance,rtol=tolerance)


@pytest.mark.parametrize('parameter,value', [('rotating_eigh_chunk_size',0),('slow_eigh_chunk_size',True),
    ('max_eigh_matrices',1.5),('profile_step_stride',0),('dt',-1),
    ('beat_angular_frequencies',[[0.,float('nan')]]*3),('sample_times',[0.]),
    ('forward_velocities',[0.,184.,230.])])
def test_bad_inputs(parameter,value):
    opts = options('cpu')
    opts[parameter] = value
    with pytest.raises(ValueError):
        simulate(**opts)


@pytest.mark.parametrize('device',DEVICES)
def test_explicit_per_point_times_match_velocity_scaling(device):
    opts = options(device)
    reference = simulate(**opts)
    traversal = 184./torch.tensor(opts.pop('forward_velocities'),dtype=torch.float64,device=device)
    opts.pop('reference_forward_velocity')
    opts['dt'] = opts['dt'][:,None]*traversal[None]
    opts['sample_times'] = opts['sample_times'][:,None]*traversal[None]
    actual = simulate(**opts,rotating_eigh_chunk_size=3,max_eigh_matrices=12)
    torch.testing.assert_close(actual.psi_final,reference.psi_final,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(actual.monitor_probabilities_final,reference.monitor_probabilities_final,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('device',DEVICES)
def test_zero_beats_match_existing_scalar_path(device):
    from state_prep_gpu.simulator import simulate_batched_scalar_microwaves_shared_slow
    opts = options(device)
    opts['beat_angular_frequencies'].zero_()
    actual = simulate(**opts,return_final_probabilities=False)
    upper = opts.pop('H_mu_upper_fields')
    opts.pop('beat_angular_frequencies')
    opts.pop('sample_times')
    reference = simulate_batched_scalar_microwaves_shared_slow(**opts,
        H_mu_base_fields=upper+upper.mH,block_eigh=False,slow_eigh_chunk_size=128)
    assert actual.probabilities_final is None
    torch.testing.assert_close(actual.psi_final,reference.psi_final,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(actual.psi_history,reference.psi_history,atol=1e-10,rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_nondefault_stream_and_profile_tile_weighting():
    opts = options('cuda')
    reference = simulate(**opts)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    profile = {}
    with torch.cuda.stream(stream):
        actual = simulate(**opts,profile_timings=profile,max_eigh_matrices=9)
    torch.testing.assert_close(actual.psi_final,reference.psi_final,atol=1e-11,rtol=1e-11)
    assert len(profile['slow_eigh_and_microwave_rotate']) == 137
    assert len(profile['microwave_build_and_eigh']) == 137
    assert len(profile['tracking_and_state_update']) == 6
    assert all(value >= 0 for times in profile.values() for value in times)
