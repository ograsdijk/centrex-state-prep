"""Timestep tiles preserve physical states, tracking, history and stream order."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from state_prep_gpu.simulator import simulate_batched_scalar_microwaves_shared_slow as simulate
from state_prep_gpu.slow_hamiltonian import BatchedSlowHamiltonianTerms

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TILE_SINGLE_PRECISION_TOL = 4e-5


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
@pytest.mark.parametrize('block_eigh', [False, True])
@pytest.mark.parametrize('different_envelope', [False, True])
def test_velocity_groups_match_independent_solves(device, dtype, block_eigh, different_envelope, monkeypatch):
    opts = options(device, dtype, dense=False)
    opts['block_eigh'] = block_eigh
    for key in ('coupling_scales', 'D_mu_diag_batch'):
        opts[key][2] = opts[key][0]
    if different_envelope:
        opts['amp_t'][70:, 2, 0] = 0.3
    speeds = torch.tensor([150., 184., 230.], dtype=torch.float64, device=device)
    reference = simulate(**dict(opts, dt=opts['dt'] * (184. / speeds)[None, :].expand(136, 3)))
    original = torch.linalg.eigh
    batches = []
    def record(matrix, *args, **kwargs):
        if matrix.ndim == 3:
            batches.append(matrix.shape[0])
        return original(matrix, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, 'eigh', record)
    actual = simulate(**opts, forward_velocities=speeds, reference_forward_velocity=184.,
                      slow_eigh_chunk_size=1)
    assert_outputs(actual, reference, TILE_SINGLE_PRECISION_TOL if dtype == torch.complex64 else 1e-11)
    if not block_eigh:
        assert batches == [3 if different_envelope else 2] * 136


@pytest.mark.parametrize('velocities,reference', [([0., 184., 200.], 184.),
    ([float('nan')], 184.), ([[184.]], 184.), ([184., 200.], 184.),
    ([184.], None), ([184.], 0.), ([184.], float('inf'))])
def test_gpu_velocity_validation(velocities, reference):
    with pytest.raises(ValueError, match='velocity|velocities'):
        simulate(**options('cpu'), forward_velocities=velocities, reference_forward_velocity=reference)


@pytest.mark.parametrize('device', DEVICES)
def test_velocity_group_keys_preserve_shift_precision(device, monkeypatch):
    opts = options(device, torch.complex64)
    opts['block_eigh'] = False
    opts['coupling_scales'][:] = 1
    opts['D_mu_diag_batch'] = torch.ones((3, 8), dtype=torch.float64, device=device)
    opts['D_mu_diag_batch'][2, 0] += 1e-9  # Distinct even if a float32 key would round it away.
    original = torch.linalg.eigh
    batches = []
    def record(matrix, *args, **kwargs):
        if matrix.ndim == 3:
            batches.append(matrix.shape[0])
        return original(matrix, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, 'eigh', record)
    simulate(**opts, forward_velocities=[150., 184., 230.], reference_forward_velocity=184.,
             slow_eigh_chunk_size=1)
    assert batches == [2] * 136


def options(device, dtype=torch.complex128, *, dense=True, steps=137):
    rng = np.random.default_rng(14)
    n, batch = 8, 3
    def tensor(a):
        return torch.as_tensor(a, dtype=dtype, device=device)
    zero = np.zeros((n,n),complex)
    hff = np.diag(np.arange(n)*0.13).astype(complex)
    mu = zero.copy()
    for group in ([0,3,5],[1,4,6]):
        a = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
        hff[np.ix_(group,group)] += 0.02*(a+a.conj().T)
        mu[np.ix_(group,group)] = 0.01*(a+a.conj().T)
    hsz = np.diag([1.,-1.,0.7,1.,-1.,1.,-1.,-0.7])
    hsx = zero.copy()
    hsx[0,1],hsx[1,0] = 0.03j,-0.03j
    hsx[1,2] = hsx[2,1] = 0.02
    hsx[6,7],hsx[7,6] = 0.025j,-0.025j
    e = np.zeros((steps,3))
    e[:,2] = np.linspace(-0.5,0.5,steps)
    if dense:
        # Absent at initialization and in the first tile, then changes direction.
        e[129:-1,0] = np.linspace(0.02,0.06,max(0,steps-130))
    e[-1] = [9.,8.,7.]  # Terminal sample is not propagated.
    psi = rng.normal(size=(batch,2,n)) + 1j*rng.normal(size=(batch,2,n))
    psi /= np.linalg.norm(psi,axis=-1,keepdims=True)
    terms = BatchedSlowHamiltonianTerms(*map(tensor,(hff,hsx,zero,hsz,zero,zero,zero)))
    if not dense:
        # Keep the terminal sample from conservatively connecting the sectors.
        e[-1,0:2] = 0
    return dict(terms=terms,E_t=torch.tensor(e,device=device),B_t=torch.zeros((steps,3),device=device),
        E0=torch.tensor([0.,0.,-0.75],device=device),B0=torch.zeros(3,device=device),
        psi0=tensor(psi),dt=0.002,
        H_mu_base_fields=tensor(np.stack([mu,0.3*mu])),
        amp_t=torch.tensor(np.ones((steps-1,batch,2)),device=device),
        coupling_scales=torch.tensor([[1.,0.6],[0.7,1.2],[1.3,0.8]],device=device),
        D_mu_diag_batch=torch.tensor(np.arange(batch)[:,None]*np.linspace(0.,0.01,n)[None,:],device=device),
        monitor_state_vectors=tensor(np.eye(n)[[0,4]]),store_every=17,block_eigh=True)


def assert_outputs(a,b,tolerance):
    for name in ("psi_final","probabilities_final","monitor_probabilities_final","last_evecs","psi_history"):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),atol=tolerance,rtol=tolerance)
    assert torch.equal(a.history_indices,b.history_indices)


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("dtype",[torch.complex64,torch.complex128])
@pytest.mark.parametrize("dense",[False,True])
@pytest.mark.parametrize("per_point_dt",[False,True])
def test_tiles_match_per_step_through_crossings_and_late_transverse_field(device,dtype,dense,per_point_dt):
    opts = options(device,dtype,dense=dense)
    from state_prep_gpu._blocks import shared_slow_blocks
    blocks = shared_slow_blocks(opts['terms'],opts['E_t'],opts['B_t'],opts['E0'],opts['B0'],opts['H_mu_base_fields'])
    assert (blocks is None) == dense
    dt = torch.linspace(0.001,0.003,136,device=device)
    opts["dt"] = dt[:,None]*torch.tensor([0.8,1.,1.2],device=device) if per_point_dt else dt
    reference = simulate(**opts,slow_eigh_chunk_size=1)
    actual = simulate(**opts,slow_eigh_chunk_size=128)
    tolerance = TILE_SINGLE_PRECISION_TOL if dtype == torch.complex64 else 1e-11
    assert_outputs(actual,reference,tolerance)
    torch.testing.assert_close(actual.psi_history[0],opts["psi0"],atol=0,rtol=0)
    assert actual.history_indices[-1].item() == 136


@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("chunk_size",[None,128])
def test_tiled_builder_is_bounded_and_excludes_terminal_sample(device,chunk_size,monkeypatch):
    import state_prep_gpu.simulator as module
    opts = options(device,steps=260)
    original = module.slow_hamiltonian_batch
    sampled = []
    def record(terms,e,b,**kw):
        if e.ndim == 2:
            sampled.append(e.clone())
        return original(terms,e,b,**kw)
    monkeypatch.setattr(module,"slow_hamiltonian_batch",record)
    simulate(**opts,slow_eigh_chunk_size=chunk_size)
    if chunk_size == 128 or device == 'cuda':
        assert [len(e) for e in sampled] == [128,128,3]
        torch.testing.assert_close(torch.cat(sampled),opts["E_t"][:-1],atol=0,rtol=0)
    else:
        assert sampled == []  # CPU tensor default retains per-step solves.


@pytest.mark.parametrize("value",[0,-1,True,False,1.5,"128"])
def test_invalid_tile_size(value):
    with pytest.raises(ValueError,match="slow_eigh_chunk_size"):
        simulate(**options("cpu",steps=3),slow_eigh_chunk_size=value)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_tiled_evolution_uses_current_cuda_stream_and_weighted_profile():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        opts = options("cuda",steps=137)
        reference = simulate(**opts,slow_eigh_chunk_size=1)
        timings = {}
        actual = simulate(**opts,slow_eigh_chunk_size=128,profile_timings=timings,profile_step_stride=7)
        completed = torch.cuda.Event()
        completed.record()
    torch.cuda.current_stream().wait_event(completed)
    assert_outputs(actual,reference,1e-11)
    assert len(timings["slow_eigh"]) == 136
    assert len(timings["state_update"]) == 20
    assert all(np.isfinite(values).all() and min(values) >= 0 for values in timings.values())
