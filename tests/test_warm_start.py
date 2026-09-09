"""
Acceptance test for the warm start (chain resume) feature.

The contract, from `docs/warm-start-plan.md`:

    Run 10 iterations straight. Separately run 5, resume for 5 more. Assert
    all six outputs are bit-identical.

This test is only meaningful because `gibbs_sample()` seeds the global RNG once
per chain and draws from that single stream in a fixed serial order. Without
that, the two runs would differ regardless and a resume bug would be
indistinguishable from RNG noise.

Run with::

    pytest tests/test_warm_start.py -v
"""

import numpy as np
import pytest
import scipy.special

import hydra_pspec as hp
from hydra_pspec import utils


NITER = 10
SPLIT = 5
SEED = 10

# Names of the six sampler outputs, in the order gibbs_sample() returns them.
OUTPUTS = ("signal_amps", "signal_ps", "fg_amps", "sys_amps", "chisq", "ln_post")


def make_problem(Ntimes=4, Nfreqs=8, Nfgmodes=3, seed=11):
    """
    Build a small, fully deterministic sampler problem.

    Same structure as `sys_sampler_wrapper.py` (multiplicative gain
    systematics on foregrounds + EoR, plus noise) but sized so that a 10
    iteration chain runs in seconds.

    Returns
    -------
    dict
        Keyword arguments for `hp.pspec.gibbs_sample`, minus `Niter`,
        `out_dir` and `resume`.
    """
    rng = np.random.RandomState(seed)

    freqs = np.linspace(100e6, 120e6, Nfreqs)
    lsts = np.linspace(0.0, 1.0, Ntimes)

    nm_list = [(1, 0), (2, 0)]
    sys_amps_true = np.array([1.0 + 0.5j, 0.5 + 1.0j])
    sys_modes = hp.sys_solver.sys_modes(
        freqs_Hz=freqs,
        times_sec=lsts * 24.0 / (2.0 * np.pi) * 3600.0,
        modes=nm_list,
    )
    sys_prior = 100.0**2 * np.eye(sys_amps_true.size)
    gain_true = 1.0 + (sys_modes @ sys_amps_true).reshape([Nfreqs, Ntimes]).T

    fourier_op = hp.utils.fourier_operator(Nfreqs, unitary=True)

    ps_true = 0.0012 * (1.0 + 0.3 * np.sin(3.0 * np.linspace(0.0, 1.0, Nfreqs)))
    S_true = hp.pspec.covariance_from_pspec(ps_true, fourier_op)
    sqrt_S_true = np.linalg.cholesky(S_true)
    eor_true = (
        sqrt_S_true
        @ (rng.randn(Nfreqs, Ntimes) + 1.0j * rng.randn(Nfreqs, Ntimes))
        / np.sqrt(2.0)
    ).T

    fgmodes = np.array(
        [
            scipy.special.legendre(i)(np.linspace(-1.0, 1.0, Nfreqs))
            for i in range(Nfgmodes)
        ]
    ).T
    fg_true = (fgmodes @ rng.randn(Nfgmodes, Ntimes)).T

    ps_prior = np.column_stack(
        (1e-7 * np.ones(Nfreqs), 1e-1 * np.ones(Nfreqs))
    )

    noise_ps_true = 0.0004 * np.ones(Nfreqs)
    N_true = hp.pspec.covariance_from_pspec(noise_ps_true, fourier_op)
    Ninv = np.diag(1.0 / np.diag(N_true))
    n = (
        np.sqrt(N_true)
        @ (rng.randn(Nfreqs, Ntimes) + 1.0j * rng.randn(Nfreqs, Ntimes))
        / np.sqrt(2.0)
    )

    d = gain_true * (fg_true + eor_true) + n.T

    return dict(
        vis=d,
        flags=np.ones((Nfreqs,), dtype=int),
        Ninv=Ninv,
        freqs=freqs,
        lsts=lsts,
        signal_ps_initial=ps_true,
        signal_ps_prior=ps_prior.T,
        fg_modes=fgmodes,
        sys_modes=sys_modes,
        sys_prior=sys_prior,
        sys_initial=sys_amps_true,
        sky_model_initial=fg_true + eor_true,
        seed=SEED,
        verbose=False,
        nproc=1,
        solver_tol=1e-13,
        sample_systematics=True,
        sample_eor_fg=True,
        sample_signal_ps=True,
    )


def read_chain(out_dir):
    """Read every sample of every dataset from `out_dir/gibbs_samples.h5`."""
    n = utils.h5_chain_length(out_dir)
    return {
        name: np.stack(
            [utils.read_gibbs_sample_h5(out_dir, index=i)[name] for i in range(n)]
        )
        for name in OUTPUTS
    }


@pytest.fixture(scope="module")
def problem():
    return make_problem()


@pytest.fixture(scope="module")
def chains(problem, tmp_path_factory):
    """
    Run the reference chain and the split chain once, and return both.

    Module-scoped so the sampler runs three times in total, not three times
    per test.
    """
    straight_dir = tmp_path_factory.mktemp("straight")
    split_dir = tmp_path_factory.mktemp("split")

    straight = hp.pspec.gibbs_sample(
        Niter=NITER, out_dir=str(straight_dir), write_Niter=NITER, **problem
    )

    first = hp.pspec.gibbs_sample(
        Niter=SPLIT, out_dir=str(split_dir), write_Niter=SPLIT, **problem
    )
    second = hp.pspec.gibbs_sample(
        Niter=NITER - SPLIT,
        out_dir=str(split_dir),
        write_Niter=NITER - SPLIT,
        resume=True,
        **problem,
    )

    return dict(
        straight=straight,
        straight_dir=straight_dir,
        first=first,
        second=second,
        split_dir=split_dir,
    )


# ---------------------------------------------------------------------------
# The acceptance test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", OUTPUTS)
def test_resumed_chain_is_bit_identical(chains, name):
    """10 iterations straight == 5 iterations + a 5 iteration resume."""
    straight = read_chain(chains["straight_dir"])[name]
    split = read_chain(chains["split_dir"])[name]

    assert split.shape == straight.shape
    np.testing.assert_array_equal(
        split,
        straight,
        err_msg=f"resumed chain differs from uninterrupted chain in {name}",
    )


@pytest.mark.parametrize("name", OUTPUTS)
def test_npy_export_matches_h5(chains, name):
    """The .npy export of a resumed chain holds the full chain, not the
    last segment (blocker B5)."""
    fname = utils.GIBBS_NPY_FILENAMES[name]
    exported = np.load(chains["split_dir"] / fname)
    reference = read_chain(chains["straight_dir"])[name]

    assert exported.shape[0] == NITER, (
        f"{fname} holds {exported.shape[0]} samples, expected the full "
        f"{NITER}-sample chain"
    )
    np.testing.assert_array_equal(exported, reference)


def test_returned_arrays_cover_the_segment_only(chains):
    """A resumed call returns its own segment; the chain lives in the h5."""
    assert chains["first"][1].shape[0] == SPLIT
    assert chains["second"][1].shape[0] == NITER - SPLIT
    assert utils.h5_chain_length(chains["split_dir"]) == NITER


def test_resume_does_not_truncate_the_chain(chains):
    """Blocker B1: the writer must not open the file with overwrite=True."""
    straight_len = utils.h5_chain_length(chains["straight_dir"])
    assert utils.h5_chain_length(chains["split_dir"]) == straight_len == NITER


# ---------------------------------------------------------------------------
# Supporting behaviour
# ---------------------------------------------------------------------------

def test_repair_truncates_ragged_datasets(chains, tmp_path):
    """Blocker B3: a crash between two per-dataset resizes is repaired."""
    import shutil

    import h5py

    out_dir = tmp_path / "ragged"
    out_dir.mkdir()
    shutil.copy(
        utils.gibbs_sample_h5_path(chains["straight_dir"]),
        utils.gibbs_sample_h5_path(out_dir),
    )

    # Simulate an append killed after signal_amps was resized but before
    # ln_post was.
    with h5py.File(utils.gibbs_sample_h5_path(out_dir), "a") as f:
        for name in ("signal_ps", "sys_amps", "chisq", "ln_post"):
            f[name].resize(NITER - 1, axis=0)

    assert utils.h5_chain_length(out_dir) == NITER - 1
    assert utils.repair_h5_chain(out_dir, verbose=False) == NITER - 1

    import h5py as _h5py

    with _h5py.File(utils.gibbs_sample_h5_path(out_dir), "r") as f:
        assert {f[name].shape[0] for name in OUTPUTS} == {NITER - 1}


def test_rng_state_round_trip(chains):
    """Blocker B4: the saved MT19937 state restores the exact stream."""
    state = utils.load_rng_state(chains["straight_dir"], apply=False)
    assert state is not None
    assert state[0] == "MT19937"
    assert state[1].shape == (624,)

    utils.load_rng_state(chains["straight_dir"])
    a = np.random.randn(5)
    utils.load_rng_state(chains["straight_dir"])
    b = np.random.randn(5)
    np.testing.assert_array_equal(a, b)


def test_resume_rejects_mismatched_data(problem, chains, tmp_path):
    """Blocker B6: resuming against different data is refused."""
    import shutil

    out_dir = tmp_path / "mismatch"
    out_dir.mkdir()
    shutil.copy(
        utils.gibbs_sample_h5_path(chains["split_dir"]),
        utils.gibbs_sample_h5_path(out_dir),
    )

    wrong = dict(problem)
    wrong["vis"] = problem["vis"] * 2.0

    with pytest.raises(ValueError, match="does not match the chain"):
        hp.pspec.gibbs_sample(
            Niter=1, out_dir=str(out_dir), write_Niter=1, resume=True, **wrong
        )

    # ...and the chain it refused to continue is still intact.
    assert utils.h5_chain_length(out_dir) == NITER


def test_resume_without_a_chain_raises(problem, tmp_path):
    """A resume pointed at an empty directory fails loudly, not silently."""
    with pytest.raises(ValueError, match="no samples found"):
        hp.pspec.gibbs_sample(
            Niter=1,
            out_dir=str(tmp_path / "empty"),
            write_Niter=1,
            resume=True,
            **problem,
        )


def test_resume_requires_out_dir(problem):
    with pytest.raises(ValueError, match="requires out_dir"):
        hp.pspec.gibbs_sample(Niter=1, out_dir=None, resume=True, **problem)
