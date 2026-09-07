"""
sys_sampler_wrapper.py
----------------------
Wrapper script for running the Hydra-pspec Gibbs sampler with a multiplicative
systematics model. Loads or generates EoR and foreground visibilities, builds
a systematics gain model, defines priors, and runs the sampler.

Usage
-----
    python sys_sampler_wrapper.py
    python sys_sampler_wrapper.py --Niter 10000
    python sys_sampler_wrapper.py --resume --Niter 10000

`--resume` continues the chain already in the output directory instead of
starting a new one, and `--Niter` is then the number of *additional*
iterations. On a resume the "true" arrays (`data_true`, `eor_true`, `fg_true`,
`gain_true`) are loaded from the output directory rather than regenerated, so
the resumed segment is guaranteed to sample the same data as the first: the
generation here is only deterministic as long as `Ntimes`, `Nfreqs`,
`dummy_flag` and the seed are untouched, and a resume against different data
would otherwise fail silently.

See docs/warm-start.md.

Output
------
    Sampler products written to `op_dir` (see Configuration section).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.special
import hydra_pspec as hp


# =============================================================================
# Timing
# =============================================================================
start_t = time.time()

with open('res/hydra_ascii.txt', 'r') as f:
    print(f.read())


# =============================================================================
# Configuration
# =============================================================================
Ntimes   = 80
Nfreqs   = 60
Nfgmodes = 10
Niter    = 250000

# Set to True to draw EoR from a Gaussian random field;
# False to load the Burba et al. simulated EoR.
dummy_flag = False

# Output directory for sampler products
op_dir = './paper_plots/250k_run/low_dl_fr_20'


# =============================================================================
# Command line
# =============================================================================
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="continue the chain already in the output directory instead of "
         "starting a new one; --Niter is then the number of ADDITIONAL "
         "iterations to run",
)
parser.add_argument(
    "--Niter",
    type=int,
    default=Niter,
    help=f"iterations to run (default: {Niter}). With --resume this is the "
         "number of additional iterations, not the total chain length.",
)
parser.add_argument(
    "--out-dir",
    default=op_dir,
    help=f"output directory for sampler products (default: {op_dir})",
)
parser.add_argument(
    "--no-export-npy",
    action="store_true",
    help="skip the .npy export of the chain. Every sample is still written to "
         "gibbs_samples.h5; use this when the chain is too large to duplicate "
         "on disk (signal_amps is 19.2 GB for a 250k-iteration 80x60 run).",
)
args = parser.parse_args()

Niter  = args.Niter
op_dir = args.out_dir
resume = args.resume

op_path = Path(op_dir)
op_path.mkdir(parents=True, exist_ok=True)

# Seeds the generation of the simulated data below (not the sampler, which is
# seeded separately by gibbs_sample). On a resume the arrays it produces are
# loaded from disk instead, so this seed is only load-bearing for a fresh run.
np.random.seed(11)


def save_output(name, arr):
    """
    Write `arr` to `op_dir/name`, unless resuming.

    On a resume these files are the record of what the running chain is
    sampling; overwriting them with regenerated arrays would destroy the only
    evidence of the data the first segment actually used.
    """
    if resume:
        return
    np.save(op_path / name, arr)


print(f"Output directory: {op_dir}")
print(f"Mode: {'resume' if resume else 'new chain'}, Niter={Niter}")

# Systematics mode pairs (delay index n, fringe-rate index m)
# Case I  : nm_list = [(3,0),  (4,0),  (5,0),  (6,0)]
# Case II : nm_list = [(10,0), (11,0), (12,0), (13,0)]
# Case III: nm_list = [(3,20), (4,20), (5,20), (6,20)]
# Truncated Case I  : nm_list = [(3,0),  (4,0)]
# Truncated Case II : nm_list = [(5,0), (6,0)]
# Truncated Case III: nm_list = [(3,8), (4,8)]
nm_list = [(3,20), (4,20), (5,20), (6,20)]   # Case III - truncated

# True systematics amplitudes
sys_amps_true = np.array([1. + 4j, 2. + 3j, 3. + 2j, 4. + 1j])
# sys_amps_true = np.array([12. + 5j, 4. + 20j]) #truncated case

# Noise power spectrum amplitude
noise_ps_val = 0.0004


# =============================================================================
# Helper functions
# =============================================================================
def calc_ps(s):
    """
    Compute the delay power spectrum of visibility data.

    Uses an inverse FFT normalisation to match the Hydra-pspec convention.

    Parameters
    ----------
    s : ndarray, shape (Ntimes, Nfreqs)
        Visibility data (real or complex).

    Returns
    -------
    ps : ndarray, shape (Nfreqs,)
        Time-averaged delay power spectrum.
    """
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs_ = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs_


# =============================================================================
# Frequency and LST grids
# =============================================================================
freqs = np.linspace(100., 120., 120)[:Nfreqs]   # MHz
lsts  = np.linspace(0., 1., Ntimes)

print(f"Ntimes={Ntimes}, Nfreqs={Nfreqs}, Nfgmodes={Nfgmodes}")
print(f"NM list: {nm_list}")


# =============================================================================
# Systematics model
# =============================================================================
sys_modes = hp.sys_solver.sys_modes(
    freqs_Hz   = freqs * 1e6,
    times_sec  = lsts * 24. / (2. * np.pi) * 3600.,
    modes      = nm_list,
)
sys_prior = 100.**2 * np.eye(sys_amps_true.size)

gain_true = (1. + (sys_modes @ sys_amps_true).reshape([Nfreqs, Ntimes]).T)
save_output('gain_true.npy', gain_true)


# =============================================================================
# EoR field and power spectrum
# =============================================================================
fourier_op = hp.utils.fourier_operator(Nfreqs, unitary=True)

if resume:
    # Load what the first segment sampled rather than regenerating it. The
    # generation below is deterministic under np.random.seed(11), so it happens
    # to reproduce the same arrays -- but only while Ntimes, Nfreqs and
    # dummy_flag are unchanged. Loading removes that dependency entirely.
    eor_true = np.load(op_path / 'eor_true.npy')
    if not dummy_flag:
        lsts  = np.load('res/npy_data/lsts_full.npy')[:Ntimes]
        freqs = np.load('res/npy_data/freqs_full.npy')[:Nfreqs] * 10e-6
    ps_true = calc_ps(eor_true)
elif dummy_flag:
    ps_true = 0.0012 * (1. + 0.3 * np.sin(3. * np.linspace(0., 1., Nfreqs)))
    S_true  = hp.pspec.covariance_from_pspec(ps_true, fourier_op)

    sqrt_S_true = np.linalg.cholesky(S_true)
    eor_true = (
        sqrt_S_true
        @ (np.random.randn(Nfreqs, Ntimes) + 1.j * np.random.randn(Nfreqs, Ntimes))
        / np.sqrt(2.)
    ).T
else:
    eor_true = np.load('res/npy_data/eor_true.npy')
    S_true   = np.load('res/test_data/eor-cov.npy')
    lsts     = np.load('res/npy_data/lsts_full.npy')[:Ntimes]
    freqs    = np.load('res/npy_data/freqs_full.npy')[:Nfreqs] * 10e-6
    eor_true = eor_true[:Ntimes, :Nfreqs]
    ps_true  = calc_ps(eor_true)

save_output('eor_true.npy', eor_true)
print(f"EoR shape: {eor_true.shape}")


# =============================================================================
# Foregrounds
# =============================================================================
if resume:
    fg_true = np.load(op_path / 'fg_true.npy')
else:
    fg_true = np.load('res/npy_data/fg_true.npy')[:Ntimes, :Nfreqs]

fgmodes = np.array([
    scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
    for i in range(Nfgmodes)
]).T

print(f"FG modes shape: {fgmodes.shape}")

save_output('fgmodes.npy', fgmodes)
save_output('fg_true.npy', fg_true)


# =============================================================================
# Priors
# =============================================================================
ps_prior = np.column_stack((
    1e-7 * np.ones(Nfreqs),
    1e-1 * np.ones(Nfreqs),
))
ps_sample = hp.pspec.sample_pspec(s=eor_true, prior=ps_prior)
print(f"PS sample shape: {ps_sample.shape}")

S_sample    = hp.pspec.covariance_from_pspec(ps_sample, fourier_op)
Sinv_sample = hp.pspec.covariance_from_pspec(1. / ps_sample, fourier_op)


# =============================================================================
# Noise
# =============================================================================
noise_ps_true = noise_ps_val * np.ones(Nfreqs)
N_true = hp.pspec.covariance_from_pspec(noise_ps_true, fourier_op)
Ninv   = np.diag(1. / np.diag(N_true))
n      = (
    np.sqrt(N_true)
    @ (np.random.randn(freqs.size, Ntimes) + 1.j * np.random.randn(freqs.size, Ntimes))
    / np.sqrt(2.)
)


# =============================================================================
# Data
# =============================================================================
if resume:
    # The noise realisation `n` drawn above is discarded on a resume: the data
    # the chain is conditioned on is whatever the first segment used.
    d = np.load(op_path / 'data_true.npy')
else:
    d = gain_true * (fg_true + eor_true) + n.T
save_output('data_true.npy', d)


# =============================================================================
# Run the Gibbs sampler
# =============================================================================
signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = hp.pspec.gibbs_sample(
    vis                = d,
    flags              = np.ones((len(freqs),), dtype=int),
    signal_ps_initial  = ps_true,
    fg_modes           = fgmodes,
    Ninv               = Ninv,
    signal_ps_prior    = ps_prior.T,
    Niter              = Niter,
    seed               = 10,
    freqs              = freqs,
    lsts               = np.linspace(0., 1., Ntimes),
    map_estimate       = False,
    verbose            = True,
    nproc              = 1,
    write_Niter        = Niter,
    out_dir            = op_dir,
    resume             = resume,
    export_npy         = not args.no_export_npy,
    sys_modes          = sys_modes,
    sys_prior          = sys_prior,
    sys_initial        = sys_amps_true,
    solver_tol         = 1e-13,
    sample_systematics = True,
    sample_eor_fg      = True,
    sample_signal_ps   = True,
    sky_model_initial  = fg_true + eor_true,
)

print(f"Total time taken: {time.time() - start_t:.1f}s")