import numpy as np
import pylab as plt
import hydra_pspec as hp
import scipy.special
from pyuvdata import UVData
from astropy.units import Quantity
from astropy import units
import matplotlib.ticker as ticker
import cmcrameri.cm as cmc
import os 
import time 

start_t= time.time()

np.random.seed(11)

# Check power spectrum
def calc_ps(s):
    # NOTE: This uses inverse FFT instead of FFT to get the right normalisation
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs # CHECK: This takes an average

Ntimes = 80 #60 #203
Nfreqs = 60
freqs = np.linspace(100., 120., 120) ##120) 
Nfgmodes = 12
Niter=10000

# op_dir = './paper_plots/masked_data'
# op_dir = './paper_plots/filtered_data'
# op_dir = './paper_plots/filtered_1000'
# op_dir = './paper_plots/masked_1000'
op_dir = './paper_plots/true_sky'

# print("Running Masked data case")
# print("Running filtered data case")
print("Running true sky case")
# Build systematics model
nm_list = [(10,0), (11,0), (12,0), (13,0)] #high dl fr 0

freqs=freqs[:Nfreqs]

print("PID: {}".format(os.getpid()))
print("Setup:\nNiter: {}\n".format(Niter))
print("Number of times: {}, Number of freqs: {}, Number of fg modes: {}".format(Ntimes,Nfreqs,Nfgmodes))
ps_true = 0.0012 * (1. + 0.3*np.sin(3. * np.linspace(0., 1., Nfreqs)))


''' Loading and making the data '''
# Generate FG mode matrix
fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(Nfgmodes)
            ]).T

print("Shape of fgmodes: ",fgmodes.shape)


eor_true = np.load('./paper_plots/high_dl_fr_0/eor_true.npy')

# Note factor of sqrt(2) above
print("Eor_true shape: {}".format(eor_true.shape))

'''Loading from npy'''
vis_fg_path = 'npy_data/fg_true.npy'
fg_true = np.load(vis_fg_path)

fg_true=fg_true[:Ntimes,:Nfreqs]


# Define power spectrum prior range and draw sample of PS from EoR field
ps_prior = np.column_stack( (1e-7 * np.ones(Nfreqs),
                            1e-1 * np.ones(Nfreqs)) )


# Generate noise
noise_ps_val = 0.0004 #0.000004 #0.000004 # 0.0004
noise_ps_true = noise_ps_val * np.ones(Nfreqs)
fourier_op = hp.utils.fourier_operator(Nfreqs, unitary=True)
N_true = hp.pspec.covariance_from_pspec(noise_ps_true, fourier_op)
Ninv = np.diag(1./np.diag(N_true)) # get diagonal, invert, pack back into diagonal
n = np.sqrt(N_true) @ (np.random.randn(freqs.size, Ntimes) 
                    + 1.j*np.random.randn(freqs.size, Ntimes)) / np.sqrt(2.)
# Note factor of sqrt(2) above
noise_ps_check = calc_ps(n.T)


print("NM list: ",nm_list)
lsts = np.linspace(0., 1., Ntimes)
sys_modes = hp.sys_solver.sys_modes(freqs_Hz=freqs*1e6, 
                                    times_sec=lsts * 24./(2.*np.pi) * 3600., 
                                    modes=nm_list)

sys_amps_true = np.array([4., 4.1, 5., -2.]) #np.array([4., 4.01])
sys_prior = 4**2. * np.eye(sys_amps_true.size)

gain_true = (1. + sys_modes @ sys_amps_true).reshape((Nfreqs, Ntimes))
np.save(op_dir+'/gain_true.npy',gain_true)

# Loading masked data
# d = np.load('./masked_data/masked_vis.npy')
# d = np.load('./filtered_data/filtered_vis.npy')
d = eor_true + fg_true
# FIXME: Units or normalisation issue with ps_prior?
ps_prior = np.column_stack( (1e-7 * np.ones(freqs.size),
                            1e-1 * np.ones(freqs.size)) ).T # should have shape (2, Nfreqs)

flags_i = np.ones((len(freqs),), dtype=int)

""" Running the sampler """

signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = \
        hp.pspec.gibbs_sample(
            vis=d,
            flags=flags_i,
            signal_ps_initial=ps_true,
            fg_modes=fgmodes,
            Ninv=Ninv,
            signal_ps_prior=ps_prior,
            Niter=Niter,
            seed=10,
            freqs=freqs,
            lsts=np.linspace(0., 1., Ntimes),
            map_estimate=False,
            verbose=True,
            nproc=1,
            write_Niter=Niter,
            out_dir=op_dir,
            sys_modes=np.ones_like(sys_modes),
            sys_prior=sys_prior,
            sys_initial=np.ones_like(sys_amps_true),
            solver_tol=1e-13,
            sample_systematics=False,
            sample_eor_fg=True,
            sample_signal_ps=True,
            sky_model_initial=np.zeros_like(fg_true+eor_true) #(fg_true.T + eor_true)
        )


end_t = time.time()

print("Total time taken: {}".format(end_t-start_t))
