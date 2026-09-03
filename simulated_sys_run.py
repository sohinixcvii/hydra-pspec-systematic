import numpy as np
import pylab as plt
import hydra_pspec as hp
import scipy.special
from pyuvdata import UVData
from astropy.units import Quantity
from astropy import units
import matplotlib.ticker as ticker
import cmcrameri.cm as cmc
import sys 
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

'''Loading from uvh5'''
uvd = UVData()
vis_path='/nvme2/scratch/sohini/hydra-pspec-systematic/hera_val/vis_corrupted_test.uvh5' #Sohini's laptop
uvd.read(vis_path)
uvd = hp.utils.form_pseudo_stokes_vis(uvd)
vis = uvd.get_data((0, 1, "xx"))

uvd = UVData()
vis_fg_path='/nvme2/scratch/sohini/hydra-pspec-systematic/test_data/vis-ptsrc-gsm.uvh5' #Sohini's laptop
uvd.read(vis_fg_path)
uvd = hp.utils.form_pseudo_stokes_vis(uvd)
fg_true = uvd.get_data((0, 1, "xx"))  # shape

uvd = UVData()
vis_eor_path='/nvme2/scratch/sohini/hydra-pspec-systematic/test_data/vis-eor.uvh5'
uvd.read(vis_eor_path)
uvd.conjugate_bls()
uvd = hp.utils.form_pseudo_stokes_vis(uvd)
eor_true = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)
freqs=uvd.freq_array
lsts=uvd.lst_array

Ntimes = len(lsts)
Nfreqs = len(freqs)

Nfgmodes = 12
Niter=10

op_dir = './paper_plots/sim_data'

# Build systematics model
# nm_list = [(10,0), (11,0), (12,0), (13,0)] #high dl fr 0
# [(dl,fr)]
nm_list = [(32, 93), (32, 94), (32, 95), (32, 96), (32, 97), (32, 98), (32, 99), (32, 100), (32, 101), (33, 93)]


print("Number of times: {}, Number of freqs: {}, Number of fg modes: {}".format(Ntimes,Nfreqs,Nfgmodes))
fourier_op = hp.utils.fourier_operator(Nfreqs, unitary=True)
ps_true = calc_ps(eor_true)
S_true = hp.pspec.covariance_from_pspec(ps_true, fourier_op)

print("Shape of ps_true: {}, shape of S_true: {}".format(ps_true.shape,S_true.shape))

''' Loading and making the data '''
# Generate FG mode matrix
fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(Nfgmodes)
            ]).T

print("Shape of fgmodes: ",fgmodes.shape)

# Note factor of sqrt(2) above
print("Eor_true shape: {}".format(eor_true.shape))

ps_true_vis=calc_ps(eor_true)

# Define power spectrum prior range and draw sample of PS from EoR field
ps_prior = np.column_stack( (1e-7 * np.ones(Nfreqs),
                            1e-1 * np.ones(Nfreqs)) )
ps_sample = hp.pspec.sample_pspec(s=eor_true, prior=ps_prior)

print("Shape of ps_sample: {}".format(ps_sample.shape))
# No need for factor of 1/Nfreqs**2 here as sample_S() changed to iFFT normalization
S_sample = hp.pspec.covariance_from_pspec(ps_sample, fourier_op)
Sinv_sample = hp.pspec.covariance_from_pspec(1. / ps_sample, fourier_op)

# Generate noise
noise_ps_val = 0.0004 #0.000004 #0.000004 # 0.0004
noise_ps_true = noise_ps_val * np.ones(Nfreqs)
N_true = hp.pspec.covariance_from_pspec(noise_ps_true, fourier_op)
Ninv = np.diag(1./np.diag(N_true)) # get diagonal, invert, pack back into diagonal
n = np.sqrt(N_true) @ (np.random.randn(freqs.size, Ntimes) 
                    + 1.j*np.random.randn(freqs.size, Ntimes)) / np.sqrt(2.)
# Note factor of sqrt(2) above
noise_ps_check = calc_ps(n.T)

print("NM list: ",nm_list)
sys_modes = hp.sys_solver.sys_modes(freqs_Hz=freqs*1e6, 
                                    times_sec=lsts * 24./(2.*np.pi) * 3600., 
                                    modes=nm_list)

sys_amps_true = np.ones(len(nm_list)) #np.array([4., 4.01])
sys_prior = 4**2. * np.eye(sys_amps_true.size)

gain_true = (1. + sys_modes @ sys_amps_true).reshape((Nfreqs, Ntimes))
np.save(op_dir+'/gain_true.npy',gain_true)

# Assign visibility to data
d = vis

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
            sys_modes=sys_modes,
            sys_prior=sys_prior,
            sys_initial=sys_amps_true,
            solver_tol=1e-13,
            sample_systematics=True,
            sample_eor_fg=True,
            sample_signal_ps=True,
            sky_model_initial=(fg_true+eor_true) #(fg_true.T + eor_true)
        )


end_t = time.time()

print("Total time taken: {}".format(end_t-start_t))
