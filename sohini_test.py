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
# sys.path.append('/Users/user/Documents/Codes/hydra_sys_project1/GCR_test_scripts/')
# from functions import *

np.random.seed(11)

Ntimes = 60 #60 #203
Nfreqs = 40
freqs = np.linspace(100., 120., 120) ##120) 
Nfgmodes = 12
op_dir = './paper_plots/low_dl_fr_0' # Path to results 
freqs=freqs[:Nfreqs]

print("Number of times: {}, Number of freqs: {}, Number of fg modes: {}".format(Ntimes,Nfreqs,Nfgmodes))


# Build systematics model
nm_list = [(0,3),(0,4),(0,5),(0,6)] #low dl fr 0
# nm_list = [(0,20), (0,21), (0,22), (0,23)] #high dl fr 0
# nm_list = [(3,3),(3,4),(3,5),(3,6)] #low dl low fr

''' Loading and making the data '''
# Generate FG mode matrix
fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(Nfgmodes)
            ]).T

print("Shape of fgmodes: ",fgmodes.shape)

'''Loading from uvh5'''
# uvd = UVData()
# vis_fg_path='/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_data/vis-ptsrc-gsm.uvh5' #Sohini's laptop
# uvd.read(vis_fg_path)
# uvd = hp.utils.form_pseudo_stokes_vis(uvd)
# fg_true = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)
# np.save('npy_data/fg_true.npy',fg_true)
# uvd = UVData()
# vis_eor_path='/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_data/vis-eor.uvh5'
# uvd.read(vis_eor_path)
# uvd.conjugate_bls()
# uvd = hp.utils.form_pseudo_stokes_vis(uvd)
# eor_true = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)
# np.save('npy_data/eor_true',eor_true)

'''Loading from npy'''
vis_fg_path = 'npy_data/fg_true.npy'
fg_true = np.load(vis_fg_path)

vis_eor_path = 'npy_data/eor_true.npy'
eor_true=np.load(vis_eor_path)

fg_true=fg_true[:Ntimes,:Nfreqs]
# Set power spectrum
fourier_op = hp.utils.fourier_operator(freqs.size, unitary=True)
# Make a power spectrum with a bit of a shape to it
ps_true = 0.0012 * (1. + 0.3*np.sin(3. * np.linspace(0., 1., freqs.size)))
S_true = hp.pspec.covariance_from_pspec(ps_true, fourier_op)


eor_true=eor_true[:Ntimes,:Nfreqs]

# Check power spectrum
def calc_ps(s):
    # NOTE: This uses inverse FFT instead of FFT to get the right normalisation
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs # CHECK: This takes an average

ps_true=calc_ps(eor_true)

# Define power spectrum prior range and draw sample of PS from EoR field
ps_prior = np.column_stack( (1e-7 * np.ones(freqs.size),
                             1e-1 * np.ones(freqs.size)) )
ps_sample = hp.pspec.sample_pspec(s=eor_true, prior=ps_prior)


# No need for factor of 1/Nfreqs**2 here as sample_S() changed to iFFT normalization
S_sample = hp.pspec.covariance_from_pspec(ps_sample, fourier_op)
Sinv_sample = hp.pspec.covariance_from_pspec(1. / ps_sample, fourier_op)

# Generate noise
noise_ps_val = 0.000004 #0.000004 # 0.0004
noise_ps_true = noise_ps_val * np.ones(freqs.size)
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
np.save(op_dir+'true_gain.npy',gain_true)

# Combine together into data
d = gain_true.T * (fg_true + eor_true) + n.T

# FIXME: Units or normalisation issue with ps_prior?
ps_prior = np.column_stack( (1e-7 * np.ones(freqs.size),
                             1e-1 * np.ones(freqs.size)) ).T # should have shape (2, Nfreqs)

flags_i = np.ones((len(freqs),), dtype=int)

""" Running the sampler """
"""
fig,ax = plt.subplots(1,3,figsize=(12,4))

im=ax[0].matshow(eor_true.real,aspect='auto')
ax[0].set_title("EoR true")
plt.colorbar(im)


im=ax[1].matshow(fg_true.T.real,aspect='auto')
ax[1].set_title("FG true")
plt.colorbar(im)

im=ax[2].matshow(gain_true.T.real,aspect='auto')
ax[2].set_title("Gain true")
plt.colorbar(im)

plt.show()
"""


signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = \
        hp.pspec.gibbs_sample(
            vis=d,
            flags=flags_i,
            signal_ps_initial=ps_true,
            fg_modes=fgmodes,
            Ninv=Ninv,
            signal_ps_prior=ps_prior,
            Niter=10000,
            seed=10,
            freqs=freqs,
            lsts=np.linspace(0., 1., Ntimes),
            map_estimate=False,
            verbose=True,
            nproc=1,
            write_Niter=10000,
            out_dir=op_dir,
            sys_modes=sys_modes,
            sys_prior=sys_prior,
            sys_initial=sys_amps_true,
            solver_tol=1e-12,
            sample_systematics=True,
            sample_eor_fg=True,
            sample_signal_ps=True,
            sky_model_initial=None #(fg_true.T + eor_true)
        )


end_t = time.time()

print("Total time taken: {}".format(start_t-end_t))
'''Plot results'''
# model = (signal_amps.mean(axis=0) + fg_amps.mean(axis=0) @ fgmodes.T)
# print("Low DL FR 0 case")
# data_true = fg_true + eor_true

# model_dlfr= data_dly_fr(model, freqs, lsts, windows='blackman-harris')
# data_true_dlfr= data_dly_fr(data_true, freqs, lsts, windows='blackman-harris')
# # model = (fg_true + eor_true)

# sys_model_true = (1. + sys_modes @ sys_amps_true).reshape((freqs.size, Ntimes))
# sys_model_sampled = (1. + sys_modes @ sys_amps.mean(axis=0)).reshape((freqs.size, Ntimes))


# plt.subplot(111)
# colours = ['r', 'g', 'b', 'y', 'c', 'm']
# for i in range(sys_amps_true.size):
#     #plt.axhline(sys_amps_true[i], color=colours[i], ls='dashed')
#     plt.plot((sys_amps[:,i] - sys_amps_true[i]).imag, label="Sys mode: %s" % str(nm_list[i])) #, color=colours[i], alpha=0.5)
# plt.axhline(0., ls='dashed', color='k')
# plt.legend(loc='upper right')
# plt.xlabel("Iteration", fontsize=15)
# plt.ylabel("amp - amp_true", fontsize=15)


# plt.tight_layout()
# plt.show()
# #exit()


# print("sys_amps true:", sys_amps_true)
# print("sys_amps samp:", sys_amps.mean(axis=0))

# # Show model and residual
# plt.subplot(241)
# plt.matshow(model.real, aspect='auto', fignum=False)
# plt.title("Sampled model")
# plt.colorbar()

# plt.subplot(242)
# plt.matshow(d.real, aspect='auto', fignum=False)
# plt.title("Data")
# plt.colorbar()

# plt.subplot(243)
# plt.matshow(d.real - (sys_model_sampled.T * model).real, aspect='auto', fignum=False)
# plt.title("Residual")
# plt.colorbar()

# plt.subplot(244)
# plt.matshow(n.real.T, aspect='auto', fignum=False)
# plt.title("Noise")
# plt.colorbar()

# plt.subplot(245)
# plt.matshow(eor_true.real, aspect='auto', fignum=False)
# plt.title("EoR true")
# plt.colorbar()

# plt.subplot(246)
# plt.matshow(signal_amps.mean(axis=0).real, aspect='auto', fignum=False)
# plt.title("EoR sampled")
# plt.colorbar()

# plt.subplot(247)
# plt.matshow(sys_model_true.real, aspect='auto', fignum=False)
# plt.title("Systematics true")
# plt.colorbar()

# plt.subplot(248)
# plt.matshow(sys_model_sampled.real, aspect='auto', fignum=False)
# plt.title("Systematics sampled")
# plt.colorbar()

# plt.gcf().set_size_inches((20., 6.))
# plt.tight_layout()
# plt.show()


# plt.subplot(111)
# plt.plot(signal_ps.T, 'r-', alpha=0.15)
# plt.plot(ps_true, 'k-',label='True PS')
# plt.plot(ps_true[::-1], 'k--',label='True PS reversed')
# plt.plot(np.fft.fftshift(ps_true), 'k--',label='True PS fftshift')
# plt.legend()
# plt.show()

# times = Quantity(np.unique(lsts * 12 / np.pi), unit='h')
# freqs_mhz = Quantity(freqs/1e6, unit='Hz')
# xticklocs=[0,20,40,60,80,100,119]
# yticklocs=[0,25,50,75,100,125,150,175,200]
# xstep = (freqs[-1]-freqs[0])/freqs.size
# ystep = (lsts[-1]-lsts[0])/Ntimes

# xticks= freqs_mhz[xticklocs]
# yticks=times[yticklocs]

# fig, ax = plt.subplots(2,3,figsize=(21,42))
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_powerlimits((6, 6))  # Force sci notation for values >= 1e6

# im=ax[0,0].matshow(data_true.real,aspect='auto',origin='lower',cmap=cmc.acton)
# plt.colorbar(im)

# im=ax[0,1].matshow(model.real,aspect='auto',origin='lower',cmap=cmc.acton)
# plt.colorbar(im)

# im=ax[0,2].matshow(data_true.real-model.real,aspect='auto',origin='lower',cmap=cmc.acton)
# plt.colorbar(im)

# im=ax[1,0].matshow(data_true_dlfr.real,aspect='auto',origin='lower',cmap=cmc.acton)
# plt.colorbar(im)

# im=ax[1,1].matshow(model_dlfr.real,aspect='auto',origin='lower',cmap=cmc.acton)
# plt.colorbar(im)

# im=ax[1,2].matshow(data_true_dlfr.real-model_dlfr.real,aspect='auto',origin='lower',cmap=cmc.acton)
# plt.colorbar(im)

# plt.show()
