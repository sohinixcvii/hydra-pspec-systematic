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
'''-------------------------------------------Graphics-----------------------------------'''
with open('res/hydra_ascii.txt', 'r') as f:
    ascii_art = f.read() 
    print(ascii_art)
'''-----------------------------------------------------------------------------------'''

'''-------------------------------------------Parameters & seed-----------------------------------'''
Ntimes = 80 #60 #203
Nfreqs = 60
freqs = np.linspace(100., 120., 120) ##120) 
freqs=freqs[:Nfreqs]
Nfgmodes = 10
Niter=100000
np.random.seed(11)
lsts = np.linspace(0., 1., Ntimes)
flags_i = np.ones((len(freqs),), dtype=int)

print("Number of times: {}, Number of freqs: {}, Number of fg modes: {}".format(Ntimes,Nfreqs,Nfgmodes))
'''-----------------------------------------------------------------------------------------------'''

'''------------------------------------Functions-----------------------'''
# Check power spectrum
def calc_ps(s):
    # NOTE: This uses inverse FFT instead of FFT to get the right normalisation
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs # CHECK: This takes an average

'''---------------------------------------------------------------------'''

'''-----------------------------Set up output directory and systematics model-------------------------'''
'''10k runs'''
# op_dir = './paper_plots/low_dl_fr_0' # low_dl_fr_0 - Case I
# op_dir = './paper_plots/high_dl_fr_0' # high_dl_fr_0 - Case II
# op_dir = './paper_plots/low_dl_fr_20' # low_dl_fr_20 - Case III

'''100k runs'''
# op_dir = './paper_plots/100k_runs/low_dl_fr_0' # low_dl_fr_0 - Case I
op_dir = './paper_plots/100k_runs/high_dl_fr_0' # high_dl_fr_0 - Case II
# op_dir = './paper_plots/100k_runs/low_dl_fr_20' # low_dl_fr_20 - Case III

# Build systematics model
# nm_list = [(3,0),(4,0),(5,0),(6,0)] #low dl fr 0 - Case I
nm_list = [(10,0), (11,0), (12,0), (13,0)] #high dl fr 0 - Case II
# nm_list = [(3,20),(4,20),(5,20),(6,20)] #low dl fr 20 - Case III

print("NM list: ",nm_list)

sys_modes = hp.sys_solver.sys_modes(freqs_Hz=freqs*1e6, 
                                    times_sec=lsts * 24./(2.*np.pi) * 3600., 
                                    modes=nm_list)

sys_amps_true = np.array([1. + 4j, 2 + 3j, 3. + 2j, 4. + 1j]) #np.array([4., 4.01])
# sys_amps_true = np.array([0.001, 0.001, 0.001, 0.001]) #np.array([4., 4.01])
sys_prior = 100**2. * np.eye(sys_amps_true.size)

gain_true = (1. + (sys_modes @ sys_amps_true).reshape([Nfreqs,Ntimes]).T)
np.save(op_dir+'/gain_true.npy',gain_true)
'''-----------------------------------------------------------------------------------------------------'''

'''--------------------------------------EoR field and power spectrum-----------------------------'''
fourier_op = hp.utils.fourier_operator(Nfreqs, unitary=True)
ps_true = 0.0012 * (1. + 0.3*np.sin(3. * np.linspace(0., 1., Nfreqs)))
S_true = hp.pspec.covariance_from_pspec(ps_true, fourier_op)

print("Shape of ps_true: {}, shape of S_true: {}".format(ps_true.shape,S_true.shape))

# Generate EoR field from this
# S_true = np.load('test_data/eor-cov.npy')
sqrt_S_true = np.linalg.cholesky(S_true)
eor_true = (sqrt_S_true @ (np.random.randn(Nfreqs,Ntimes) 
                          + 1.j*np.random.randn(Nfreqs,Ntimes)) / np.sqrt(2.)).T
# Note factor of sqrt(2) above
print("Eor_true shape: {}".format(eor_true.shape))
# Check that generated EoR field has a similar power spectrum to the true one
ps_check = calc_ps(eor_true)
np.save(op_dir+'/eor_true.npy',eor_true)
'''---------------------------------------------------------------------------------------------'''



'''-------------------------------------Foregrounds----------------------------------'''
'''Loading from uvh5'''
# uvd = UVData()
# vis_fg_path='/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_data/vis-ptsrc-gsm.uvh5' #Sohini's laptop
# uvd.read(vis_fg_path)
# uvd = hp.utils.form_pseudo_stokes_vis(uvd)
# fg_true = uvd.get_data((0, 1, "xx"))  # shape
#  (Ntimes, Nfreqs)
# np.save('npy_data/fg_true.npy',fg_true)
# uvd = UVData()
# vis_eor_path='/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_data/vis-eor.uvh5'
# uvd.read(vis_eor_path)
# uvd.conjugate_bls()
# uvd = hp.utils.form_pseudo_stokes_vis(uvd)
# eor_true = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)
# np.save('npy_data/eor_true',eor_true)

'''Loading from npy'''
# vis_eor_path = 'npy_data/eor_true.npy'
# eor_true=np.load(vis_eor_path)

vis_fg_path = 'res/npy_data/fg_true.npy'
fg_true = np.load(vis_fg_path)

# Generate FG mode matrix
fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(Nfgmodes)
            ]).T

print("Shape of fgmodes: ",fgmodes.shape)
np.save(op_dir+'/fgmodes.npy',fgmodes)
exit()

fg_true=fg_true[:Ntimes,:Nfreqs]
np.save(op_dir+'/fg_true.npy',fg_true)

# '''------------Creating dummy foregrounds-------------------'''
# A = fgmodes[:, :Nfgmodes]   # (60, 10)
# B = fg_true.T               # (60, 80)

# # Works for real or complex
# X_hat, *_ = np.linalg.lstsq(A, B, rcond=None)  # X_hat: (10, 80)
# fg_amps_fit = X_hat
# fg_fit = (fgmodes @ fg_amps_fit).T
# np.save(op_dir+'/fg_true_fit.npy',fg_fit)
# np.save(op_dir+'/fgmodes.npy',fgmodes)
# '''-----------------------------------------------------------'''

'''-----------------------------------------------------------------------------------------'''


'''-----------------------------Priors-----------------------------'''
# Set power spectrum

# eor_true=eor_true[:Ntimes,:Nfreqs]

ps_true_vis=calc_ps(eor_true)

# Define power spectrum prior range and draw sample of PS from EoR field
ps_prior = np.column_stack( (1e-7 * np.ones(Nfreqs),
                            1e-1 * np.ones(Nfreqs)) )
ps_sample = hp.pspec.sample_pspec(s=eor_true, prior=ps_prior)

print("Shape of ps_sample: {}".format(ps_sample.shape))

# No need for factor of 1/Nfreqs**2 here as sample_S() changed to iFFT normalization
S_sample = hp.pspec.covariance_from_pspec(ps_sample, fourier_op)
Sinv_sample = hp.pspec.covariance_from_pspec(1. / ps_sample, fourier_op)
'''----------------------------------------------------------------------------'''

'''-----------------------------Noise-----------------------------'''
# Generate noise
noise_ps_val = 0.0004 #0.000004 #0.000004 # 0.0004 -- usual case
noise_ps_true = noise_ps_val * np.ones(Nfreqs)
N_true = hp.pspec.covariance_from_pspec(noise_ps_true, fourier_op)
Ninv = np.diag(1./np.diag(N_true)) # get diagonal, invert, pack back into diagonal
n = np.sqrt(N_true) @ (np.random.randn(freqs.size, Ntimes) 
                    + 1.j*np.random.randn(freqs.size, Ntimes)) / np.sqrt(2.)
# Note factor of sqrt(2) above
noise_ps_check = calc_ps(n.T)
'''-----------------------------------------------------------------------------------------'''

'''-----------------------------Combine to get data-----------------------------'''
# Combine together into data
d = gain_true * (fg_true + eor_true) + n.T #Simulated FG
np.save(op_dir+'/data_true.npy',d)
'''-----------------------------------------------------------------------------------------'''


'''-----------------------------Running the sampler------------------------------'''

signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = \
        hp.pspec.gibbs_sample(
            vis=d,
            flags=flags_i,
            signal_ps_initial=ps_true,
            fg_modes=fgmodes,
            Ninv=Ninv,
            signal_ps_prior=ps_prior.T, #Should be (2, Nfreqs)
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
            sky_model_initial=(fg_true + eor_true) #(fg_true.T + eor_true)
        )


end_t = time.time()
'''-----------------------------------------------------------------------------------------'''

print("Total time taken: {}".format(end_t-start_t))