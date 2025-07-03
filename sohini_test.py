import numpy as np
import pylab as plt
import hydra_pspec as hp
import scipy.special
from pyuvdata import UVData

np.random.seed(11)

Ntimes = 203
freqs = np.linspace(100., 120., 120) 
Nfgmodes = 4


# Generate FG mode matrix
fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(Nfgmodes)
            ]).T


# Generate data from exactly known FG model
fg_amps_true = (1. + np.random.randn(Nfgmodes, Ntimes)) + 1.j*(1. + np.random.randn(Nfgmodes, Ntimes))
fg_amps_true[0,:] += 10.

print(fgmodes.shape, fg_amps_true.shape)
fg_true = fgmodes @ fg_amps_true

# Set power spectrum
fourier_op = hp.utils.fourier_operator(freqs.size, unitary=True)
# Make a power spectrum with a bit of a shape to it
ps_true = 0.0012 * (1. + 0.3*np.sin(3. * np.linspace(0., 1., freqs.size)))
S_true = hp.pspec.covariance_from_pspec(ps_true, fourier_op)

uvd = UVData()
vis_eor_path='/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_data/vis-eor.uvh5'
uvd.read(vis_eor_path)
uvd.conjugate_bls()
uvd = hp.utils.form_pseudo_stokes_vis(uvd)
eor_true = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)

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
noise_ps_val = 0.00004 # 0.0004
noise_ps_true = noise_ps_val * np.ones(freqs.size)
N_true = hp.pspec.covariance_from_pspec(noise_ps_true, fourier_op)
Ninv = np.diag(1./np.diag(N_true)) # get diagonal, invert, pack back into diagonal
n = np.sqrt(N_true) @ (np.random.randn(freqs.size, Ntimes) 
                       + 1.j*np.random.randn(freqs.size, Ntimes)) / np.sqrt(2.)
# Note factor of sqrt(2) above
noise_ps_check = calc_ps(n.T)

# Build systematics model
nm_list = [(0,15), (0,16)]
lsts = np.linspace(0., 1., Ntimes)
sys_modes = hp.sys_solver.sys_modes(freqs_Hz=freqs*1e6, 
                                    times_sec=lsts * 24./(2.*np.pi) * 3600., 
                                    modes=nm_list)
im=plt.matshow(sys_modes.T.real,aspect='auto')
plt.colorbar(im)
plt.title("The H operator, modes: "+str(nm_list))
plt.show()
sys_amps_true = np.array([4., 4.01])
sys_prior = 4.**2. * np.eye(sys_amps_true.size)

gain_true = (1. + sys_modes @ sys_amps_true).reshape((freqs.size, Ntimes))

# Combine together into data
d = gain_true.T * (fg_true.T + eor_true) + n.T

# FIXME: Units or normalisation issue with ps_prior?
ps_prior = np.column_stack( (1e-7 * np.ones(freqs.size),
                             1e-1 * np.ones(freqs.size)) ).T # should have shape (2, Nfreqs)

print("PS shapes:", ps_prior.shape, ps_true.shape)
print("PS max: {} PS min: {}".format(ps_true.max(),ps_true.min()))
print("PS prior max: {} PS true min: {}".format(ps_prior.max(),ps_prior.min()))
flags_i = np.ones((len(freqs),), dtype=int)
print("Shape of flags array: ",flags_i.shape)


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

signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = \
        hp.pspec.gibbs_sample(
            vis = d,
            flags = flags_i ,
            signal_ps_initial = ps_true,
            fg_modes = fgmodes,
            Ninv = Ninv,
            signal_ps_prior = ps_prior,
            Niter=10,
            seed=10,
            map_estimate=False,
            verbose=True,
            nproc=1,
            write_Niter=10,
            out_dir='./phil_test_outdir',   
            sys_modes=sys_modes,
            sys_prior=sys_prior,
            sys_initial=sys_amps_true,
            solver_tol=1e-12
        )



# Plot data residual
model = (signal_amps.mean(axis=0) + fg_amps.mean(axis=0) @ fgmodes.T)


# Show model and residual
plt.subplot(141)
plt.matshow(model.real, aspect='auto', fignum=False)
plt.title("Sampled model")
plt.colorbar()

plt.subplot(142)
plt.matshow(d.real, aspect='auto', fignum=False)
plt.title("Data")
plt.colorbar()

plt.subplot(143)
plt.matshow(d.real - model.real, aspect='auto', fignum=False)
plt.title("Residual")
plt.colorbar()

plt.subplot(144)
plt.matshow(eor_true.real, aspect='auto', fignum=False)
plt.title("EoR data")
plt.colorbar()

plt.gcf().set_size_inches((16., 4.))
plt.show()

plt.subplot(111)
plt.plot(ps_true, 'k-',label='True PS')
plt.plot(signal_ps.mean(axis=0), 'r--',label='Sampled ps')
plt.legend()
plt.show()


fig,ax = plt.subplots(3,3,figsize=(12,16))

im=ax[0,0].matshow(eor_true.real,aspect='auto')
ax[0,0].set_ylabel("True")
ax[0,0].set_title("EoR")
plt.colorbar(im)

im=ax[0,1].matshow(fg_true.real.T,aspect='auto')
plt.colorbar(im)
ax[0,1].set_title("Foregrounds")


im=ax[0,2].matshow(gain_true.real,aspect='auto')
plt.colorbar(im)
ax[0,2].set_title("Gain")


im=ax[1,0].matshow(signal_amps.real.mean(axis=0),aspect='auto')
plt.colorbar(im)
ax[1,0].set_ylabel("Sampled (mean)")

im=ax[1,1].matshow(fg_amps.real.mean(axis=0) @ fgmodes.T,aspect='auto')
plt.colorbar(im)

im=ax[1,2].matshow((1. + sys_modes @ sys_amps_true).reshape((freqs.size, Ntimes)).real,aspect='auto')
plt.colorbar(im)

im=ax[2,0].matshow((signal_amps.mean(axis=0)-eor_true).real,aspect='auto')
plt.colorbar(im)
ax[2,0].set_ylabel("Residuals")

im=ax[2,1].matshow((fg_amps.mean(axis=0) @ fgmodes.T - fg_true.T).real,aspect='auto')
plt.colorbar(im)

im=ax[2,2].matshow((gain_true-(1. + sys_modes @ sys_amps_true).reshape((freqs.size, Ntimes))).real,aspect='auto')
plt.colorbar(im)

plt.show()