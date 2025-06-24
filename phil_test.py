
import numpy as np
import pylab as plt
import hydra_pspec as hp
import scipy.special

np.random.seed(11)

Ntimes = 40
freqs = np.linspace(100., 120., 60) 
#  FIXME: This scales really badly? 40 is very slow for example
# Bad scaling probably due to issues with inv/pinv hanging in build_matrices
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

"""
# Plot FG model to check what it looks like
plt.subplot(121)
plt.matshow(fg_true.real, aspect='auto', fignum=False)
plt.colorbar()
plt.xlabel("Time sample")
plt.ylabel("Freq. channel")

plt.subplot(122)
plt.matshow(fg_true.imag, aspect='auto', fignum=False)
plt.colorbar()
plt.xlabel("Time sample")
plt.ylabel("Freq. channel")

plt.gcf().set_size_inches((10., 4.))
plt.show()
"""

# Set power spectrum
fourier_op = hp.utils.fourier_operator(freqs.size, unitary=True)
# Make a power spectrum with a bit of a shape to it
ps_true = 0.0012 * (1. + 0.3*np.sin(3. * np.linspace(0., 1., freqs.size)))
S_true = hp.pspec.covariance_from_pspec(ps_true, fourier_op)


"""
# Plot input 21cm power spectrum and covariance matrix
plt.subplot(121)
plt.plot(ps_true)
plt.xlabel("Fourier mode idx")
plt.ylabel("PS")

plt.subplot(122)
plt.matshow(np.log10(np.abs(S_true.real)), aspect='auto', fignum=False)
plt.colorbar()
plt.xlabel("Freq. channel")
plt.ylabel("Freq. channel")
plt.show()
"""

# Check power spectrum
def calc_ps(s):
    # NOTE: This uses inverse FFT instead of FFT to get the right normalisation
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs # CHECK: This takes an average

# Generate EoR field from this
sqrt_S_true = np.linalg.cholesky(S_true)
eor_true = sqrt_S_true @ (np.random.randn(freqs.size, Ntimes) 
                          + 1.j*np.random.randn(freqs.size, Ntimes)) / np.sqrt(2.)
# Note factor of sqrt(2) above

# Check that generated EoR field has a similar power spectrum to the true one
ps_check = calc_ps(eor_true.T)

"""
# Plot EoR model to check what it looks like
plt.subplot(221)
plt.matshow(eor_true.real, aspect='auto', fignum=False)
plt.colorbar()
plt.xlabel("Time sample")
plt.ylabel("Freq. channel")

plt.subplot(222)
plt.matshow(eor_true.imag, aspect='auto', fignum=False)
plt.colorbar()
plt.xlabel("Time sample")
plt.ylabel("Freq. channel")

plt.subplot(223)
plt.plot(ps_true, 'k-')
plt.plot(ps_check, 'r--')
plt.xlabel("Fourier mode idx")
plt.ylabel("PS")
plt.gcf().set_size_inches((10., 8.))
plt.show()


exit()
"""


ps_prior = np.column_stack( (1e-7 * np.ones(freqs.size),
                             1e-1 * np.ones(freqs.size)) )
ps_sample = hp.pspec.sample_pspec(s=eor_true.T, prior=ps_prior)

print("ps_sample", ps_sample.shape)

# No need for factor of 1/Nfreqs**2 here as sample_S() changed to iFFT normalization
S_sample = hp.pspec.covariance_from_pspec(ps_sample, fourier_op)
Sinv_sample = hp.pspec.covariance_from_pspec(1. / ps_sample, fourier_op)

"""
yy = Sinv_sample @ S_sample
plt.matshow(yy.imag)
plt.colorbar()
plt.show()

#plt.plot(ps_true, 'k-')
#plt.plot(ps_sample, 'r--')
#plt.show()
exit()
"""

# Generate noise
noise_ps_val = 0.00004 # 0.0004
noise_ps_true = noise_ps_val * np.ones(freqs.size)
N_true = hp.pspec.covariance_from_pspec(noise_ps_true, fourier_op)
Ninv = np.diag(1./np.diag(N_true)) # get diagonal, invert, pack back into diagonal
n = np.sqrt(N_true) @ (np.random.randn(freqs.size, Ntimes) 
                       + 1.j*np.random.randn(freqs.size, Ntimes)) / np.sqrt(2.)
# Note factor of sqrt(2) above
noise_ps_check = calc_ps(n.T)

"""
# Plot noise to check what it looks like
plt.subplot(221)
plt.matshow(n.real, aspect='auto', fignum=False)
plt.colorbar()
plt.xlabel("Time sample")
plt.ylabel("Freq. channel")

plt.subplot(222)
plt.matshow(n.imag, aspect='auto', fignum=False)
plt.colorbar()
plt.xlabel("Time sample")
plt.ylabel("Freq. channel")

plt.subplot(223)
plt.plot(ps_true, 'k-', label="EoR PS true")
plt.plot(ps_check, 'r--', label="EoR PS check")

plt.plot(noise_ps_true, 'k-', alpha=0.3, label="Noise PS true")
plt.plot(noise_ps_check, 'b--', label="Noise PS meas.")

plt.ylim((0., 1.5*np.max(ps_check)))

plt.legend(loc='upper right')
plt.xlabel("Fourier mode idx")
plt.ylabel("PS")
plt.gcf().set_size_inches((10., 8.))
plt.show()
"""

# Combine together into data
d = fg_true + eor_true + n

"""
plt.subplot(121)
plt.matshow(d.real, aspect='auto', fignum=False)
plt.colorbar()

plt.subplot(122)
plt.matshow(d.imag, aspect='auto', fignum=False)
plt.colorbar()

plt.gcf().set_size_inches((10., 4.))
plt.show()
"""

#ps_prior = np.column_stack( (0.1 * ps_true.min() * np.ones(freqs.size),
#                             10. * ps_true.max() * np.ones(freqs.size)) )

# FIXME: Units or normalisation issue with ps_prior?
ps_prior = np.column_stack( (1e-7 * np.ones(freqs.size),
                             1e-1 * np.ones(freqs.size)) ).T # should have shape (2, Nfreqs)

print("PS:", ps_prior.shape, ps_true.shape)


signal_cr, signal_ps, fg_amps, b_sys, chisq, ln_post = \
        hp.pspec.gibbs_sample_with_fg(
            vis = d.T,
            flags = np.ones(freqs.size, dtype=int),
            ps_initial = ps_true,
            fgmodes = fgmodes,
            Ninv = Ninv,
            ps_prior = ps_prior,
            Niter=4,
            seed=10,
            map_estimate=False,
            verbose=True,
            nproc=1,
            write_Niter=10000,
            out_dir='./phil_test_outdir',
            freqs=freqs,
            lsts=np.linspace(0., 1., Ntimes),
            nm_list=np.array([(0,0)]),
        )


#print(signal_ps)

"""
plt.subplot(111)
plt.plot(ps_true, 'k-')
plt.plot(signal_ps[0], 'r--')
plt.show()
"""


"""
plt.subplot(121)
plt.matshow(signal_cr[0].real, aspect='auto', fignum=False)
plt.colorbar()

plt.subplot(122)
plt.matshow(eor_true.real.T, aspect='auto', fignum=False)
plt.colorbar()

plt.gcf().set_size_inches((8., 4.))
plt.show()
"""

"""
# FG amps
for i in range(fg_amps_true.shape[0]):
    plt.plot(fg_amps_true[i,:], 'k-')
    plt.plot(fg_amps[0,:,i], ls='dashed')
plt.show()
"""

plt.subplot(121)
for i in range(4):
    #plt.plot(eor_true.T[i].real, color='k')
    #plt.plot(signal_cr[0,i].real, ls='dashed')

    plt.plot(signal_cr[0,i].real - eor_true.T[i].real)


plt.subplot(122)
plt.plot(ps_true, 'k-')
plt.plot(signal_ps.T, 'r-', alpha=0.3)

plt.show()