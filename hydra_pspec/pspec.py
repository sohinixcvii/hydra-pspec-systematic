import numpy as np
import scipy as sp
import scipy.linalg
from scipy.stats import invgamma
from scipy.interpolate import interp1d
from . import sys_solver as sys_sol
from multiprocess import current_process
from . import utils
import time
from uvtools.dspec import gen_window
from uvtools.utils import FFT

def data_dly_fr(data, freqs, times, windows=None,
                    freq_window_kwargs=None, time_window_kwargs=None):
    """
    Transform data to delay fringe-rate space
    
    This function takes a 2D array of visibility data (in units of Jy), as well 
    as the corresponding frequency and time arrays (in units of Hz and JD, respectively), 
    and makes a 2x2 grid of plots where each plot shows each one of the possible choices 
    for Fourier transforming along an axis. The upper-left plot is in the frequency-time 
    domain; the upper-right plot is in the frequency-fringe-rate domain; the lower-left 
    plot is in the delay-time domain; and the lower-right plot is in the delay-fringe-rate 
    domain.
    
    Parameters
    ----------
    data : ndarray, shape=(NTIMES,NFREQS)
        Array containing the visibility to be plotted. Assumed to be in units of Jy. 
        
    freqs : ndarray, shape=(NFREQS,)
        Array containing the observed frequencies. Assumed to be in units of Hz.
        
    times : ndarray, shape=(NTIMES,)
        Array containing the observed times. Assumed to be in units of JD.
        
    windows : tuple of str or str, optional
        Choice of taper to use for the fringe-rate and delay transforms. Must be 
        either tuple, list, or string. If a tuple or list, then it must be either 
        length 1 or length 2; if it is length 2, then the zeroth entry is the taper 
        to be applied along the time axis for the fringe-rate transform, with the 
        other entry specifying the taper to be applied along the frequency axis 
        for the delay transform. Each entry is passed to uvtools.dspec.gen_window. 
        If ``windows`` is a length 1 tuple/list or a string, then it is assumed 
        that the same taper is to be used for both axes. Default is to use no 
        taper (or, equivalently, a boxcar).

    freq_window_kwargs : dict, optional
        Keyword arguments to pass to uvtools.dspec.gen_window for generating the 
        frequency taper. Default is to pass no keyword arguments.
        
    time_window_kwargs : dict, optional
        Keyword arguments to pass to uvtools.dspec.gen_window for generating the 
        time taper. Default is to pass no keyword arguments.
    
    Returns
    -------
    data_dl_fr :
        data in delay-fringe rate space
    """
    # do some data prep
    freq_window_kwargs = freq_window_kwargs or {}
    time_window_kwargs = time_window_kwargs or {}
    if windows is not None:
        time_window = gen_window(windows, times.size, **time_window_kwargs)
        freq_window = gen_window(windows, freqs.size, **freq_window_kwargs)
    else:
        time_window = gen_window(None, times.size, **time_window_kwargs)
        freq_window = gen_window(None, freqs.size, **freq_window_kwargs)
        
    time_window = time_window[:, None]
    freq_window = freq_window[None, :]
<<<<<<< HEAD
    # data_fr = FFT(data * time_window, axis=0)
    # data_dly = FFT(data * freq_window, axis=1)
    data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)
    
    # fringe_rates = fourier_freqs(times * units.day.to('s')) * 1e3 # mHz
    # dlys = fourier_freqs(freqs) * 1e9 # ns

    return data_fr_dly

def sample_S(s=None, sk=None, prior=None, max_prior_iter=1000):
    """
    Draw samples of the bandpowers of S, p(S|s). This assumes that the conditional
    distributions for the bandpowers are uncorrelated with one another, i.e. the Fourier-
    space covariance S has no off-diagonals.

    Parameters:
        s (array_like):
            A set of real-space samples of the field, of shape
            `(Ntimes, Nfreq)`. This will be Fourier transformed.
            Alternatively, `sk` can be given.

        sk (array_like):
            A set of Fourier-space samples of the field, of shape
            `(Ntimes, Nfreq)`.  The monopole is expected to be at the center
            of the frequency axis, i.e. the frequency axis has been fftshifted.

        prior (array_like):
            Array of delta function prior values, used to set certain modes to a
            fixed value.
    """
    if s is None and sk is None:
        raise ValueError("Must pass in s (real space) or sk (Fourier space) vector.")

    if sk is None:
        axes = (1,)
        sk = np.fft.ifftshift(s, axes=axes)
        sk = np.fft.fftn(sk, axes=axes)
        sk = np.fft.fftshift(sk, axes=axes)
    # np.savetxt('sk_save',sk)
    Nobs, Nfreqs = sk.shape

    if prior is None:
        prior = np.zeros((2, Nfreqs), dtype=float)

    # The scale parameter for the inverse gamma distribution (beta) is
    # equivalent to (Ntimes - 1) times the variance over the time axis of the
    # delay spectrum of the Gaussian Constrained Realization of the EoR
    beta = np.sum(sk * sk.conj(), axis=0).real
    # The shape parameter (alpha) differs from that used in Eriksen et al. 2008
    # i.e. `alpha = Nobs/2 - 1` because our data vector is complex and has
    # twice as many numbers as a purely real data vector
    alpha = Nobs - 1.0

    # We obtain samples of the power spectrum (x) by instead sampling the random
    # variable y = x / beta and then obtain x via x = y * beta
    x = np.zeros(Nfreqs)
    # print("Nfreqs: ",Nfreqs)
    for i in range(Nfreqs):
        # print("Iteration: ",i)
        # print("Priors: ",prior[:,i])
        # print("Beta value: ",beta[i]," Alpha: ",alpha)
        if np.any(prior[:, i] > 0):
            # print("If triggered")

            # The pdf for a log-uniform prior is proportional to 1 / x.
            # Multiplying the inverse gamma likelihood by this prior results
            # in an additional factor of 1 / x which increases the effective
            # value of the shape parameter (alpha) by 1.  With a log-uniform
            # prior, we thus sample from an inverse gamma distribution with
            # shape parameter alpha + 1.
            x[i] = invgamma.rvs(a=alpha+1) * beta[i]
            outside_prior = x[i] > prior[0, i] or x[i] < prior[1, i]
            if outside_prior:
                # print("While loop started")
                prior_iter=0
                # Resample until we obtain a sample within the prior bounds
                # x_resample=[]
                while (x[i] > prior[0, i] or x[i] < prior[1, i]) and prior_iter<max_prior_iter:
                    x[i] = invgamma.rvs(a=alpha+1) * beta[i]
                    prior_iter+=1
                    # x_resample.append(x[i])
                # np.savetxt('x_resample',x_resample)

                if prior_iter>=max_prior_iter:
                    #DIAG: artificially inflating priors
                    prior[0,i]=1000
                    prior[1,i]=0
                    prior_iter=0
                    # print("Updating priors")
                    while (x[i] > prior[0, i] or x[i] < prior[1, i]) and prior_iter<max_prior_iter:
                        x[i] = invgamma.rvs(a=alpha+1) * beta[i]
                        prior_iter+=1
                    if prior_iter>=max_prior_iter:
                        raise ValueError("Number of prior resamples exceeded max_prior_iter")
        else:
            # print("Else triggered")
            x[i] = invgamma.rvs(a=alpha) * beta[i]
        # print("\n")

    return x


def sprior(signals, bins, factor):
=======
    data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)
    

    return data_fr_dly

'''ICDF sampler'''
def draw_icdf_samples(alpha, beta, x):
    
    """
    Draw a single sample from an inverse gamma distribution using inverse CDF sampling.

    This function performs inversion sampling from an inverse gamma distribution 
    defined by the shape parameter `alpha + 1` and scale parameter `beta`. The 
    sampling is constrained to the domain specified by `x`, and ensures that 
    the sample falls within this range by rescaling the CDF.

    Parameters
    ----------
    alpha : float
        Shape parameter (minus 1) of the inverse gamma distribution. 
        The full shape used is `alpha + 1` for consistency with certain prior formulations.
    
    beta : float
        Scale parameter of the inverse gamma distribution.

    x : ndarray
        A 1D array of values over which to compute the CDF and draw a sample.
        Should be sorted in increasing order and span the desired sampling range.

    Returns
    -------
    float
        A sample drawn from the inverse gamma distribution within the range defined by `x`.
    """
    
    cdf = invgamma.cdf(x, a=alpha+1, loc=0, scale=beta)
    cdf -= cdf.min() # shift minimum down to zero
    if cdf.max()==0.0:
        cdf /= (cdf.max()+1e-5) # rescale maximum to 1
    else:
        cdf /= cdf.max() # rescale maximum to 1

    # Remove duplicate entries in cdf so interpolator can work properly; 
    # tends to result in sample points near the extrema of the prior bounds anyway
    cdf_unique, idxs_unique = np.unique(cdf, return_index=True)
    u = np.random.uniform(high=cdf.max()) #High set to this value, otherwise the sample drawn is always out of the upper bound
    # Draw sample using inversion sampling method
    # Note: Must use linear interpolation to avoid very bad interpolation results
    return interp1d(cdf_unique, x[idxs_unique], kind='linear')(u)

def sample_pspec(s, prior, ngrid=120, sk=None,max_prior_iter=10000):
    """
    Draw a sample from an inverse gamma distribution using inversion 
    sampling between uniform prior bounds.
    
    This works by sampling the cdf of the inverse gamma distribution 
    on a (logarithmic) grid and then interpolating to convert a uniform 
    random draw into a random draw with the correct pdf.

    Parameters:
        alpha: (float)
            Inverse gamma alpha parameter.
        beta: (float)
            Inverse gamma beta (scale) parameter.
        ngrid: (int)
            Number of sample points to use for interpolator.

    Returns:
        sample: (float)
            Sample drawn from the inverse gamma distribution between the 
            specified prior bounds.
    """

    if sk is None:
        axes = (1,)
        sk = np.fft.fftshift(s, axes=axes)
        sk = np.fft.ifftn(sk, axes=axes) * np.sqrt(s.shape[1]) # note normalisation
        sk = np.fft.ifftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    
    alpha = Nobs-1
    beta = np.sum(sk * sk.conj(), axis=0).real # normalisation

    # Sample cdf logarithmically between provided prior bounds
    xgrid = np.logspace(np.log10(prior.min()), np.log10(prior.max()), ngrid)
    
    samples = np.zeros(Nfreqs)
    for i in range(Nfreqs):
        samples[i] = draw_icdf_samples(alpha, beta[i], xgrid)
    return samples


def sprior(signals, bins, factor):
    """
    Compute the prior on covariance samples based on the Fourier transform of the input signals.

    This function calculates a prior on the covariance of Fourier-transformed signals. The prior is defined
    by a range determined by a `factor` which scales the observed power spectrum, and only a specific number 
    of frequency bins around zero delay are retained, with others set to zero.

    Parameters
    ----------
    signals : numpy.ndarray
        A 2D array of shape (Nobs, Nfreq) where `Nobs` is the number of observations and `Nfreq` is the number
        of frequency channels. This array contains the observed signals to be transformed.

    bins : int
        The number of bins on either side of zero delay to retain in the prior. For example, `bins=2` will 
        retain the frequency bins corresponding to delays [-2, -1, 0, 1, 2].

    factor : float
        A scaling factor that defines the range of the prior. The upper bound of the prior is the observed 
        power spectrum multiplied by `factor`, and the lower bound is the observed power spectrum divided 
        by `factor`.

    Returns
    -------
    prior : numpy.ndarray
        A 2D array of shape (2, Nfreq) containing the prior bounds. The first row (`prior[0]`) contains 
        the upper bounds, and the second row (`prior[1]`) contains the lower bounds. Frequency bins outside 
        the specified range (determined by `bins`) are set to zero.
    """
>>>>>>> origin/multi_phil

    # prior on cov samples

    # bins - number of bins past zero delay to take, either side. e.g. bins=2 takes delays [-2,-1,0,1,2] from centre
    # factor is maximum factor to multiply / divide the truth by
    Nobs, Nfreq = signals.shape

    sk_ = np.fft.fft(signals, axis=-1)
    ds = np.sum(sk_ * sk_.conj(), axis=0).real
    prior = np.zeros((2, Nfreq))

    prior[0] = ds * factor
    prior[1] = ds / factor

    prior[0, bins + 1 : -bins] = 0
    prior[1, bins + 1 : -bins] = 0

    return prior / (Nobs / 2 - 1)


<<<<<<< HEAD
def gcr_fgmodes_1d(
    idx, vis, w, matrices, fgmodes, f0=None, map_estimate=False, verbose=False,
    multiprocess_seed=912983
):
    """
    Perform the GCR step on a single time sample.

    Parameters:
        idx (int):
            Time index.  Used to generate a unique random seed for each process
            if using `multiprocess.Pool` and multiple processes.
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        w (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        matrices (array_like):
            Array containing precomputed matrices needed by the linear system.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        verbose (bool):
            If True, output basic timing stats about each iteration.
        multiprocess_seed (int):
            Reference random seed used for all processes and time indices.
            Used to generate a unique random seed for each spawned process and
            each time index.  Defaults to 912983.

    """
    # If multiple process are spawned via `multiprocess.Pool`, each process
    # inherits the random seed of the parent process.  We need to set a unique
    # seed per process to avoid spurious correlations between GCRs at different
    # time indices.  We can do so using the process ID (PID, unique per
    # process) and time index (unique for each time).  For fewer than 1000
    # processes, we can guarantee a unique random seed by summing the
    # multiprocess_seed (a reference seed which is fixed for all processes and
    # times), the PID*1000, and the time index.
    # WARNING: if more than 1000 processes is every used this sum will not
    # guarantee a unique seed for each process!
    pid = current_process().pid
    seed = multiprocess_seed + pid*1000 + idx
    np.random.seed(seed)

    Nfreqs, Nmodes = fgmodes.shape
    d = vis.reshape((1, max(Nfreqs, len(vis.T))))

    # Extract precomputed matrices needed by the linear system
    Sh = matrices[0][0]
    S = matrices[0][1]
    Ni = matrices[0][2]
    Nih = matrices[0][3]
    A = matrices[1][0]
    Ai = matrices[1][1]

=======
def gcr_fg_and_signal_per_time(idx, 
                               vis, 
                               Einv, 
                               sqrtE, 
                               sqrtNinv, 
                               Nparams, 
                               sys_model, 
                               flags, 
                               Ninv, 
                               fg_modes, 
                               map_estimate=False, 
                               verbose=False,
                               multiprocess_seed=None,
                               solver='lgmres',
                               solver_tol=1e-12):
    """
    Solves the GCR equation for the joint foreground + signal model 
    for a single time
    
    Parameters:
        idx (int):
            Time index in the loop. Only used for setting the random seed 
            and debug output.
        vis (array_like):
            Visibility data being modelled (Ntimes, Nfreqs)
        Nparams (int):
            Number of model parameters.
        sys_model (array_like):
            Systematics gain model for this time index. Shape (Nfreqs,)
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
        Einv (array_like):
            Current value of the EoR signal frequency-frequency covariance inverse.
        sqrtE (array_like):
            Square-root of E matrix (Nfreqs, Nfreqs)
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        sqrtNinv (array_like):
            Square-root of Ninv, same shape as Ninv
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        solver_tol (float):
            Tolerance `tol` for scipy linear solvers.
        
    Returns:
        xsoln (array_like):
            Solution of the GCR for idx time index. First half is EoR solution, second half is foreground amplitudes. (2*Nfreqs, 1)

        residual (float):
            Residual |Axsoln-b|; indicates solution accuracy
        
        info (int):
            Info from the linear solver. Contains convergence information. 0 indicates success. 
    """
    # Set parallel-safe random seed
    pid = current_process().pid
    seed = None
    np.random.seed(seed)

    Nfreqs, Nmodes = fg_modes.shape
    d = vis.reshape((1, max(Nfreqs, len(vis.T))))  # Do NOT use order='F'

    # Construct necessary operators for GCR
    Ninv_sys = (sys_model.conj().T * Ninv.diagonal() *  sys_model)
    Ni_flagged = flags.T * Ninv_sys * flags  # Ninv with flags and systematics
    
    # Construct block operator matrix
    A = np.zeros((Nparams, Nparams), dtype=complex)
    
    # A_11: g^daggerdag E^-1 g + g^dagger * N^-1 * g
    A[:Nfreqs, :Nfreqs] = sys_model.conj()[:,np.newaxis] * Einv * sys_model[:,np.newaxis] \
                        + np.diag(Ni_flagged)
    
    # A_12: g^dagger * N^-1 * g * G
    A[:Nfreqs, Nfreqs:] = Ni_flagged[:,np.newaxis] * fg_modes
    
    # A_21: G^dagger * g^dagger * N^-1 * g
    A[Nfreqs:, :Nfreqs] = (fg_modes.conj() * Ni_flagged[:,np.newaxis]).T
    
    # A_22: G^dagger * g^dagger * N^-1 * g * G 
    A[Nfreqs:, Nfreqs:] = fg_modes.T.conj() @ (Ni_flagged[:,np.newaxis] * fg_modes)
    # Basic diagonal preconditioner
    Ainv_estimate = np.diag(1. / np.diag(A))
    #Ainv_estimate = np.linalg.pinv(A)

    # Construct fluctuation terms
>>>>>>> origin/multi_phil
    if map_estimate:
        oma = np.zeros((Nfreqs, 1), dtype=complex)
        omb = np.zeros((Nfreqs, 1), dtype=complex)
    else:
        # Unit complex Gaussian random realisation
<<<<<<< HEAD
        omi, omj = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs, 1)
        omk, oml = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs, 1)
        oma, omb = (omi + 1.0j * omj) / 2**0.5, (omk + 1.0j * oml) / 2**0.5

    # Construct RHS vector
    b = np.zeros((Nfreqs + Nmodes, 1), dtype=complex)
    b[:Nfreqs] = S @ Ni @ (w * d).T + Sh @ oma + S @ Nih @ omb
    b[Nfreqs:] = fgmodes.T.conj() @ (Ni @ (w * d).T + Nih @ omb)

    # Run CG solver, preconditioned by M=Ai
    x0 = None
    if f0 is not None:
        x0 = np.concatenate((np.zeros(Nfreqs, dtype=complex), f0))
    xsoln, info = sp.sparse.linalg.cg(A, b, maxiter=int(1e5), x0=x0, M=Ai)
    if verbose:
        residual = np.abs(A @ xsoln - b[:, 0]).mean()
=======
        omi, omj = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs,1)
        omk, oml = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs,1)
        oma, omb = (omi + 1.0j * omj) / 2**0.5, (omk + 1.0j * oml) / 2**0.5  
    
    # Construct RHS vector
    b = np.zeros((Nfreqs + Nmodes, 1), dtype=complex)
    b[:Nfreqs] = (sys_model.conj() * Ninv.diagonal() * d).T \
               +  sys_model.conj()[:,np.newaxis] * (sqrtE @ oma + sqrtNinv[:,np.newaxis] * omb)
    b[Nfreqs:] = fg_modes.T.conj() @ (
                     (sys_model.conj() * Ninv.diagonal() * d).T \
                   + (sys_model.conj()[:,np.newaxis] * sqrtNinv[:,np.newaxis] * omb) )
    
    # Run CG solver, preconditioned by M ~ A^-1
    x0 = None
    # xsoln, info = sp.sparse.linalg.cgs(A, b, x0=x0, M=Ainv_estimate, tol=solver_tol, maxiter=8000)
    xsoln, info = sp.sparse.linalg.gmres(A, b, x0=x0, M=Ainv_estimate, tol=solver_tol, maxiter=8000)
    
    # Check solution
    if info > 0:
        # Try again with different solver
        xsoln, info2 = sp.sparse.linalg.bicgstab(A, 
                                                 b, 
                                                 x0=x0, 
                                                 M=Ainv_estimate, 
                                                 tol=solver_tol, 
                                                 maxiter=8000)
        if info2 != 0:
            raise ValueError("GCR solver failed after retry; pid %d, time idx %d, info %d, info2 %d" \
                             % (pid, idx, info, info2))
    if info < 0:
        raise ValueError("GCR solver failed; pid %d, time idx %d, info %d" \
                         % (pid, idx, info))

    # Print residual if verbose mode enabled
    if verbose:
        residual = np.sqrt( np.sum(np.abs(A @ xsoln - b[:, 0])**2.) ) # residual = |Ax - b|
>>>>>>> origin/multi_phil
    else:
        residual = None

    # Return solution vector
    return xsoln, residual, info


<<<<<<< HEAD
def gcr_fgmodes(
    vis, w, matrices, fgmodes, f0=None, nproc=1, map_estimate=False,
    verbose=False
=======
def gcr_fg_and_signal(
    vis, 
    flags, 
    fg_modes, 
    Nparams, 
    sys_model, 
    signal_ps, 
    Ninv, 
    fourier_op,
    nproc=1, 
    map_estimate=False,
    solver='lgmres',
    solver_tol=1e-12,
    verbose=False,
>>>>>>> origin/multi_phil
):
    """
    Perform the GCR step on all time samples, using parallelisation if
    possible.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
<<<<<<< HEAD
        w (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        matrices (array_like):
            Array containing precomputed matrices needed by the linear system.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        fourier_op (array_like):
            Pre-computed Fourier operator.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
=======
        flags (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        signal_ps (array_like):
            Signal power spectrum.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        sys_model (array_like):
            Current multiplicative systematics model, of shape `(Ntimes, Nfreqs)`.
        fourier_op (array_like):
            Pre-computed Fourier operator.
>>>>>>> origin/multi_phil
        nproc (int):
            Number of processes to use for parallelised functions.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
<<<<<<< HEAD
=======
        solver_tol (float):
            Tolerance `tol` for scipy linear solvers.
>>>>>>> origin/multi_phil
        verbose (bool):
            If True, output basic timing stats about each iteration.

    Returns:
        samples (array_like):
            Array of signal + foreground realisations for each time sample,
            of shape `(Ntimes, Nfreqs + Nmodes)`.
    """
<<<<<<< HEAD
    samples = np.zeros((vis.shape[0], vis.shape[1] + fgmodes.shape[1]), dtype=complex)
    if verbose:
        residuals = np.zeros(vis.shape[0], dtype=float)
        info = np.zeros(vis.shape[0], dtype=float)
    else:
        residuals = None
        info = None
    idxs = np.arange(vis.shape[0])

    # Run GCR method on each time sample in parallel
    if verbose:
        st = time.time()
    with Pool(nproc) as pool:
        samples, residuals, info = zip(*pool.map(
            lambda idx: gcr_fgmodes_1d(
                idx=idx,
                vis=vis[idx],
                w=w,
                matrices=matrices,
                fgmodes=fgmodes,
                f0=f0,
                map_estimate=map_estimate,
                verbose=verbose
            ),
            idxs,
        ))
    samples = np.array(samples).reshape((vis.shape[0], -1))
=======
    # Set up samples array
    samples = np.zeros((vis.shape[0], vis.shape[1] + fg_modes.shape[1]), dtype=complex)
    
    # Prepare residuals/info arrays for each time
    residuals, info = None, None
    if verbose:
        residuals = np.zeros(vis.shape[0], dtype=float)
        info = np.zeros(vis.shape[0], dtype=float)
    
    # Time indices
    time_idxs = np.arange(vis.shape[0])
    
    # Pre-compute quantities that are constant in time
    E = covariance_from_pspec(signal_ps, fourier_op)
    Einv = covariance_from_pspec(1./signal_ps, fourier_op)
    sqrtE = sp.linalg.sqrtm(E) 
    sqrtNinv = np.sqrt(np.diag(Ninv))
    
    # Run GCR solver on each time sample in parallel
    if verbose:
        t_start = time.time()
    
    samples = []
    residuals = []
    info = []
    for idx in time_idxs:
        _s, _r, _i = gcr_fg_and_signal_per_time(
                idx=idx,
                vis=vis[idx],
                fg_modes=fg_modes,
                Nparams=Nparams,
                sys_model=sys_model[idx],
                flags=flags,
                Einv=Einv,
                sqrtE=sqrtE,
                Ninv=Ninv,
                sqrtNinv=sqrtNinv, 
                map_estimate=map_estimate,
                solver=solver,
                solver_tol=solver_tol,
                verbose=verbose,
                multiprocess_seed=100000
            )
        samples.append(_s)
        residuals.append(_r)
        info.append(_i)

    samples = np.array(samples).reshape((vis.shape[0], -1)) # Do NOT use order F
>>>>>>> origin/multi_phil
    residuals = np.array(residuals)
    info = np.array(info)

    # Return sample
    if verbose:
<<<<<<< HEAD
        print(f"{time.time() - st:<12.1f}", end="")
=======
        print(f"{time.time() - t_start:<12.4f}", end="")
>>>>>>> origin/multi_phil
        print(f"{info.mean():<8.1f}", end="")
        print(f"{residuals.mean():<12.2e}", end="")
    return samples


def covariance_from_pspec(ps, fourier_op):
    """
    Transform the sampled power spectrum into a frequency-frequency covariance
    matrix that can be used for the next iteration.
    """
    Nfreqs = ps.size
    Csigfft = np.zeros((Nfreqs, Nfreqs), dtype=complex)
    Csigfft[np.diag_indices(Nfreqs)] = ps
    C = (fourier_op.T.conj() @ Csigfft @ fourier_op)
    return C


<<<<<<< HEAD
def build_matrices(Nparams, flags, signal_S, Ninv, fgmodes):
    """
    Calculate matrices and build A in Ax=b for the GCR step.
    
    Parameters:
        Nparams (int):
            Number of model parameters.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
        signal_S (array_like):
            Current value of the EoR signal frequency-frequency covariance.
=======
def goodness_of_fit_statistics(data, 
                               data_model, 
                               flags, 
                               Ninv, 
                               signal_amps, 
                               Sinv, 
                               include_prior=False,
                               verbose=False):
    """
    Calculate the chi^2 and log-posterior for a given model.

    Parameters:
        data (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        data_model (array_like):
            Data model to be compared with `data` (must have same shape).
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
>>>>>>> origin/multi_phil
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
<<<<<<< HEAD
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
    
    Returns:
        matrices (list of array_like):
            List containing necessary GCR operators (`matrices[0]`) and the
            linear operator A in the GCR Ax=b solve step.
    """
    Nfreqs = signal_S.shape[0]
    
    # Construct matrix structure
    matrices = [0, 0]
    matrices[0] = np.zeros((4, Nfreqs, Nfreqs), dtype=complex)
    matrices[1] = np.zeros((2, Nparams, Nparams), dtype=complex)

    # Construct necessary operators for GCR
    matrices[0][0] = sp.linalg.sqrtm(signal_S)  # Sh
    matrices[0][1] = signal_S.copy()  # S
    matrices[0][2] = flags.T * Ninv * flags  # Ni # FIXME
    matrices[0][3] = sp.linalg.sqrtm(matrices[0][2])  # Nih

    # Construct operator matrix
    A = np.zeros((Nparams, Nparams), dtype=complex)
    A[:Nfreqs, :Nfreqs] = np.eye(Nfreqs) + matrices[0][1] @ matrices[0][2]  # 1 + S @ Ni
    A[:Nfreqs, Nfreqs:] = matrices[0][1] @ matrices[0][2] @ fgmodes
    A[Nfreqs:, :Nfreqs] = fgmodes.T.conj() @ matrices[0][2]
    A[Nfreqs:, Nfreqs:] = fgmodes.T.conj() @ matrices[0][2] @ fgmodes

    matrices[1][0] = A
    matrices[1][1] = np.linalg.pinv(A)  # pseudo-inverse, to be used as a preconditioner
    
    return matrices


def gibbs_step_fgmodes(
    vis,
    flags,
    signal_S,
    # b_sys,
    fgmodes,
    Ninv,
    nm_list,
    h_j,
    b_sys_past,
    freqs,
    lsts,
    B,
    ps_prior=None,
    f0=None,
    nproc=1,
    map_estimate=False,
    verbose=False
=======
        signal_amps (array_like):
            Signal amplitudes.
        Sinv (array_like):
            Signal covariance matrix (inverse).
        verbose (bool):
            Whether to output basic debug info.

    Returns:
        chisq (array_like):
            chi^2 value for each element on the data.

        ln_post (float):
            log posterior probability (unnormalised).
    """
    # Chi-squared is computed as the sum of ( |data - model - sys_model| / noise )^2,
    # i.e. as a sum of standard normal random variables.
    chisq = np.abs(data - data_model)**2 * Ninv.diagonal()[None, :]
    chisq_mean = chisq[:, flags].mean()
    chisq = chisq.real

    if verbose:
        chisq_mean = chisq[:, flags].mean()
        print(f"{chisq_mean:<9.3e}", end=" ")

    # Whether to include the prior term in ln_post
    use_prior = 0.
    if include_prior:
        use_prior = 1.

    # Log posterior; each time is treated as an independent sample, so the joint
    # ln_post for all times is the sum of the ones for each time.
    ln_post = np.sum(np.diagonal(
        -(
            (data - data_model)[:, flags].conj()
            @ Ninv[flags][:, flags]
            @ (data - data_model)[:, flags].T
        )
    ))
    ln_post = np.real(ln_post)
    if verbose:
        print(f"{ln_post:<12.1f}")
    return chisq, ln_post


def gibbs_step(
    vis,
    flags,
    Ninv,
    signal_ps,
    signal_ps_prior,
    fg_modes,
    sys_modes,
    sys_amps,
    sys_prior,
    iter,
    sky_model=None,
    nproc=1,
    sample_systematics=True,
    sample_eor_fg=True,
    sample_signal_ps=True,
    map_estimate=False,
    solver='lgmres',
    solver_tol=1e-12,
    verbose=True
>>>>>>> origin/multi_phil
):
    """
    Perform a single Gibbs iteration for a Gibbs sampling scheme using a foreground model
    based on frequency templates for multiple foreground modes.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
<<<<<<< HEAD
        signal_S (array_like):
            Current value of the EoR signal frequency-frequency covariance.
        b_sys (array_like):
            ...
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
=======
>>>>>>> origin/multi_phil
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
<<<<<<< HEAD
        ps_prior (array_like):
            EoR signal power spectrum prior.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):
            Number of processes to use for parallelised functions.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        verbose (bool):
            If True, output basic timing stats about each iteration.
        nm_list:
            List of n frequency modes and m LST modes corresponding to suspected presence of artefacts
        freqs:
            Array of frequencies of shape ()
        lsts:
            Array of frequencies of shape ()
        b_sys_p: 
            Present best estimate of the systematics vector of shape (len(nm_list))
        B:
            Systematics covariance matrix

    Returns:
        signal_cr (array_like):
            Samples of the signal, shape `(Ntimes, Nfreqs)`.
        S_sample (array_like):
            Sample of the signal covariance, shape `(Nfreqs, Nfreqs)`. This is
            simply a transformation of the power spectrum.
=======
        signal_ps (array_like):
            Current value of the EoR signal power spectrum.
        signal_ps_prior (array_like):
            EoR signal power spectrum prior.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        sys_modes (array_like):
            Systematics mode matrix.
        sys_amps (array_like):
            Systematics coefficients from the previous iteration. Shape `(Nsys_modes,)`.
        sys_prior (array_like):
            Systematic coefficient prior covariance matrix, of shape 
            `(Nsys_modes, Nsys_modes)`.
        iter (int):
            Nth Gibbs sampler iteration (for plotting)
        sky_model (array_like):
            Sky model to use if the signal + FG GCR sampling step is switched off. 
            Otherwise, it will be overwritten in the first conditional sampling step.
        nproc (int):
            Number of processes to use for parallelised functions.
        sample_systematics (bool):
            Whether to sample systematics model parameters or keep them fixed.
        sample_signal_ps (bool):
            Whether to sample the signal power spectrum.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        solver_tol (float):
            Tolerance `tol` for scipy linear solvers.
        verbose (bool):
            If True, output basic timing stats about each iteration.

    Returns:
        signal_amps (array_like):
            Samples of the signal, shape `(Ntimes, Nfreqs)`.
>>>>>>> origin/multi_phil
        ps_sample (array_like):
            Sample of the signal power spectrum bandpowers, shape `(Nfreqs,)`.
        fg_amps (array_like):
            Sample of the foreground amplitudes, shape `(Nmodes,)`.
<<<<<<< HEAD
        b_sys (array_like):
            Array of systematics amplitudes of shape (len(nm_list))
    """
    # print("Running gibbs_step_fgmodes")
    # Shape of data and operators
    Ntimes=vis.shape[0]
    Nfreqs = vis.shape[1] 
    Nmodes = fgmodes.shape[1]
    Nparams = Nfreqs + Nmodes
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"
    # print("init setup done")
    # Precompute 2D Fourier operator matrix
    fourier_op = utils.fourier_operator(Nfreqs)

    # Precompute h_j systematics projection operator
    # FIXME: Should put this outside this function and pass h_j in like we do with fgmodes; 
    # h_j = sys.h_j_op(freqs=freqs, lsts=lsts, nm_list=nm_list)

    # Get matrices necessary for the GCR step
    matrices = build_matrices(Nparams, flags, signal_S, Ninv, fgmodes)
    sys_model_past= h_j @ b_sys_past
    sys_model_past=np.reshape(sys_model_past,[Ntimes,Nfreqs])
    # print("Past sys model done")
    # (1) Solve GCR equation to get EoR signal and foreground amplitude realisations
    cr = gcr_fgmodes(
        vis=vis - sys_model_past, w=flags, matrices=matrices, fgmodes=fgmodes, f0=f0, nproc=nproc,
        map_estimate=map_estimate, verbose=verbose
    )
    #DIAG: saving the residuals for diagnostic test
    # np.savetxt('residuals',vis-sys_model_past)

    # print("GCR fgmodes solved")
    # Extract separate signal and FG parts from the solution
    signal_cr = cr[:, : -fgmodes.shape[1]]
    fg_amps = cr[:, -fgmodes.shape[1] :]
    
    # Full model of data is sum of EoR (GCR) + FG model
    model = signal_cr + fg_amps @ fgmodes.T  # np.einsum('ijk,lk->ijl', fg_amps, fgmodes)
    # print("Model made")
    # 1a. Solve GCR equation to obtain estimate of systematic component
    # pr.enable()
    b_sys = sys_sol.gcr_sys(vis=vis - model, Ninv=Ninv, B=B, nm_list=nm_list, h_j=h_j, times=lsts, freqs=freqs)
    # pr.disable()

    # ps = pstats.Stats(pr, stream=sys.stdout)
    # ps.strip_dirs()
    # ps.sort_stats('time').print_stats()
    # print("GCR sys done")
    # Update systematics model
    sys_model = h_j @ b_sys # Shape of flattened data
    sys_model= np.reshape(sys_model,[Ntimes,Nfreqs]) #Gives data-like model 

    # Chi-squared is computed as the sum of ( |data - model - sys_model| / noise )^2,
    # i.e. as a sum of standard normal random variables.
    # FIXME: this will need to be changed to account for time-dependent
    # flags (i.e. when we have a different N per time).
    chisq = np.abs(vis - model - sys_model)**2 * Ninv.diagonal()[None, :]
    if verbose:
        chisq_mean = chisq[:, flags].mean()
        if chisq_mean > 10:
            print(f"{chisq_mean:<9.1e}", end="")
        else:
            print(f"{chisq_mean:<9.3f}", end="")
    # (2) Sample EoR signal power spectrum (and also convert to equivalent
    # covariance matrix sample)
    # print("Signal_cr shape: ", signal_cr.shape," ps_prior shape ",ps_prior.shape)
    ps_sample = sample_S(s=signal_cr, prior=ps_prior)
    # The factor of 1/Nfreqs**2 here is an FFT normalization
    S_sample = covariance_from_pspec(ps_sample / Nfreqs**2, fourier_op)
    # Log posterior
    # Each time is treated as an independent sample.  So, the joint
    # log posterior for all times is the sum of the individual log
    # posteriors for each time.
    # WARNING: np.linalg.inv should be avoided for general, dense matrices.
    # S_sample should be diagonally dominant and thus this should be okay.
    Sinv = np.linalg.inv(S_sample)
    ln_post = np.sum(np.diagonal(
        -(
            (vis - model - sys_model)[:, flags].conj()
            @ Ninv[flags][:, flags]
            @ (vis - model - sys_model)[:, flags].T
        )
        - (
            signal_cr[:, flags].conj()
            @ Sinv[flags][:, flags]
            @ signal_cr[:, flags].T
        )
    ))
    # ln_post = ln_post.real
    ln_post=np.real(ln_post)
    if verbose:
        print(f"{ln_post:<12.1f}")
    # Return samples
    return signal_cr, S_sample, ps_sample, fg_amps, b_sys, chisq, ln_post 

def gibbs_sample_with_fg(
    vis,
    flags,
    S_initial,
    fgmodes,
    Ninv,
    ps_prior,
    freqs,
    lsts,
    nm_list,
    Niter=100,
    seed=None,
=======
        sys_amps (array_like):
            Array of systematics amplitudes of shape (len(nm_list))
    """
    # Shape of data and operators
    Ntimes = vis.shape[0]
    Nfreqs = vis.shape[1] 
    Nfg_modes = fg_modes.shape[1]
    Nparams = Nfreqs + Nfg_modes
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"

    # Precompute 2D Fourier operator matrix
    fourier_op = utils.fourier_operator(Nfreqs)

    # Precompute current systematics model
    # Note: Be very careful which order this is reshaped!
    sys_model = (1. + (sys_modes @ sys_amps).reshape((Nfreqs, Ntimes)).T)  # Do NOT use order F
    if sample_eor_fg:
        # (1) Sample signal and foreground amplitudes using GCR
        cr = gcr_fg_and_signal(
                        vis=vis, 
                        fg_modes=fg_modes, 
                        Nparams=Nparams, 
                        sys_model=sys_model, 
                        flags=flags, 
                        signal_ps=signal_ps, 
                        Ninv=Ninv,
                        fourier_op=fourier_op, 
                        nproc=nproc, 
                        map_estimate=map_estimate,
                        solver=solver,
                        solver_tol=solver_tol,
                        verbose=verbose)   #Running test on the d=(1+delta g)s+n form of the equations 
        
        # Extract separate signal and FG parts from the solution
        signal_amps = cr[:, :-Nfg_modes]
        fg_amps = cr[:, -Nfg_modes:]
        
        # Update sky model (without multiplicative systematics); sum of EoR + FG model
        sky_model = (signal_amps + fg_amps @ fg_modes.T)
    else:
        sky_model = sky_model

    # (2) Sample multiplicative systematics parameters
    if sample_systematics:
        sys_amps = sys_sol.gcr_systematics(
                                    data=vis,
                                    Ninv=Ninv,
                                    sky_model=sky_model, 
                                    sys_modes=sys_modes,
                                    sys_prior=sys_prior, 
                                    verbose=verbose
                                    )
    
    # (3) Sample EoR signal power spectrum (and also convert to signal covariance matrix)
    if sample_signal_ps:
        signal_ps_sample = sample_pspec(s=signal_amps, prior=signal_ps_prior)

        # No need for factor of 1/Nfreqs**2 here as sample_pspec() changed to iFFT normalization
        Sinv_sample = covariance_from_pspec(1. / signal_ps_sample, fourier_op) #/ Nfreqs**2. # note FFT norm
    else:
        signal_ps_sample = signal_ps
        Sinv_sample = 0.
        

    # Calculate goodness of fit statistics
    chisq, ln_post = goodness_of_fit_statistics(
                                    data=vis, 
                                    data_model=sys_model * sky_model, 
                                    flags=flags, 
                                    Ninv=Ninv, 
                                    signal_amps=signal_amps, 
                                    Sinv=Sinv_sample, 
                                    verbose=verbose)
    
    # Return samples
    return signal_amps, signal_ps_sample, fg_amps, sys_amps, chisq, ln_post 


def gibbs_sample(
    vis,
    flags,
    Ninv,
    freqs,
    lsts,
    signal_ps_initial,
    signal_ps_prior,
    fg_modes,
    sys_modes,
    sys_prior,
    sys_initial,
    sky_model_initial=None,
    Niter=100,
    seed=None,
    sample_systematics=True,
    sample_eor_fg=True,
    sample_signal_ps=True,
    solver='lgmres',
    solver_tol=1e-12,
>>>>>>> origin/multi_phil
    verbose=True,
    nproc=1,
    write_Niter=100,
    out_dir=None,
    map_estimate=False
):
    """
    Run a Gibbs chain on data for a single baseline, using a foreground model
    based on frequency templates for multiple foreground modes.

    This will return samples of EoR signal and foreground amplitude
    constrained realisations, and the signal frequency-frequency covariance
    and power spectrum.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
<<<<<<< HEAD
        S_initial (array_like):
            Initial guess for the EoR signal frequency-frequency covariance.
            A better guess should result in faster convergence.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
=======
>>>>>>> origin/multi_phil
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
<<<<<<< HEAD
        ps_prior (array_like):
            EoR signal power spectrum prior.
=======
        freqs:
            Frequency array (Nfreqs,)
        lsts:
            Time array in LSTS (Ntimes,)
        signal_ps_initial (array_like):
            Initial guess for the EoR signal power spectrum. A better guess 
            should result in faster convergence.
        signal_ps_prior (array_like):
            EoR signal power spectrum prior, or shape (2, Nfreqs). `ps_prior[0]` 
            contains the lower bound of the prior, `ps_prior[1]` the upper bound. 
        fg_modes (array_like):
            Foreground mode array, of shape `(Nfreqs, Nmodes)`. This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        sys_modes (array_like):
            Systematics mode array, of shape `(Nfreqs * Ntimes, Nsysmodes)`.
        sys_prior (array_like):
            Prior covariance for the systematic amplitudes, of shape 
            `(Nsysmodes, Nsysmodes)` .
        sys_initial (array_like):
            Initial guess of systematics parameters.
>>>>>>> origin/multi_phil
        Niter (int):
            Number of iterations of the sampler to run.
        seed (int):
            Random seed to use for random parts of the sampler.
<<<<<<< HEAD
=======
        solver_tol (float):
            Tolerance `tol` for scipy linear solvers.
>>>>>>> origin/multi_phil
        verbose (bool):
            If True, output basic timing stats about each iteration.
        nproc (int):
            Number of processes to use for parallelised functions.
        write_Niter (int):
            Number of iterations between output file writing.
        out_dir (str or Path):
            Directory where samples will be saved to disk.  If None (default),
            samples are not written to disk.
        map_estimate (bool):
            Provide the maximum a posteriori sample only, i.e. sets
            `Niter = 1`.
<<<<<<< HEAD
        freqs:
            Frequency array (shape: )
        lsts:
            Time array in LSTS (shape: )

    Returns:
        signal_cr (array_like):
            Samples of the signal, shape `(Niter, Ntimes, Nfreqs)`.
        signal_S (array_like):
            Samples of the signal covariance, shape `(Niter, Nfreqs, Nfreqs)`.
            These are simply transformations of the power spectrum.
=======
        

    Returns:
        signal_amps (array_like):
            Samples of the signal, shape `(Niter, Ntimes, Nfreqs)`.
>>>>>>> origin/multi_phil
        signal_ps (array_like):
            Sample of the signal power spectrum bandpowers, shape
            `(Niter, Nfreqs)`.
        fg_amps (array_like):
            Samples of the foreground amplitudes, shape `(Niter, Nmodes)`.
<<<<<<< HEAD
=======
        sys_amps (array_like):
            Sample of systematics coefficient vectors (Niter, number of systematics modes)
>>>>>>> origin/multi_phil
        chisq (array_like):
            Chi-squared value per iteration, shape `(Niter, Ntimes, Nfreqs)`.
        ln_post (array_like):
            Natural log of the posterior probability per iteration, shape
            `(Niter,)`.
<<<<<<< HEAD

    """
    # print("Gibbs sample with fg running")
=======
    """
>>>>>>> origin/multi_phil
    if map_estimate:
        Niter = 1
        write_Niter = 1
    else:
        # Set random seed
        np.random.seed(seed)

    # Get shape of data/foreground modes
    Ntimes, Nfreqs = vis.shape
<<<<<<< HEAD
    Nmodes = fgmodes.shape[1]
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"
    assert fgmodes.shape[0] == Nfreqs, "fgmodes must have shape (Nfreqs, Nmodes)"
=======
    Nmodes = fg_modes.shape[1]
    Nsys_modes = sys_modes.shape[-1]
    assert sys_prior.shape[0] == sys_prior.shape[1] \
        == sys_initial.shape[0] == sys_modes.shape[-1], \
        "sys_modes, sys_prior, and sys_initial must have the same number of modes"
    assert sys_modes.shape[0] == Ntimes * Nfreqs, \
        "sys_modes must have shape (Ntimes * Nfreqs, Nsysmodes)"
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"
    assert fg_modes.shape[0] == Nfreqs, "fgmodes must have shape (Nfreqs, Nmodes)"
    assert signal_ps_prior.shape == (2, Nfreqs), "ps_prior must have shape (2, Nfreqs)"
>>>>>>> origin/multi_phil
    if len(Ninv.shape) == 3:
        assert (
            Ninv.shape[0] == Ntimes
        ), "Ninv shape must be (Ntimes, Nfreqs, Nfreqs) or (Nfreqs, Nfreqs)"
<<<<<<< HEAD
    # print("Shapes saved")
    # Set up arrays for sampling
    signal_cr = np.zeros((Niter, Ntimes, Nfreqs), dtype=complex)
    signal_S = np.zeros((Niter, Nfreqs, Nfreqs))
    signal_ps = np.zeros((Niter, Nfreqs))
    fg_amps = np.zeros((Niter, Ntimes, Nmodes), dtype=complex)
    b_sys = np.zeros((Niter, len(nm_list)), dtype=complex)
    # Useful debugging statistics
    chisq = np.zeros((Niter, Ntimes, Nfreqs))
    ln_post = np.zeros(Niter)
    # print("Arrays set up")
    # Set initial value for signal_S
    signal_S = S_initial.copy()

    # Precompute h_j systematics projection operator
    h_j = sys_sol.h_j_op(freqs=freqs, lsts=lsts, nm_list=nm_list)
    # print("h_j operator created")
    # Loop over iterations
    if verbose:
        print("Iter     Time [s]    Info    |Ax - b|    Chisq    ln Post")
        print("-----    --------    ----    --------    -----    -------")

    for i in range(Niter):
        print("Iteration: ",i)
        if verbose:
            print(f"{i+1:<9d}", end="")
        if i==0:
            b_sys_past=np.loadtxt('b_sys_past',dtype=complex)
            b_sys_past=np.array([[b] for b in b_sys_past])
        else:
            b_sys_past=b_sys[i-1]
        B_cov=(b_sys_past**2)*np.eye(len(b_sys_past))
        print("B_cov done. Shape: ", B_cov.shape)
        # Do Gibbs iteration
        signal_cr[i], signal_S, signal_ps[i], fg_amps[i], b_sys[i], chisq[i], ln_post[i]\
            = gibbs_step_fgmodes(
                vis=vis * flags,
                flags=flags,
                signal_S=signal_S,
                fgmodes=fgmodes,
                Ninv=Ninv,
                freqs=freqs,
                lsts=lsts,
                nm_list=nm_list,
                B=B_cov,
                h_j=h_j,
                b_sys_past=b_sys_past,
                ps_prior=ps_prior,
                f0=None,
                nproc=nproc,
                map_estimate=map_estimate,
                verbose=verbose
            )
        print("Iter: ",i," done")
=======
    
    # Check for sensible initial power spectrum
    assert np.all( np.logical_and(signal_ps_initial >= signal_ps_prior[0,:],
                                  signal_ps_initial <= signal_ps_prior[1,:]) ), \
           "Initial power spectrum ps_initial is not within ps_prior range."

    # Set up arrays for sampling
    signal_amps = np.zeros((Niter, Ntimes, Nfreqs), dtype=complex)
    signal_ps = np.zeros((Niter, Nfreqs))
    fg_amps = np.zeros((Niter, Ntimes, Nmodes), dtype=complex)
    sys_amps = np.zeros((Niter, Nsys_modes), dtype=complex)
    
    # Debugging statistics
    chisq = np.zeros((Niter, Ntimes, Nfreqs))
    ln_post = np.zeros(Niter)
    
    # Set initial values the signal power spectrum and systematics amplitudes
    signal_ps_current = signal_ps_initial
    sys_amps_current = sys_initial

    # Loop over iterations
    if verbose:
        print("Iter     Time [s]    Info    |Ax - b|    T_Sys(s)    Sys Info    Sys |Ax-b|    Chisq    ln Post")
        print("-----    --------    ----    --------    --------    --------    ----------    -----    -------")

    for i in range(Niter):
        if verbose:
            print(f"{i+1:<9d}", end="")

        # Do Gibbs iteration
        signal_amps[i], signal_ps[i], fg_amps[i], sys_amps[i], chisq[i], ln_post[i] \
            = gibbs_step(
                vis=vis * flags,
                flags=flags,
                Ninv=Ninv,
                signal_ps=signal_ps_current,
                signal_ps_prior=signal_ps_prior,
                fg_modes=fg_modes,
                sys_prior=sys_prior,
                sys_modes=sys_modes,
                sys_amps=sys_amps_current,
                sky_model=sky_model_initial,
                nproc=nproc,
                iter=i,
                map_estimate=map_estimate,
                solver=solver,
                solver_tol=solver_tol,
                sample_systematics=sample_systematics,
                sample_eor_fg=sample_eor_fg,
                sample_signal_ps=sample_signal_ps,
                verbose=verbose
            )

        # Update signal PS and systematics
        signal_ps_current = signal_ps[i]
        sys_amps_current = sys_amps[i]
        utils.append_gibbs_sample_h5(
            fp=out_dir,
            overwrite=(i == 0),          # truncate on the very first call
            signal_amps=signal_amps[i],
            signal_ps=signal_ps[i],
            fg_amps=fg_amps[i],
            sys_amps=sys_amps[i],
            chisq=chisq[i],
            ln_post=ln_post[i] # scalar is fine
        )
        
>>>>>>> origin/multi_phil
        if out_dir is not None and (i+1) % write_Niter == 0:
            # Write current set of samples to disk
            utils.write_numpy_files(
                out_dir,
<<<<<<< HEAD
                signal_cr[:i+1],
                signal_S[:i+1],
                signal_ps[:i+1],
                fg_amps[:i+1],
                b_sys[:i+1],
                chisq[:i+1],
                ln_post[:i+1]
            )
        print("Data saved for ",i)
=======
                signal_amps[:i+1],
                signal_ps[:i+1],
                fg_amps[:i+1],
                sys_amps[:i+1],
                chisq[:i+1],
                ln_post[:i+1]
            )
>>>>>>> origin/multi_phil
    if out_dir is not None and Niter % write_Niter > 0:
        # Write all samples to disk
        utils.write_numpy_files(
            out_dir,
<<<<<<< HEAD
            signal_cr,
            signal_S,
            signal_ps,
            fg_amps,
            b_sys,
=======
            signal_amps,
            signal_ps,
            fg_amps,
            sys_amps,
>>>>>>> origin/multi_phil
            chisq,
            ln_post
        )

    if verbose:
        print()

<<<<<<< HEAD
    return signal_cr, signal_S, signal_ps, fg_amps, b_sys, chisq, ln_post
=======
    return signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post
>>>>>>> origin/multi_phil
