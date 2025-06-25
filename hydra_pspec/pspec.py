import numpy as np
import scipy as sp
from scipy.signal.windows import blackmanharris as BH
from scipy.stats import invgamma
from scipy.interpolate import interp1d
import scipy.linalg
from . import sys_solver as sys_sol
from multiprocess import Pool, current_process
from . import utils
import os, time
import cProfile
import pstats
import sys
import uvtools
from uvtools.dspec import gen_window
from uvtools.utils import FFT
from pyuvdata import UVData
from tqdm import tqdm  #For progress bars
from .plotting_functions import master_plotter #For plotting iterations
uvd=UVData() #Loading uvh5 files
pr=cProfile.Profile() #For profiling

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
    
    #prior_min = prior[1,57]
    #prior_max=prior[0,57]
    
    alpha = Nobs-1
    beta = np.sum(sk * sk.conj(), axis=0).real # normalisation

    # Sample cdf logarithmically between provided prior bounds
    xgrid = np.logspace(np.log10(prior.min()), np.log10(prior.max()), ngrid) #FIXME: the prior min is 0, can't have that. 
    
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


def gcr_fgmodes_1d(idx, 
                      vis, 
                      Einv, 
                      sqrtE, 
                      sqrtNinv, 
                      Nparams, 
                      y, 
                      flags, 
                      Ninv, 
                      fgmodes, 
                      f0=None, 
                      map_estimate=False, 
                      verbose=False,
                      multiprocess_seed=912983):
    """
    Solves the GCR equation on a time by time basis. 
    Pass samples for each sample from visibilities and systematics gain to this function. 
    Returns a solution that's a 1D array with shape (2*Nfreqs,)
    
    Parameters:
        idx: int
            Time index in the loop
        vis: array_like
            Visibility data being modelled (Ntimes, Nfreqs)
        Nparams: int
            Number of model parameters.
        y: array_like
            Systematics gain for idx time index (Nfreqs,)
        flags: array_like
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
        Einv: array_like
            Current value of the EoR signal frequency-frequency covariance inverse.
        sqrtE: array_like
            Square-root of E matrix (Nfreqs, Nfreqs)
        Ninv: array_like
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        sqrtNinv: array_like
            Square-root of Ninv, same shape as Ninv
        fgmodes:
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        
    Returns:

        xsoln: array_like
            Solution of the GCR for idx time index. First half is EoR solution, second half is foreground amplitudes. (2*Nfreqs, 1)

        residual: float
            Residual |Axsoln-b|; indicates solution accuracy
        
        info:
            Info from the cgs solver. Contains convergence information. 0 indicates success. 
    """
    # Set parallel-safe random seed
    pid = current_process().pid
    seed = multiprocess_seed + pid*1000 + idx
    np.random.seed(seed)

    Nfreqs, Nmodes = fgmodes.shape
    d = vis.reshape((1, max(Nfreqs, len(vis.T))))

    # Construct necessary operators for GCR
    inner_prod = (y.conj().T * Ninv.diagonal() *  y)
    Ni_flagged = flags.T * (inner_prod) * flags  # Ni with flags and systematics sandwich
    
    # Construct block operator matrix
    A = np.zeros((Nparams, Nparams), dtype=complex)
    A[:Nfreqs, :Nfreqs] = y.conj()[:,np.newaxis] * Einv * y[:,np.newaxis] + np.diag(Ni_flagged)  # A11: y.dag E^-1 y + y.dag * Ni * y
    A[:Nfreqs, Nfreqs:] = Ni_flagged[:,np.newaxis] * fgmodes # A12: y.dag * Ni * y * G
    A[Nfreqs:, :Nfreqs] = (fgmodes.conj() * Ni_flagged[:,np.newaxis]).T # A21: G.dag * y.dag * Ni * y
    A[Nfreqs:, Nfreqs:] = fgmodes.T.conj() @ (Ni_flagged[:,np.newaxis] * fgmodes) # A22: G.dag * y.dag * Ni * y * G 
    
    # Basic diagonal preconditioner
    Ainv_estimate = np.diag(1. / np.diag(A))
    #Ainv_estimate = np.linalg.pinv(A)

    # Construct fluctuation terms
    if map_estimate:
        oma = np.zeros((Nfreqs, 1), dtype=complex)
        omb = np.zeros((Nfreqs, 1), dtype=complex)
    else:
        # Unit complex Gaussian random realisation
        omi, omj = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs,1)
        omk, oml = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs,1)
        oma, omb = (omi + 1.0j * omj) / 2**0.5, (omk + 1.0j * oml) / 2**0.5  
    
    # Construct RHS vector
    b = np.zeros((Nfreqs + Nmodes, 1), dtype=complex)
    b[:Nfreqs] = (y.conj() * Ninv.diagonal() * d).T \
               +  y.conj()[:,np.newaxis] * (sqrtE @ oma + sqrtNinv[:,np.newaxis] * omb)
    b[Nfreqs:] = fgmodes.T.conj() @ (
                     (y.conj() * Ninv.diagonal() * d).T \
                   + (y.conj()[:,np.newaxis] * sqrtNinv[:,np.newaxis] * omb) )
    
    # Run CG solver, preconditioned by M=Ai
    x0 = None
    if f0 is not None:
        x0 = np.concatenate((np.zeros(Nparams, dtype=complex), f0))
    
    xsoln, info = sp.sparse.linalg.cgs(A, b, x0=x0, M=Ainv_estimate, tol=1e-12) #maxiter=int(1e5) , rtol=1e-12 , x0=x0, M=Ai
    
    # Check solution
    if info > 0:
        # Try again with different solver
        xsoln, info = sp.sparse.linalg.bicgstab(A, b, x0=x0, M=Ainv_estimate, tol=1e-12)
        if info != 0:
            raise ValueError("GCR solver failed after retry; pid %d, time idx %d, info %d" \
                             % (pid, idx, info))
    if info < 0:
        raise ValueError("GCR solver failed; pid %d, time idx %d, info %d" \
                         % (pid, idx, info))

    # Print residual if verbose mode enabled
    if verbose:
        residual = np.sqrt( np.sum(np.abs(A @ xsoln - b[:, 0])**2.) ) # residual = |Ax - b|
    else:
        residual = None

    # Return solution vector
    return xsoln, residual, info


"""
def build_matrices(Nparams, y, flags, E, Ninv, fgmodes):
    \"""
    OBSOLETE
    Calculate matrices and build A in Ax=b for the GCR step.
    
    Parameters:
        Nparams (int):
            Number of model parameters.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
        E (array_like):
            Current value of the EoR signal frequency-frequency covariance.
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        y (array_like):
            Data-like systematics gain for this iteration of sampler. (Ntimes, Nfreqs)
    
    Returns:
        A (array_like):
            A matrix for the GCR equation (2*Nfreqs,2*Nfreqs)
        Ni_flagged: array_like
            Ninv with flags (Nfreqs,Nfreqs)
        A_pinv:
            Pseudo inverse of A used as a preconditioner (same as A)
    \"""
    Nfreqs = E.shape[0]
    #print("1")

    # FIXME: Since E is a FT matrix, we should be able to invert by inverting the power spectrum
    
    E_inv=np.linalg.pinv(E) #FIXME: Use a different inversion method [was inv()]




    #print("2")
    # Construct necessary operators for GCR
    inner_prod= (y.conj().T * Ninv.diagonal() *  y)
    Ni_flagged = flags.T * (inner_prod) * flags  # Ni with flags and systematics sandwich
    
    #print("3")
    # Construct operator matrix
    A = np.zeros((Nparams, Nparams), dtype=complex)
    A[:Nfreqs, :Nfreqs] = y.conj()[:,np.newaxis] * E_inv * y[:,np.newaxis] + np.diag(Ni_flagged)  # A11: y.dag E^-1 y + y.dag * Ni * y
    A[:Nfreqs, Nfreqs:] = Ni_flagged[:,np.newaxis] * fgmodes # A12: y.dag * Ni * y * G
    A[Nfreqs:, :Nfreqs] = (fgmodes.conj() * Ni_flagged[:,np.newaxis]).T #A21: G.dag * y.dag * Ni * y
    A[Nfreqs:, Nfreqs:] = fgmodes.T.conj() @ (Ni_flagged[:,np.newaxis] * fgmodes) #A22: G.dag * y.dag * Ni * y * G 

    
    A_pinv = np.diag(1./np.diag(A)) # very basic diagonal inverse estimate
    #A_pinv = scipy.linalg.pinv(A)  # pseudo-inverse, to be used as a preconditioner
    #print("5")

    return A, Ni_flagged, A_pinv
"""


'''GCR equation: Time loop'''
def gcr_fgmodes(
    vis, 
    flags, 
    fgmodes, 
    Nparams, 
    sys_model_past, 
    signal_ps, 
    Ninv, 
    fourier_op,
    f0=None, 
    nproc=1, 
    map_estimate=False,
    verbose=False
):
    """
    Perform the GCR step on all time samples, using parallelisation if
    possible.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        matrices (array_like):
            Array containing precomputed matrices needed by the linear system.
        signal_ps (array_like):
            xxx
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        fourier_op (array_like):
            Pre-computed Fourier operator.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):
            Number of processes to use for parallelised functions.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        verbose (bool):
            If True, output basic timing stats about each iteration.

    Returns:
        samples (array_like):
            Array of signal + foreground realisations for each time sample,
            of shape `(Ntimes, Nfreqs + Nmodes)`.
    """
    samples = np.zeros((vis.shape[0], vis.shape[1] + fgmodes.shape[1]), dtype=complex)
    if verbose:
        residuals = np.zeros(vis.shape[0], dtype=float)
        info = np.zeros(vis.shape[0], dtype=float)
    else:
        residuals = None
        info = None
    idxs = np.arange(vis.shape[0])
    
    # Pre-compute quantities that are constant in time
    E = covariance_from_pspec(signal_ps, fourier_op)
    Einv = covariance_from_pspec(1./signal_ps, fourier_op)
    sqrtE = sp.linalg.sqrtm(E) 
    sqrtNinv = np.sqrt(np.diag(Ninv))
    
    # Run GCR method on each time sample in parallel
    if verbose:
        st = time.time()
    with Pool(nproc) as pool:
        samples, residuals,  info = zip(*pool.map(
            lambda idx: gcr_fgmodes_1d(
                idx=idx,
                vis=vis[idx],
                fgmodes=fgmodes,
                f0=f0,
                Nparams=Nparams,
                y=sys_model_past[idx],
                flags=flags,
                Einv=Einv,
                sqrtE=sqrtE,
                Ninv=Ninv,
                sqrtNinv=sqrtNinv, 
                map_estimate=map_estimate,
                verbose=verbose
            ),
            idxs,
        )
        )
    samples = np.array(samples).reshape((vis.shape[0], -1))
    residuals = np.array(residuals)
    info = np.array(info)

    # Return sample
    if verbose:
        print(f"{time.time() - st:<12.1f}", end="")
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



def gibbs_step_fgmodes(
    vis,
    flags,
    signal_ps,
    fgmodes,
    Ninv,
    sys_modes,
    b_sys_past,
    sys_model_past,
    Bi,
    iter,
    ps_prior=None,
    f0=None,
    nproc=1,
    map_estimate=False,
    verbose=True
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
        signal_ps (array_like):
            Current value of the EoR signal power spectrum.
        sys_modes (array_like):
            Systematics mode matrix.
        b_sys_past (array_like):
            Systematics coefficients from the last iteration of shape (number of systematics modes,)
        sys_model_past (array_like):
            Systematics model from the last iteration (1+H@b_sys_past)
        Bi (array_like):
            Inverse of systematic covarience matrix (number of systematics modes,number of systematics modes)
        iter (int):
            Nth Gibbs sampler iteration (for plotting)
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
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

    Returns:
        signal_cr (array_like):
            Samples of the signal, shape `(Ntimes, Nfreqs)`.
        ps_sample (array_like):
            Sample of the signal power spectrum bandpowers, shape `(Nfreqs,)`.
        fg_amps (array_like):
            Sample of the foreground amplitudes, shape `(Nmodes,)`.
        b_sys (array_like):
            Array of systematics amplitudes of shape (len(nm_list))
    """
    # Shape of data and operators
    Ntimes = vis.shape[0]
    Nfreqs = vis.shape[1] 
    Nmodes = fgmodes.shape[1]
    Nparams = Nfreqs + Nmodes
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"

    # Precompute 2D Fourier operator matrix
    fourier_op = utils.fourier_operator(Nfreqs)
    
    # (1) Solve GCR equation to get EoR signal and foreground amplitude realisations
    cr = gcr_fgmodes(
                    vis=vis, 
                    fgmodes=fgmodes, 
                    Nparams=Nparams, 
                    sys_model_past=sys_model_past, 
                    flags=flags, 
                    signal_ps=signal_ps, 
                    Ninv=Ninv,
                    fourier_op=fourier_op, 
                    f0=f0, 
                    nproc=nproc, 
                    map_estimate=map_estimate,
                    verbose=verbose)   #Running test on the d=(1+delta g)s+n form of the equations 
    
    # Extract separate signal and FG parts from the solution
    signal_cr = cr[:, : -fgmodes.shape[1]]
    fg_amps = cr[:, -fgmodes.shape[1] :]
    
    # Sky model of data is sum of EoR (GCR) + FG model
    model = (signal_cr + fg_amps @ fgmodes.T)  # np.einsum('ijk,lk->ijl', fg_amps, fgmodes) # sky model

    
    """
    b_sys=sys_sol.gcr_sys_v1(Binv=Bi,d=vis-clean_vis,Ninv=Ninv,s=clean_vis,H=h_j,b_sys_past=b_sys_past,verbose=verbose,iter=iter)

    # Update systematics model
    # sys_model = h_j @ b_sys #Shape of flattened data:  delta_g
    # sys_model= np.reshape(sys_model,[Ntimes,Nfreqs],order='F') #Gives data-like model: delta_g
    # sys_model += np.ones_like(sys_model, dtype='complex')  #Adding ones to the systematics solution: gain=(1+delta_g)
    """

    sys_model = sys_model_past
    
    model = (sys_model * model)
    # Chi-squared is computed as the sum of ( |data - model - sys_model| / noise )^2,
    # i.e. as a sum of standard normal random variables.
    # FIXME: this will need to be changed to account for time-dependent
    # flags (i.e. when we have a different N per time).
    chisq = np.abs(vis - model)**2 * Ninv.diagonal()[None, :]
    chisq_mean = chisq[:, flags].mean()
    chisq = chisq.real

    if verbose:
        chisq_mean = chisq[:, flags].mean()
        print(f"{chisq_mean:<9.1e}", end=" ")
    
    # FIXME
    b_sys = b_sys_past

    # (2) Sample EoR signal power spectrum (and also convert to equivalent
    ps_sample = sample_pspec(s=signal_cr, prior=ps_prior)

    # No need for factor of 1/Nfreqs**2 here as sample_pspec() changed to iFFT normalization
    Sinv_sample = covariance_from_pspec(1. / ps_sample, fourier_op) #/ Nfreqs**2. # note FFT norm

    # Log posterior; each time is treated as an independent sample, so the joint
    # ln_post for all times is the sum of the ones for each time.
    ln_post = np.sum(np.diagonal(
        -(
            (vis - (model))[:, flags].conj()
            @ Ninv[flags][:, flags]
            @ (vis - (model))[:, flags].T
        )
        - (
            signal_cr[:, flags].conj()
            @ Sinv_sample[flags][:, flags]
            @ signal_cr[:, flags].T
        )
    ))
    ln_post = np.real(ln_post)
    if verbose:
        print(f"{ln_post:<12.1f}")
    
    # Return samples
    return signal_cr, ps_sample, fg_amps, b_sys, chisq, ln_post 


def gibbs_sample_with_fg(
    vis,
    flags,
    ps_initial,
    fgmodes,
    Ninv,
    ps_prior,
    freqs,
    lsts,
    sys_modes,
    bsys_initial,
    Niter=100,
    seed=None,
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
        ps_initial (array_like):
            Initial guess for the EoR signal power spectrum. A better guess 
            should result in faster convergence.
        sys_modes (array_like):
            Systematics mode array, of shape `(Nfreqs * Ntimes, Nsysmodes)`.
        fgmodes (array_like):
            Foreground mode array, of shape `(Nfreqs, Nmodes)`. This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        bsys_initial (array_like):
            Initial guess of systematics parameters.
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        ps_prior (array_like):
            EoR signal power spectrum prior, or shape (2, Nfreqs). `ps_prior[0]` 
            contains the lower bound of the prior, `ps_prior[1]` the upper bound. 
        Niter (int):
            Number of iterations of the sampler to run.
        seed (int):
            Random seed to use for random parts of the sampler.
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
        freqs:
            Frequency array (Nfreqs,)
        lsts:
            Time array in LSTS (Ntimes,)

    Returns:
        signal_cr (array_like):
            Samples of the signal, shape `(Niter, Ntimes, Nfreqs)`.
        signal_ps (array_like):
            Sample of the signal power spectrum bandpowers, shape
            `(Niter, Nfreqs)`.
        fg_amps (array_like):
            Samples of the foreground amplitudes, shape `(Niter, Nmodes)`.
        b_sys (array_like):
            Sample of systematics coefficient vectors (Niter, number of systematics modes)
        chisq (array_like):
            Chi-squared value per iteration, shape `(Niter, Ntimes, Nfreqs)`.
        ln_post (array_like):
            Natural log of the posterior probability per iteration, shape
            `(Niter,)`.

    """
    if map_estimate:
        Niter = 1
        write_Niter = 1
    else:
        # Set random seed
        np.random.seed(seed)

    # Get shape of data/foreground modes
    Ntimes, Nfreqs = vis.shape
    Nmodes = fgmodes.shape[1]
    Nsys_modes = sys_modes.shape[-1]
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"
    assert fgmodes.shape[0] == Nfreqs, "fgmodes must have shape (Nfreqs, Nmodes)"
    assert ps_prior.shape == (2, Nfreqs), "ps_prior must have shape (2, Nfreqs)"
    if len(Ninv.shape) == 3:
        assert (
            Ninv.shape[0] == Ntimes
        ), "Ninv shape must be (Ntimes, Nfreqs, Nfreqs) or (Nfreqs, Nfreqs)"
    
    # Check for sensible initial power spectrum
    assert np.all( np.logical_and(ps_initial >= ps_prior[0,:],
                                  ps_initial <= ps_prior[1,:]) ), \
           "Initial power spectrum ps_initial is not within ps_prior range."

    # Set up arrays for sampling
    signal_cr = np.zeros((Niter, Ntimes, Nfreqs), dtype=complex)
    signal_S = np.zeros((Niter, Nfreqs, Nfreqs))
    signal_ps = np.zeros((Niter, Nfreqs))
    fg_amps = np.zeros((Niter, Ntimes, Nmodes), dtype=complex)
    b_sys = np.zeros((Niter, Nsys_modes), dtype=complex)
    
    # Useful debugging statistics
    chisq = np.zeros((Niter, Ntimes, Nfreqs))
    ln_post = np.zeros(Niter)
    
    # Set initial value for signal_S
    #signal_S = S_initial.copy()
    signal_ps_current = ps_initial

    # Precompute h_j systematics projection operator
    # FIXME: include path as arg in import file
    #nm_list_select = np.loadtxt('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/nm_list_select')
    #h_j = sys_sol.h_j_op(freqs=freqs,lsts=lsts,nm_list=nm_list) #nm_list containing dl fr values
    # Loop over iterations
    if verbose:
        print("Iter     Time [s]    Info    |Ax - b|    T_Sys(s)    Sys Info    Sys |Ax-b|    Chisq    ln Post")
        print("-----    --------    ----    --------    --------    --------    ----------    -----    -------")

    for i in range(Niter):
        if verbose:
            print(f"{i+1:<9d}", end="")
        """
        if i==0:
            #FIXME: loading files in loop, find a better way
            vis = np.load('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/dummy_sky.npy',allow_pickle=False) #FIXME: this is a test code. Remove once done. 
            clean_vis=np.load('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/dummy_sky.npy',allow_pickle=False)

            # FIXME: include path as arg in import file
            # uvd=UVData()
            # uvd.read('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_data/vis-eor-fgs.uvh5')
            # uvd=utils.form_pseudo_stokes_vis(uvd)
            # antpairpols = uvd.get_antpairpols()
            # clean_vis=uvd.get_data(antpairpols[0], force_copy=True)
            sys_model_past= (vis/clean_vis) #FIXME: initialising systematics. Find a better way
            b_sys_past = np.ones(h_j.shape[1],dtype='complex') #Starting from 0 works best, DO NOT CHANGE. 
            
        else:
            b_sys_past=b_sys[i-1]
            sys_model_past = (h_j @ b_sys_past).reshape([Ntimes,Nfreqs],order='F')
            sys_model_past = np.ones_like(sys_model_past,dtype='complex')+sys_model_past # Implementing the 1+del g model. 
        # B_cov_inv=np.sqrt(b_sys_past)*np.eye(len(b_sys_past)) #Uncomment this to sample for B
        """
        if i > 0:
            b_sys_past = b_sys[i-1]
        else:
            b_sys_past = bsys_initial #np.zeros(Nsys_modes)
        B_cov_inv = np.eye(Nsys_modes) # B is an identity matrix 
        
        # Calculate systematics model
        sys_model_past = 1. + (sys_modes @ b_sys_past).reshape((Nfreqs, Ntimes)).T

        # Do Gibbs iteration
        signal_cr[i], signal_ps[i], fg_amps[i], b_sys[i], chisq[i], ln_post[i]\
            = gibbs_step_fgmodes(
                vis=vis * flags,
                flags=flags,
                signal_ps=signal_ps_current,
                fgmodes=fgmodes,
                Ninv=Ninv,
                Bi=B_cov_inv,
                sys_modes=sys_modes,
                b_sys_past=b_sys_past,
                sys_model_past=sys_model_past,
                ps_prior=ps_prior,
                f0=None,
                nproc=nproc,
                iter=i,
                map_estimate=map_estimate,
                verbose=verbose
            )
        signal_ps_current = signal_ps[i] # update current PS state
        
        """
        if out_dir is not None and (i+1) % write_Niter == 0:
            # Write current set of samples to disk
            utils.write_numpy_files(
                out_dir,
                signal_cr[:i+1],
                signal_S[:i+1],
                signal_ps[:i+1],
                fg_amps[:i+1],
                b_sys[:i+1],
                chisq[:i+1],
                ln_post[:i+1]
            )
        """
    """
    if out_dir is not None and Niter % write_Niter > 0:
        # Write all samples to disk
        utils.write_numpy_files(
            out_dir,
            signal_cr,
            signal_S,
            signal_ps,
            fg_amps,
            b_sys,
            chisq,
            ln_post
        )
    """

    if verbose:
        print()

    return signal_cr, signal_ps, fg_amps, b_sys, chisq, ln_post
