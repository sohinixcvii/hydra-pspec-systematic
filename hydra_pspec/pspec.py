import numpy as np
import scipy as sp
from scipy.stats import mode
from scipy.signal.windows import blackmanharris as BH
from scipy.stats import invgamma
from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d
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
from tqdm import tqdm
from .plotting_functions import master_plotter
uvd=UVData()
pr=cProfile.Profile()

sample_c = 0
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
    
    # if beta==0: #Robustness line for degenerate CDFs, causes issues in interpolation otherwise
    #     return 0
    
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

def sample_S(s, prior, ngrid=120, sk=None,max_prior_iter=10000):
    """
    Draw a sample from an inverse gamma distribution using inversion 
    sampling between uniform prior bounds.
    
    This works by sampling the cdf of the inverse gamma distribution 
    on a (logarithmic) grid and then interpolating to convert a uniform 
    random draw into a random draw with the correct pdf.

    Parameters:
        alpha (float):
            Inverse gamma alpha parameter.
        beta (float):
            Inverse gamma beta (scale) parameter.
        prior_min (float):
            Minimum of the prior range.
        prior_max (float):
            Maximum of the prior range.
        ngrid (int):
            Number of sample points to use for interpolator.

    Returns:
        sample (float):
            Sample drawn from the inverse gamma distribution between the 
            specified prior bounds.
    """

    if sk is None:
        axes = (1,)
        sk = np.fft.ifftshift(s, axes=axes)
        sk = np.fft.fftn(sk, axes=axes)
        sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    
    prior_min = prior[1,57]
    prior_max=prior[0,57]
    
    alpha=Nobs-1
    beta = np.sum(sk * sk.conj(), axis=0).real

    # Sample cdf logarithmically between provided prior bounds
    x = np.logspace(np.log10(prior_min + 0.00001), np.log10(prior_max), ngrid) #FIXME: the prior min is 0, can't have that. 
    for i in range(Nfreqs):
        if np.any(prior[:, i] > 0):

            # The pdf for a log-uniform prior is proportional to 1 / x.
            # Multiplying the inverse gamma likelihood by this prior results
            # in an additional factor of 1 / x which increases the effective
            # value of the shape parameter (alpha) by 1.  With a log-uniform
            # prior, we thus sample from an inverse gamma distribution with
            # shape parameter alpha + 1.
            prior[1,i] = prior[1,i]+0.000001 #FIXME: priors should be fixed outside of this loop 

            x[i] = draw_icdf_samples(alpha,beta[i], x)
            outside_prior = x[i] > prior[0, i] or x[i] < prior[1, i]
            
            if outside_prior:
                prior_iter=0
                # Resample until we 
                # obtain a sample within the prior bounds or attemps become > max_prior_iter
                print("iteration: ", i)
                x_arr=[]
                # pbar=tqdm(total=max_prior_iter)
                while (x[i] > prior[0, i] or x[i] < prior[1, i]) and prior_iter<max_prior_iter:
                    x[i] = draw_icdf_samples(alpha,beta[i],x)
                    x_arr.append(x[i])
                    prior_iter+=1
                    # pbar.update(1)
                # pbar.close()
                print("Last sample: ",x_arr[-1])
                if prior_iter>=max_prior_iter:
                    print("Priors: {}".format(prior[:,i]))
                    print("\nAlpha: {}, Beta: {}".format(alpha,beta[i]))
                    raise ValueError("Number of prior resamples exceeded max_prior_iter")
        else:
            x[i] = draw_icdf_samples(alpha,beta[i],x)

    return x

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


'''Second modified version of GCR'''
def gcr_fgmodes_1d_v2(idx, vis, w, Eh, Nih, Nparams, y, flags, E, Ninv, fgmodes, f0=None, map_estimate=False, verbose=False,
    multiprocess_seed=912983): # Eh, Nih, --> only needed when we activate omegas

    pid = current_process().pid
    seed = multiprocess_seed + pid*1000 + idx
    np.random.seed(seed)

    Nfreqs, Nmodes = fgmodes.shape
    d = vis.reshape((1, max(Nfreqs, len(vis.T))))

    # Extract precomputed matrices needed by the linear system
    A, Ni, Ai = build_matrices(Nparams, y, flags, E, Ninv, fgmodes)

    if map_estimate:
        oma = np.zeros((Nfreqs, 1), dtype=complex)
        omb = np.zeros((Nfreqs, 1), dtype=complex)
    else:
        # Unit complex Gaussian random realisation
        omi, omj = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs,1)
        omk, oml = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs,1)
        oma, omb = (omi + 1.0j * omj) / 2**0.5, (omk + 1.0j * oml) / 2**0.5
    # print("Shapes: \nE:",E.shape,
    #     #   "\ny.conj:",y.conj().shape,
    #     "\nNi: ",Ni.shape,
    #     "\nw:",w.shape,
    #     "\nvis: ",vis.shape,
    #     "\nEh: ",Eh.shape,
    #     "\noma: ",oma.shape,
    #     "\nomb: ",omb.shape,
    #     "\nNih: ",Nih.shape,
    #     "\nfgmodes.T.conj(): ",fgmodes.T.conj().shape,
    #     "\nfgmodes: ",fgmodes.shape,
    #     "\nA: ",A.shape,
    #     "\nd: ",d.shape,
    #     "\ny.conj() * Nih * omb: ",(y.conj()[:,np.newaxis] * Nih[:,np.newaxis] * omb).shape,
    #     "\nNi * w * d: ",(Ni[:,np.newaxis] * w[:,np.newaxis] * d.T).shape)
    
    # Construct RHS vector
    b = np.zeros((Nfreqs + Nmodes, 1), dtype=complex)
    
    b[:Nfreqs] = (y.conj() * Ninv.diagonal() * d).T + y.conj()[:,np.newaxis] * Eh @ oma #+ (y.conj()[:,np.newaxis] * Nih[:,np.newaxis] * omb)
    b[Nfreqs:] = fgmodes.T.conj() @ (y.conj() * Ninv.diagonal() * d).T #+ fgmodes.T.conj() @ (y.conj()[:,np.newaxis] * Nih[:,np.newaxis] * omb)
    
    # Run CG solver, preconditioned by M=Ai
    x0 = None
    if f0 is not None:
        x0 = np.concatenate((np.zeros(Nparams, dtype=complex), f0))
    
    xsoln, info = sp.sparse.linalg.cgs(A, b,x0=x0, M=Ai,tol=1e-12) #maxiter=int(1e5) , rtol=1e-12 , x0=x0, M=Ai
    if verbose:
        residual = np.abs(A @ xsoln - b[:, 0]).mean()
        # x0=np.concatenate([d.T.real,d.T.imag],axis=0)
        # residual_og = np.abs(A @ x0 - b[:, 0]).mean()
    else:
        residual = None

    # Return solution vector
    return xsoln, residual,  info  #residual_og, b

'''Second modified version'''
def gcr_fgmodes(
    vis, w, fgmodes, Nparams, sys_model_past, flags, signal_S, Ninv, f0=None, nproc=1, map_estimate=False,
    verbose=False
):
    """
    Perform the GCR step on all time samples, using parallelisation if
    possible.

    Parameters:
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
    # print("Shape of samples array: ",samples.shape)
    if verbose:
        residuals = np.zeros(vis.shape[0], dtype=float)
        # residuals_og = np.zeros(vis.shape[0], dtype=float)
        info = np.zeros(vis.shape[0], dtype=float)
    else:
        residuals = None
        info = None
    idxs = np.arange(vis.shape[0])
    #Time invariant calculations
    Eh=sp.linalg.sqrtm(signal_S)  
    Nih = np.sqrt(np.diag(Ninv))
    # Run GCR method on each time sample in parallel
    if verbose:
        st = time.time()
    with Pool(nproc) as pool:
        samples, residuals,  info = zip(*pool.map(
            lambda idx: gcr_fgmodes_1d_v2(
                idx=idx,
                vis=vis[idx],
                w=w,
                fgmodes=fgmodes,
                f0=f0,
                Nparams=Nparams,
                y=sys_model_past[idx],
                flags=flags,
                E=signal_S,
                Ninv=Ninv,
                Eh=Eh, #use when omegas are active
                Nih=Nih,
                map_estimate=map_estimate,
                verbose=verbose
            ),
            idxs,
        )
        )
    samples = np.array(samples).reshape((vis.shape[0], -1))
    # print(samples.shape)
    residuals = np.array(residuals)
    info = np.array(info)
    # residuals_og= np.array(residuals_og)
    # Return sample
    if verbose:
        print(f"{time.time() - st:<12.1f}", end="")
        print(f"{info.mean():<8.1f}", end="")
        print(f"{residuals.mean():<12.2e}", end="")
        # print(f"{residuals_og.mean():<12.2e}", end="")
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


def build_matrices(Nparams, y, flags, E, Ninv, fgmodes):
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
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
    
    Returns:
        matrices (list of array_like):
            List containing necessary GCR operators (`matrices[0]`) and the
            linear operator A in the GCR Ax=b solve step.
    """
    Nfreqs = E.shape[0]
    
    E_inv=np.linalg.inv(E) #FIXME   
    # G_inv=np.linalg.inv(fgmodes) #FIXME
    
    # Construct necessary operators for GCR
    inner_prod= (y.conj().T * Ninv.diagonal() *  y)
    Ni_flagged = flags.T * (inner_prod) * flags  # Ni # FIXME
    
    # Construct operator matrix
    A = np.zeros((Nparams, Nparams), dtype=complex)
    A[:Nfreqs, :Nfreqs] = y.conj()[:,np.newaxis] * E_inv * y[:,np.newaxis] + np.diag(Ni_flagged)  # A11: y.dag E^-1 y + y.dag * Ni * y
    A[:Nfreqs, Nfreqs:] = Ni_flagged[:,np.newaxis] * fgmodes # A12: y.dag * Ni * y * G
    A[Nfreqs:, :Nfreqs] = (fgmodes.conj() * Ni_flagged[:,np.newaxis]).T #A21: G.dag * y.dag * Ni * y
    A[Nfreqs:, Nfreqs:] = fgmodes.T.conj() @ (Ni_flagged[:,np.newaxis] * fgmodes) #A22: y.dag * G^-1 *y + G.dag * y.dag * Ni * y * G (y.conj()[:,np.newaxis] * G_inv * y[:,np.newaxis] + )

    A_pinv = np.linalg.pinv(A)  # pseudo-inverse, to be used as a preconditioner
    
    return A, Ni_flagged, A_pinv

def gibbs_step_fgmodes(
    vis,
    flags,
    signal_S,
    fgmodes,
    Ninv,
    nm_list,
    h_j,
    b_sys_past,
    sys_model_past,
    freqs,
    lsts,
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
        signal_S (array_like):
            Current value of the EoR signal frequency-frequency covariance.
        b_sys (array_like):
            ...
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
        ps_sample (array_like):
            Sample of the signal power spectrum bandpowers, shape `(Nfreqs,)`.
        fg_amps (array_like):
            Sample of the foreground amplitudes, shape `(Nmodes,)`.
        b_sys (array_like):
            Array of systematics amplitudes of shape (len(nm_list))
    """
    # Shape of data and operators
    Ntimes=vis.shape[0]
    Nfreqs = vis.shape[1] 
    Nmodes = fgmodes.shape[1]
    Nparams = Nfreqs + Nmodes
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"

    # Precompute 2D Fourier operator matrix
    fourier_op = utils.fourier_operator(Nfreqs)

    # Get matrices necessary for the GCR step
    # sys_model_past=np.load('/Users/user/Documents/Codes/hydra_sys_project1/GCR_test_scripts/gain.npy',allow_pickle=False)
    sys_model_past=np.ones_like(vis)
    # (1) Solve GCR equation to get EoR signal and foreground amplitude realisations
    cr = gcr_fgmodes(
        vis=vis, w=flags, fgmodes=fgmodes, Nparams=Nparams, sys_model_past=sys_model_past, flags=flags, signal_S=signal_S, Ninv=Ninv, f0=f0, nproc=nproc, map_estimate=map_estimate,
    verbose=verbose)   #FIXME: Running test on the d=(1+delta g)s+n form of the equations 
    # t0=time.time()
    
    # print("Eor-FG GCR done in time: {}".format(t1-t0))
    # np.save('test_files/eor_fg_data.npy',vis)
    # np.save('test_files/eor_fg_gain.npy',sys_model_past)
    # np.save('test_files/signal_S.npy',signal_S)
    
    # Extract separate signal and FG parts from the solution
    signal_cr = cr[:, : -fgmodes.shape[1]]
    fg_amps = cr[:, -fgmodes.shape[1] :]
    # t2=time.time()
    
    # Full model of data is sum of EoR (GCR) + FG model
    model = (signal_cr + fg_amps @ fgmodes.T)  # np.einsum('ijk,lk->ijl', fg_amps, fgmodes) # sky model
    # np.save('test_files/model_init.npy',model)
    # np.save('test_files/signal_cr.npy',signal_cr)
    # np.save('test_files/fg_amps.npy',fg_amps)
    # print("data saved and models made in time: {}".format(t2-t1))
    # t3=time.time()
    # 1a. Solve GCR equation to obtain estimate of systematic component
    # pr.enable()
    
    
    '''This block is for only when we are informing gcr_sys with true solution'''
    #FIXME: remove the following file-loading from code
    # uvd=UVData()
    # uvd.read('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_data/vis-eor-fgs.uvh5')
    # antpairpols = uvd.get_antpairpols()    
    # uvd = utils.form_pseudo_stokes_vis(uvd)
    # clean_vis=uvd.get_data(antpairpols[0], force_copy=True)
    
    clean_vis=np.load('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/dummy_sky.npy',allow_pickle=False)
    master_plotter([clean_vis],col_labels=[' '],fig_title='Clean visibility loaded from file')
    
    b_sys=sys_sol.gcr_sys_v1(Binv=100*Bi,d=vis-clean_vis,Ninv=Ninv,s=clean_vis.flatten('F'),H=h_j,b_sys_past=b_sys_past,verbose=verbose,iter=iter)
    # pr.disable()
    # t4=time.time()

    # ps = pstats.Stats(pr, stream=sys.stdout)
    # ps.strip_dirs()
    # ps.sort_stats('time').print_stats()

    # # Update systematics model
    # sys_model = h_j @ b_sys # Shape of flattened data
    # sys_model= np.reshape(sys_model,[Ntimes,Nfreqs],order='F') #Gives data-like model 
    # sys_model += np.ones_like(sys_model, dtype='complex')  # Adding ones to the systematics solution
    
    sys_model = np.ones_like(vis) #FIXME: test code. Delete after use
    
    # sys_ref = np.load('sys_select.npy',allow_pickle=False)
    # model = ((sys_model+sys_ref) * model) - sys_ref * model
    model = (sys_model * model)
    # t5=time.time()
    # print("Data saved and models made in time: {}".format(t5-t4))
    # Chi-squared is computed as the sum of ( |data - model - sys_model| / noise )^2,
    # i.e. as a sum of standard normal random variables.
    # FIXME: this will need to be changed to account for time-dependent
    # flags (i.e. when we have a different N per time).
    chisq = np.abs(vis - model)**2 * Ninv.diagonal()[None, :]
    chisq_mean = chisq[:, flags].mean()

    if verbose==True:
        chisq_mean = chisq[:, flags].mean()
        print(f"{chisq_mean:<9.1e}", end=" ")

        # if chisq_mean > 10:
        #     print(f"{chisq_mean:<9.1e}", end=" ")
        # else:
        #     print(f"{chisq_mean:<9.3f}", end=" ")
    
    # (2) Sample EoR signal power spectrum (and also convert to equivalent
    # covariance matrix sample)
    # t6=time.time()
    # print("Sampler starting after time: {}".format(t6-t5))
    ps_sample = sample_S(s=signal_cr, prior=ps_prior)
    #FIXME: Fix prior bounds to properly sample PS
    # ps_sample = np.load('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_files/ps_sample.npy',allow_pickle=False)
    # t7=time.time()
    # print("Sampling done in time: {}".format(t7-t6))
    # print("Nfreqs: ",Nfreqs)
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
            (vis - (model))[:, flags].conj()
            @ Ninv[flags][:, flags]
            @ (vis - (model))[:, flags].T
        )
        - (
            signal_cr[:, flags].conj()
            @ Sinv[flags][:, flags]
            @ signal_cr[:, flags].T
        )
    ))
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
        S_initial (array_like):
            Initial guess for the EoR signal frequency-frequency covariance.
            A better guess should result in faster convergence.
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
            Frequency array (shape: )
        lsts:
            Time array in LSTS (shape: )

    Returns:
        signal_cr (array_like):
            Samples of the signal, shape `(Niter, Ntimes, Nfreqs)`.
        signal_S (array_like):
            Samples of the signal covariance, shape `(Niter, Nfreqs, Nfreqs)`.
            These are simply transformations of the power spectrum.
        signal_ps (array_like):
            Sample of the signal power spectrum bandpowers, shape
            `(Niter, Nfreqs)`.
        fg_amps (array_like):
            Samples of the foreground amplitudes, shape `(Niter, Nmodes)`.
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
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"
    assert fgmodes.shape[0] == Nfreqs, "fgmodes must have shape (Nfreqs, Nmodes)"
    if len(Ninv.shape) == 3:
        assert (
            Ninv.shape[0] == Ntimes
        ), "Ninv shape must be (Ntimes, Nfreqs, Nfreqs) or (Nfreqs, Nfreqs)"
    # Set up arrays for sampling
    signal_cr = np.zeros((Niter, Ntimes, Nfreqs), dtype=complex)
    signal_S = np.zeros((Niter, Nfreqs, Nfreqs))
    signal_ps = np.zeros((Niter, Nfreqs))
    fg_amps = np.zeros((Niter, Ntimes, Nmodes), dtype=complex)
    b_sys = np.zeros((Niter, nm_list.shape[0]), dtype=complex)
    # Useful debugging statistics
    chisq = np.zeros((Niter, Ntimes, Nfreqs))
    ln_post = np.zeros(Niter)
    # Set initial value for signal_S
    signal_S = S_initial.copy()

    # Precompute h_j systematics projection operator
    # FIXME: include path as arg in import file
    nm_list_select = np.loadtxt('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/nm_list_select')
    h_j = sys_sol.h_j_op(freqs=freqs,lsts=lsts,nm_list=nm_list_select) #nm_list containing dl fr values
    # Loop over iterations
    if verbose:
        print("Iter     Time [s]    Info    |Ax - b|    T_Sys(s)    Sys Info    Sys |Ax-b|    Chisq    ln Post")
        print("-----    --------    ----    --------    --------    --------    ----------    -----    -------")

    for i in range(Niter):
        # print("Iteration: ",i)
        # print("Signal: {}".format(signal_S))
        if verbose:
            print(f"{i+1:<9d}", end="")
        if i==0:
            vis = np.load('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/dummy_sky.npy',allow_pickle=False) #FIXME: this is a test code. Remove once done. 
            clean_vis=np.load('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/dummy_sky.npy',allow_pickle=False)

            # FIXME: include path as arg in import file
            # uvd=UVData()
            # uvd.read('/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/test_data/vis-eor-fgs.uvh5')
            # uvd=utils.form_pseudo_stokes_vis(uvd)
            # antpairpols = uvd.get_antpairpols()
            # clean_vis=uvd.get_data(antpairpols[0], force_copy=True)
            sys_model_past= (vis/clean_vis)
            b_sys_past = np.ones(h_j.shape[1],dtype='complex') #Starting from 0 works best, DO NOT CHANGE. 
            
        else:
            b_sys_past=b_sys[i-1]
            sys_model_past = (h_j @ b_sys_past).reshape([Ntimes,Nfreqs],order='F')
            sys_model_past = np.ones_like(sys_model_past,dtype='complex')+sys_model_past # Implementing the 1+del g model. 
        # B_cov_inv=np.sqrt(b_sys_past)*np.eye(len(b_sys_past)) #Uncomment this to sample for B
        B_cov_inv=1*np.eye(len(b_sys_past)) #B is an identity matrix 

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
                Bi=B_cov_inv,
                h_j=h_j,
                b_sys_past=b_sys_past,
                sys_model_past=sys_model_past,
                ps_prior=ps_prior,
                f0=None,
                nproc=nproc,
                iter=i,
                map_estimate=map_estimate,
                verbose=verbose
            )      
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

    if verbose:
        print()

    return signal_cr, signal_S, signal_ps, fg_amps, b_sys, chisq, ln_post
