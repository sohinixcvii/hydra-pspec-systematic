import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pyuvdata import UVData
from uvtools.plot import waterfall
from uvtools.dspec import gen_window
from uvtools.utils import FFT, fourier_freqs
from astropy import units
from astropy.units import Quantity
import pyuvdata.utils as uvutils


# def cost_fn(x, data, h_j, clean_vis, noise_std=1.0):
    # Ntimes, Nfreqs = data.shape[0], data.shape[1]
    # Nmodes = h_j.shape[1]
    # # Calculate model (currently assumes perfect clean model!)
    # model = (h_j @ complexify(x)).reshape((Nfreqs, Ntimes)).T * clean_vis
    
    # # Calculate difference and return real + imaginary components
    # diff = ((data - model) / noise_std).flatten()
    # return realify(diff)
    
def form_pseudo_stokes_vis(uvd, convention=1.0):
    """
    Form pseudo-Stokes I visibilities from xx and yy.

    Parameters:
        uvd (pyuvdata.UVData):
            UVData object containing XX and YY polarization visibilities.
        convention (float):
            Factor for getting pI from XX + YY, i.e.
            pI = convention * (XX + YY).  Defaults to 1.0.

    Returns:
        uvd (pyuvdata.UVData):
            UVData object containing pI visibilities.

    """
    assert isinstance(uvd, UVData), "uvd must be a pyuvdata.UVData object."

    if uvutils.polstr2num("pI") not in uvd.polarization_array:
        xx_pol_num = uvutils.polstr2num("xx")
        yy_pol_num = uvutils.polstr2num("yy")
        xpol_ind = np.where(uvd.polarization_array == xx_pol_num)[0]
        ypol_ind = np.where(uvd.polarization_array == yy_pol_num)[0]
        uvd.data_array[..., xpol_ind] += uvd.data_array[..., ypol_ind]
        uvd.data_array *= convention
        uvd.select(polarizations=["xx"])

    return uvd

def fourier_mode_2d(freqs_Hz, times_sec, modes, box=None):
    """
    Construct a set of 2D Fourier modes from a list of wavenumber integers, 
    to form an incomplete set of 2D Fourier modes.

    Parameters
    ----------
    freqs_Hz (array_like):
        Frequency array, in Hz. Should be ordered.
        
    times_sec (array_like):
        Time array, in hours. Should be ordered.

    modes (list of tuple of int):
        List of mode integer pairs to include in operator.

    box (tuple of tuple):
        NOT IMPLEMENTED
        Keep all modes within a box, defined by the tuple:
        `((delay_min, delay_max), (frate_min, frate_max))`.
        The delays are in ns and the fringe rates in mHz.
    """
    Nfreqs, Ntimes = freqs_Hz.size, times_sec.size
    
    # Get grid spacing in expected units
    dfreq = (freqs_Hz[1] - freqs_Hz[0])
    dtime = (times_sec[1] - times_sec[0])

    # Get FFT wavenumbers
    kfreq = np.fft.fftfreq(Nfreqs, d=dfreq) # sec #* 1e9 # ns
    ktime = np.fft.fftfreq(Ntimes, d=dtime) # Hz * 1e3 # mHz

    # Get FFT mode integers
    nfreq = (np.fft.fftfreq(Nfreqs) * Nfreqs).astype(int)
    ntime = (np.fft.fftfreq(Ntimes) * Ntimes).astype(int)

    # Frequency/time grids with respect to origin
    f = freqs_Hz - freqs_Hz[0]
    t = times_sec - times_sec[0]

    # Get indices of modes we want to keep
    basis_fns = np.zeros((len(modes), Nfreqs, Ntimes), dtype=np.complex128)
    for i, mode in enumerate(modes):
        nf, nt = mode
        # print(nf, nt)
        assert isinstance(nf, int), "modes must only contain pairs of integers"
        assert isinstance(nt, int), "modes must only contain pairs of integers"
        assert nf in nfreq, "Delay mode nf=%d not in available range (%d -- %d)." \
            % (nf, nfreq.min(), nfreq.max())
        assert nt in ntime, "Fringe rate mode nt=%d not in available range (%d -- %d)." \
            % (nt, ntime.min(), ntime.max())

        # Get mode indices
        idx_f = np.where(nfreq == nf)[0][0]
        idx_t = np.where(ntime == nt)[0][0]
        #mode_idxs.append( (idx_f, idx_t) )

        # print(kfreq[idx_f], ktime[idx_t])

        # Add basis function to operator
        basis_fns[i] = np.exp(2.*np.pi*1.j * (  kfreq[idx_f] * f[:,np.newaxis]
                                     + ktime[idx_t] * t[np.newaxis,:] ) ) \
                     / np.sqrt(Nfreqs * Ntimes)
        
    return basis_fns, kfreq * 1e9, ktime * 1e3


def sys_modes(freqs_Hz, times_sec, modes):
    """
    Construct systematic mode operator, which is a 2D Fourier basis.
    """
    u, kfreq, ktime = fourier_mode_2d(freqs_Hz=freqs_Hz, 
                                      times_sec=times_sec, 
                                      modes=modes)
    return u.reshape((u.shape[0], -1)).T

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
    # data_fr = FFT(data * time_window, axis=0)
    # data_dly = FFT(data * freq_window, axis=1)
    data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)
    

    return data_fr_dly

def fourier_operator(n, unitary=True):
    """
    Fourier operator for matrix side length n.

    Multiplying a data vector by this matrix operator is equivalent to running
    the following code:
    ```
    data = ...
    # ifftshift and fftshift are interchangeable
    data_fft = numpy.fft.ifftshift(data)
    data_fft = numpy.fft.fft(data_fft)
    data_fft = numpy.fft.fftshift(data_fft)
    ```

    Parameters:
    	n (int):
    		Length of the data that the operator will be applied to.
        unitary (bool):
            Whether the matrix should be unitary, i.e. F^dagger F = I.

    Returns:
    	fourier_op (array_like):
    		Complex Fourier operator matrix of shape `(n, n)`.
    """
    norm = 1.
    if unitary:
        norm = np.sqrt(n)

    i_x = (np.arange(n) - n//2).reshape(1, -1)
    i_k = (np.arange(n) - n//2).reshape(-1, 1)

    fourier_op = np.exp(-2*np.pi*1j * (i_k * i_x / n)) / norm
    return fourier_op


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

def data_fr_dly_to_t_f(data_fr_dly, freqs, times,
                       windows=None, freq_window_kwargs=None, time_window_kwargs=None,
                       eps=1e-12):
    """
    Invert the forward transform:
        data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)

    Parameters
    ----------
    data_fr_dly : (Ntimes, Nfreqs) complex array
        Data in fringe-rate / delay space (the output you already have).
    freqs : (Nfreqs,) array
        Frequencies (only used to rebuild the same frequency window).
    times : (Ntimes,) array
        Times (only used to rebuild the same time window).
    windows : str or sequence or None
        Same argument you passed to `gen_window` in the forward step.
    freq_window_kwargs, time_window_kwargs : dict or None
        Same kwargs you used in the forward step.
    eps : float
        Small number to avoid division-by-zero when un-tapering.

    Returns
    -------
    data : (Ntimes, Nfreqs) complex array
        Reconstructed visibilities in time–frequency space.
    """
    freq_window_kwargs = freq_window_kwargs or {}
    time_window_kwargs = time_window_kwargs or {}

    # Rebuild the *same* windows you used forward
    if windows is not None:
        time_window = gen_window(windows, times.size, **time_window_kwargs)
        freq_window = gen_window(windows, freqs.size, **freq_window_kwargs)
    else:
        time_window = gen_window(None, times.size, **time_window_kwargs)
        freq_window = gen_window(None, freqs.size, **freq_window_kwargs)

    # Broadcast to 2D
    time_window = time_window[:, None]         # (Ntimes, 1)
    freq_window = freq_window[None, :]         # (1, Nfreqs)

    # 1) inverse along frequency (delay -> freq)
    x = np.fft.ifft(data_fr_dly, axis=1)

    # 2) divide out the frequency window (safe divide)
    x = x / np.where(np.abs(freq_window) > eps, freq_window, 1.0)

    # 3) inverse along time (fringe-rate -> time)
    data = np.fft.ifft(x, axis=0)

    # 4) divide out the time window (safe divide)
    data = data / np.where(np.abs(time_window) > eps, time_window, 1.0)

    return data
