import matplotlib.pyplot as plt
import numpy as np
# from config import *
import os
import scipy.stats as sci_st
from pyuvdata import UVData
from astropy import units
from pathlib import Path
from matplotlib.colors import LogNorm, Normalize
from uvtools.dspec import gen_window
from uvtools.utils import FFT, fourier_freqs
from uvtools.plot import waterfall

# if hera_sim.__version__.startswith('0'):
#     from hera_sim.rfi import _listify
# else:
#     from hera_sim.utils import _listify
op_dir='paper_plots/'

def plot_waterfalls(data, freqs, times, windows=None, mode='log', fig=None,ax=None, xlabel=None,
                    vmin=None, vmax=None, cmap='inferno', dynamic_range=None, limit_drng='all',
                    baseline=None, horizon_color='magenta', plot_limits=None, colorbar_flag=True,
                    freq_window_kwargs=None, time_window_kwargs=None):
    """
    Make a 2x2 grid of waterfall plots.
    
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
        
    mode : str, optional
        Which transformation to apply to the data before plotting. Options are:
          - 'phs':  Phase angle.
          - 'abs':  Absolute value.
          - 'real': Real value.
          - 'imag': Imaginary value.
          - 'log':  Log (base-10) of absolute value.
        Default: 'log'.

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for visualizing the data. Default is to use the inferno 
        colormap.
        
    dynamic_range : float, optional
        Number of orders of magnitude to use for limiting the dynamic range of the 
        colormap. This parameter is only used if ``mode`` is set to 'log' and the 
        ``limit_drng`` parameter is not None. If the conditions to use this 
        parameter are met, then the vmin parameter is set to be dynamic_range orders 
        of magnitude less than vmax. That is, if vmax = np.log10(np.abs(data)).max(), 
        then vmin = vmax - dynamic_range. Default is to not limit the dynamic range.
        
    limit_drng : str or array-like of str, optional
        Choice of which plots for which to limit the dynamic range. Possible choices 
        are 'freq', 'time', 'delay', and 'fringe_rate'. If any of these are chosen, 
        then the plots that have one of the axes match the specified choices will 
        have their dynamic range limited. For example, passing 'delay' to this 
        parameter will limit the dynamic range for the delay-time and delay-fringe-rate 
        plots. Default is to limit the dynamic range for all plots. 
        
    baseline : float or array-like of float, optional
        Baseline length or baseline position in units of meters. If this parameter is 
        specified, then the geometric horizon is plotted as a vertical line in the 
        delay-space plots. Default is to not plot the geometric horizon.
        
    horizon_color : str, 3-tuple, or 4-tuple, optional
        Color to use for the vertical lines indicating the geometric horizon. This 
        may either be a string, 3-tuple specifying RGB values, or 4-tuple specifying 
        RGBA values. Default is to use magenta.
        
    plot_limits : dict, optional
        Dictionary whose keys may be any of ('freq', 'time', 'delay', 'fringe-rate') 
        and whose values are length 2 array-like objects specifying the bounds for 
        the corresponding axis. For horizontal axes, these should be ordered from low 
        to high; for vertical axes, these should be ordered from high to low. For 
        example, passing {'delay' : (-500, 500)} will limit the delay axis to values 
        between -500 and +500 nanoseconds. Frequency units should be in Hz; time 
        units should be in JD; delay units should be in ns; fringe rate units should 
        be in mHz. Default is to use the full extent of each axis.
        
    freq_window_kwargs : dict, optional
        Keyword arguments to pass to uvtools.dspec.gen_window for generating the 
        frequency taper. Default is to pass no keyword arguments.
        
    time_window_kwargs : dict, optional
        Keyword arguments to pass to uvtools.dspec.gen_window for generating the 
        time taper. Default is to pass no keyword arguments.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib.figure.Figure object containing the plots.
    """
    # do some data prep
    freq_window_kwargs = freq_window_kwargs or {}
    time_window_kwargs = time_window_kwargs or {}
    time_window = gen_window(windows, times.size, **time_window_kwargs)
    freq_window = gen_window(windows, freqs.size, **freq_window_kwargs)
        
    time_window = time_window[:, None]
    freq_window = freq_window[None, :]
    data_fr = FFT(data * time_window, axis=0)
    data_dly = FFT(data * freq_window, axis=1)
    data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)
    
    fringe_rates = fourier_freqs(times * units.day.to('s')) * 1e3 # mHz
    dlys = fourier_freqs(freqs) * 1e9 # ns
    plot_freqs = freqs / 1e6
    jd = int(np.floor(times[0]))
    plot_times = times - jd
    
    if baseline is not None:
        horizon = np.linalg.norm(baseline) / constants.c.value * 1e9
    
    if ax==None:
        fig = plt.figure(figsize=(10,10),facecolor='white')
        ax = fig.subplots(1,1)
    ax.set_facecolor('white')
    ax.tick_params(direction='out', length=6, width=2, colors='black')
            #    grid_color='r', grid_alpha=0.5)
    # for j, ax in enumerate(axes.ravel()):
    j=1
    column = j % 2
    row = j // 2
    if xlabel==None:
        xlabel = "Delay [ns]"
    ylabel = "Time Since JD%d [days]" % jd if column == 0 else "Fringe Rate [mHz]"
    ax.set_xlabel(xlabel, color='black')
    ax.set_ylabel(ylabel, color='black')
    
    xlimits, ylimits = None, None
        # if column == 0 and row == 0:
        #     use_data = data
        #     extent = (
        #         plot_freqs.min(), plot_freqs.max(), plot_times.max(), plot_times.min()
        #     )
        #     vis_label = r"$\log_{10}|V(\nu, t)|$ [Jy]"
        #     if plot_limits is not None:
        #         xlimits = plot_limits.get("freq", extent[:2])
        #         ylimits = plot_limits.get("time", extent[2:])
        # elif column == 0 and row == 1:
        #     use_data = data_dly
        #     extent = (dlys.min(), dlys.max(), plot_times.max(), plot_times.min())
        #     vis_label = r"$\log_{10}|\tilde{V}(\tau, t)|$ [Jy Hz]"
        #     if plot_limits is not None:
        #         xlimits = plot_limits.get("delay", extent[:2])
        #         ylimits = plot_limits.get("time", extent[2:])
        # elif column == 1 and row == 0:
        #     use_data = data_fr
        #     extent = (
        #         plot_freqs.min(), plot_freqs.max(), fringe_rates.max(), fringe_rates.min()
        #     )
        #     vis_label = r"$\log_{10}|\tilde{V}(\nu, f)|$ [Jy s]"
        #     if plot_limits is not None:
        #         xlimits = plot_limits.get("freq", extent[:2])
        #         ylimits = plot_limits.get("fringe_rate", extent[2:])
        # else:
    use_data = data_fr_dly
    extent = (dlys.min(), dlys.max(), fringe_rates.max(), fringe_rates.min())
    vis_label = r"$\log_{10}|\tilde{V}(\tau, f)|$ [Jy Hz s]"
    if plot_limits is not None:
        xlimits = plot_limits.get("delay", extent[:2])
        ylimits = plot_limits.get("fringe_rate", extent[2:])
            
    xlimits = xlimits or extent[:2]
    ylimits = ylimits or extent[2:]
    
    if vmin is None and vmax is None:
        if dynamic_range is not None and mode == 'log':
            vmax = np.log10(np.abs(use_data)).max()
            vmin = vmax - dynamic_range
        else:
            vmin, vmax = None, None
        
    clip_drng = False
    if limit_drng == 'all':
        limit_drng = ("freq", "time", "delay", "fringe_rate")
    if "time" in limit_drng:
        if column == 0:
            clip_drng = True
    if "freq" in limit_drng:
        if row == 0:
            clip_drng = True
    if "delay" in limit_drng:
        if row == 1:
            clip_drng = True
    if "fringe_rate" in limit_drng:
        if column == 1:
            clip_drng = True
            
    if not clip_drng:
        vmin, vmax = None, None
        
    cbar_label = vis_label if mode == 'log' else "Phase [rad]"
    fig.sca(ax)
    cax = waterfall(
        use_data, extent=extent, mode=mode, vmin=vmin, vmax=vmax, cmap=cmap,origin='lower'
    )
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    if baseline is not None and row == 1:
        ax.axvline(horizon, color=horizon_color, ls='--')
        ax.axvline(-horizon, color=horizon_color, ls='--')
    
    if colorbar_flag==True:
        _ = plt.colorbar(cax)
        _.set_label(cbar_label,c='black')
        _.ax.tick_params(axis='y',which='both', color='black', labelcolor='black')
    return cax,data_fr_dly

def plot_waterfalls_from_dlfr(data_dlfr, freqs, times,mode='log', fig=None,ax=None, xlabel=None,
                    vmin=None, vmax=None, cmap='inferno', dynamic_range=None, limit_drng='all',
                    baseline=None, horizon_color='magenta', plot_limits=None, colorbar_flag=True,cbar_label=None):
    """
    Make a 2x2 grid of waterfall plots.
    
    This function takes a 2D array of visibility data in the DL-FR space, as well 
    as the corresponding frequency and time arrays (in units of Hz and JD, respectively) and makes a plot 
    on the given axis.
    
    Parameters
    ----------
    data : ndarray, shape=(NTIMES,NFREQS)
        Array containing the visibility to be plotted. Assumed to be in units of Jy. 
        
    freqs : ndarray, shape=(NFREQS,)
        Array containing the observed frequencies. Assumed to be in units of Hz.
        
    times : ndarray, shape=(NTIMES,)
        Array containing the observed times. Assumed to be in units of JD.
        
    mode : str, optional
        Which transformation to apply to the data before plotting. Options are:
          - 'phs':  Phase angle.
          - 'abs':  Absolute value.
          - 'real': Real value.
          - 'imag': Imaginary value.
          - 'log':  Log (base-10) of absolute value.
        Default: 'log'.

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for visualizing the data. Default is to use the inferno 
        colormap.
        
    dynamic_range : float, optional
        Number of orders of magnitude to use for limiting the dynamic range of the 
        colormap. This parameter is only used if ``mode`` is set to 'log' and the 
        ``limit_drng`` parameter is not None. If the conditions to use this 
        parameter are met, then the vmin parameter is set to be dynamic_range orders 
        of magnitude less than vmax. That is, if vmax = np.log10(np.abs(data)).max(), 
        then vmin = vmax - dynamic_range. Default is to not limit the dynamic range.
        
    limit_drng : str or array-like of str, optional
        Choice of which plots for which to limit the dynamic range. Possible choices 
        are 'freq', 'time', 'delay', and 'fringe_rate'. If any of these are chosen, 
        then the plots that have one of the axes match the specified choices will 
        have their dynamic range limited. For example, passing 'delay' to this 
        parameter will limit the dynamic range for the delay-time and delay-fringe-rate 
        plots. Default is to limit the dynamic range for all plots. 
        
    baseline : float or array-like of float, optional
        Baseline length or baseline position in units of meters. If this parameter is 
        specified, then the geometric horizon is plotted as a vertical line in the 
        delay-space plots. Default is to not plot the geometric horizon.
        
    horizon_color : str, 3-tuple, or 4-tuple, optional
        Color to use for the vertical lines indicating the geometric horizon. This 
        may either be a string, 3-tuple specifying RGB values, or 4-tuple specifying 
        RGBA values. Default is to use magenta.
        
    plot_limits : dict, optional
        Dictionary whose keys may be any of ('freq', 'time', 'delay', 'fringe-rate') 
        and whose values are length 2 array-like objects specifying the bounds for 
        the corresponding axis. For horizontal axes, these should be ordered from low 
        to high; for vertical axes, these should be ordered from high to low. For 
        example, passing {'delay' : (-500, 500)} will limit the delay axis to values 
        between -500 and +500 nanoseconds. Frequency units should be in Hz; time 
        units should be in JD; delay units should be in ns; fringe rate units should 
        be in mHz. Default is to use the full extent of each axis.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib.figure.Figure object containing the plots.
    """
    fringe_rates = fourier_freqs(times * units.day.to('s')) * 1e3 # mHz
    dlys = fourier_freqs(freqs) * 1e9 # ns
    jd = int(np.floor(times[0]))

    if baseline is not None:
        horizon = np.linalg.norm(baseline) / constants.c.value * 1e9
    
    if ax==None:
        fig = plt.figure(figsize=(10,10),facecolor='white')
        ax = fig.subplots(1,1)
    ax.set_facecolor('white')
    ax.tick_params(direction='out', length=6, width=2, colors='black')
            #    grid_color='r', grid_alpha=0.5)
    # for j, ax in enumerate(axes.ravel()):
    j=1
    column = j % 2
    row = j // 2
    if xlabel==None:
        xlabel = "Delay [ns]"
    ylabel = "Time Since JD%d [days]" % jd if column == 0 else "Fringe Rate [mHz]"
    ax.set_xlabel(xlabel, color='black')
    ax.set_ylabel(ylabel,color='black')
    
    xlimits, ylimits = None, None
    use_data = data_dlfr
    extent = (dlys.min(), dlys.max(), fringe_rates.max(), fringe_rates.min())
    vis_label = r"$\log_{10}|\tilde{V}(\tau, f)|$ [Jy Hz s]"
    if plot_limits is not None:
        xlimits = plot_limits.get("delay", extent[:2])
        ylimits = plot_limits.get("fringe_rate", extent[2:])
            
    xlimits = xlimits or extent[:2]
    ylimits = ylimits or extent[2:]
    
    if vmin is None and vmax is None:
        if dynamic_range is not None and mode == 'log':
            vmax = np.log10(np.abs(use_data)).max()
            vmin = vmax - dynamic_range
        
    clip_drng = False
    if limit_drng == 'all':
        limit_drng = ("freq", "time", "delay", "fringe_rate")
    if "time" in limit_drng:
        if column == 0:
            clip_drng = True
    if "freq" in limit_drng:
        if row == 0:
            clip_drng = True
    if "delay" in limit_drng:
        if row == 1:
            clip_drng = True
    if "fringe_rate" in limit_drng:
        if column == 1:
            clip_drng = True
            
    if not clip_drng:
        vmin, vmax = None, None
        
    if mode=='log' and cbar_label is None:
        cbar_label = vis_label 
    elif mode=='log' and cbar_label is not None:
        cbar_label = cbar_label+' '+vis_label
    elif mode=='phase':
        cbar_label="Phase [rad]"

    fig.sca(ax)
    cax = waterfall(
        use_data, extent=extent, mode=mode, vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    if baseline is not None and row == 1:
        ax.axvline(horizon, color=horizon_color, ls='--')
        ax.axvline(-horizon, color=horizon_color, ls='--')
    
    if colorbar_flag==True:
        _ = plt.colorbar(cax)
        _.set_label(cbar_label,c='black')
        _.ax.tick_params(axis='y',which='both', color='black', labelcolor='black')
    return cax