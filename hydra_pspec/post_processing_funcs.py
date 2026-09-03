import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import json
import sys

from pathlib import Path
from jsonargparse import Namespace
from jsonargparse.typing import Path_fr
from pprint import pprint
from pyuvdata import UVData
from astropy import units
from astropy.units import Quantity
from scipy.signal.windows import blackmanharris
from scipy.signal import fftconvolve
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm

try:
    from hydra_pspec.utils import fourier_operator, form_pseudo_stokes_vis
    from hydra_pspec.pspec import covariance_from_pspec
except ImportError:
    print(f"WARNING: no hydra_pspec module found!")


from rich.console import Console
cns = Console()
try:
    _ = get_ipython()
    interactive = True
    from rich.jupyter import print as jprint
except NameError:
    interactive = False

def cprint(*args, highlight=False, soft_wrap=True, **kwargs):
    if interactive:
        jprint(*args, highlight=highlight, soft_wrap=soft_wrap, **kwargs)
    else:
        cns.print(*args, highlight=highlight, soft_wrap=soft_wrap, **kwargs)


def format_time(t):
    if not isinstance(t, Quantity):
        t = Quantity(t, unit='s')
    if t.to('h').value > 1:
        t = t.to('h')
    elif t.to('min').value > 1:
        t = t.to('min')
    elif t.to('ms').value < 1:
        t = t.to('us')
    elif t.to('s').value < 1:
        t = t.to('ms')
        
    return t


def dict_key_print(d, i=0, indent='| '):
    """
    Recursively print dictionary keys.
    
    Parameters
    ----------
    d : dict
        Dictionary.  Can be a nested dictionary.
    i : int
        Current level of nesting.  Defaults to 0.
    indent : str or int
        Indentation "character" to use in print.  If an integer, uses `indent`
        spaces. If a str, print `i` copies of `indent`.
    """
    if not isinstance(d, dict):
        return

    for key in d.keys():
        if isinstance(indent, int):
            buffer = ' '*(i * indent)
        elif isinstance(indent, str):
            buffer = indent * i
        print(buffer + f'{key}')
        dict_key_print(d[key], i=i+1)


def get_freqs_lsts_from_uvd(uvd):
    if uvd.future_array_shapes:
        spw_ind = None
    else:
        spw_ind = 0

    freqs = uvd.freq_array[spw_ind] * units.Hz
    df = freqs[1] - freqs[0]
    extent_freq = [
        (freqs.min() - df/2).to('MHz').value,
        (freqs.max() + df/2).to('MHz').value
    ]

    lsts = np.unique(uvd.lst_array) * 12 / np.pi * units.h
    dlst = lsts[1] - lsts[0]
    extent_lst = [
        (lsts.min() - dlst/2).to('h').value,
        (lsts.max() + dlst/2).to('h').value
    ]

    return freqs, extent_freq, lsts, extent_lst


def weighted_quantile(data, q, weights=None):
    """
    Compute the weighted quantile from a set of data and weights.

    Parameters
    ----------
    data : array-like
        Input (one-dimensional) array.
    q : float
        Quantile in [0, 1].
    weights : array-like
        Array of weights with shape matching `data`.  If no weights are passed,
        the data are assumed to be uniformly weighted.

    Returns
    -------
    quantile : float
        Quantile value.

    """
    if q < 0 or q > 1:
        raise ValueError("q must be in [0, 1]")
    if weights is None:
        weights = np.ones(d.size)
    if isinstance(data, Quantity):
        data_unit = data.unit
        data = data.value
    else:
        data_unit = None
    if isinstance(weights, Quantity):
        weights_unit = weights.unit
        weights = weights.value
    sort_inds = np.argsort(data)
    d = data[sort_inds]
    w = weights[sort_inds]
    cdf_w = np.cumsum(w) / np.sum(w)
    quantile = np.interp(q, cdf_w, d)
    if data_unit is not None:
        quantile = Quantity(quantile, unit=data_unit)
    return quantile


def delay_power_spectrum(
    vis, freqs, mean_sub=True, taper=True, norm=False,
    vis_unit='Jy', axis=1
):
    if not isinstance(freqs, Quantity):
        freqs = Quantity(freqs, unit='Hz')
    df = np.diff(freqs).mean()
    
    if not isinstance(vis, Quantity):
        vis = Quantity(vis.copy(), unit=vis_unit)
    else:
        vis = vis.copy()
    Nfreqs = vis.shape[axis]
    
    if taper:
        shape_inds = tuple(np.delete(np.arange(len(vis.shape)), axis))
        taper = np.expand_dims(blackmanharris(Nfreqs), axis=shape_inds)
        vis *= taper
    if mean_sub:
        mean = np.expand_dims(vis.mean(axis=axis), axis=axis)
        vis -= mean
    
    axes = (axis,)
    fft_vis = np.fft.ifftshift(vis, axes=axes)
    fft_vis = np.fft.fftn(fft_vis, axes=axes)
    fft_vis = np.fft.fftshift(fft_vis, axes=axes)
    if norm:
        fft_vis *= df.to('Hz')
    
    dps = np.abs(fft_vis)**2
    if norm:
        dps /= (df.to('Hz') * Nfreqs)
    
    delays = np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=df.to('1/s')))
    delays = delays.to('ns')
    
    return fft_vis, dps, delays


def calc_frac_diff(
    x, y, x_lbound=None, x_ubound=None, y_lbound=None, y_ubound=None
):
    """
    Compute the fractional difference between x and y with/without errorbars.
    
    Computed as `x/y - 1`, i.e. the fractional difference of x relative to y.

    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    use_x_err = x_lbound is not None and x_ubound is not None
    use_y_err = y_lbound is not None and y_ubound is not None
    
    fd = x/y - 1
    
    if use_x_err or use_y_err:
        # 0 index: lower bound
        # 1 index: higher bound
        fd_err = np.zeros((2,) + fd.shape)

    if use_x_err:
        # Calculate relative uncertainties from upper and lower bounds
        frac_err_xp = x_ubound / x - 1  # sigma_x^+ / x
        frac_err_xm = 1 - x_lbound / x  # sigma_x^- / x

    if use_y_err:
        # Calculate relative uncertainties from upper and lower bounds
        frac_err_yp = y_ubound / y - 1  # sigma_y^+ / y
        frac_err_ym = 1 - y_lbound / y  # sigma_y^- / y

    if use_x_err and use_y_err:
        # Minimum fractional difference occurs at (x^-, y^+) with
        # x^- = x - sigma_x^- (lower bound of x)
        # y^+ = y + sigma_y^+ (upper bound of y)
        fd_err[0] = x / y * (frac_err_xm + frac_err_yp)
        # Maximum fractional difference occurs at (x^+, y^-) with
        # x^+ = x + sigma_x^+ (upper bound of x)
        # y^- = y - sigma_y^- (lower bound of y)
        fd_err[1] = x / y * (frac_err_xp + frac_err_ym)
    elif use_x_err:
        # Minimum fractional difference occurs at x^-
        fd_err[0] = x / y * frac_err_xm
        # Maximum fractional difference occurs at x^+
        fd_err[1] = x / y * frac_err_xp
    elif use_y_err:
        # Minimum fractional difference occurs at y^+
        fd_err[0] = x / y * frac_err_yp
        # Maximum fractional difference occurs at y^-
        fd_err[1] = x / y * frac_err_ym
    
    if use_x_err or use_y_err:
        return fd, fd_err
    else:
        return fd


def get_errorbars_from_conf_interval(lbound, mean, ubound, desc="Errorbars"):
    """
    Compute upper/lower errorbars from a lower bound, mean, and upper bound.

    Parameters
    ----------
    lbound : array_like
        Lower bounds with shape `(Ndata,)`.
    mean : array_like
        Means with shape `(Ndata,)`
    ubound : array_like
        Upper bounds with shape `(Ndata,)`
    desc : str
        Description to add to print statements if negative errorbars are found.

    Returns
    -------
    errorbars : `numpy.ndarray`
        2D array with shape `(2, Ndata)`.  The first row (`errorbars[0]`)
        contains the lower errorbars.  The second row (`errorbars[1]`) contains
        the upper errorbars.  Row indexing chosen for compatibility with
        `matplotlib.pyplot.errorbar`.
    """
    Ndata = lbound.size
    errorbars = np.zeros((2, Ndata))

    # Lower errorbars
    errorbars[0] = mean - lbound
    if np.any(errorbars[0] < 0):
        print(f'{desc}: Negative values found in lower errorbar')
        inds = np.where(errorbars[0] < 0)[0]
        pprint(inds)
        pprint(errorbars[0, inds])
        errorbars[0, inds] = np.abs(errorbars[0, inds])

    # Upper errorbars
    errorbars[1] = ubound - mean
    if np.any(errorbars[1] < 0):
        print(f'{desc}: Negative values found in lower errorbar')
        inds = np.where(errorbars[1] < 0)[0]
        pprint(inds)
        pprint(errorbars[0, inds])
        errorbars[1, inds] = np.abs(errorbars[1, inds])

    return errorbars


def get_data_and_plot(
    res_path,
    vis_paths,
    samples_key="samples",
    pI_norm=1.0,
    Nprior_inds=0,
    Nburn=0,
    fd_ylim=(-0.35, 0.35),
    suptitle=None,
    verbose=False
):
    """
    Load all necessary data to make a summary plot a la `summary_plot`.

    """
    # Load data
    vis_data, hp_data = get_data(
        res_path,
        vis_paths,
        samples_key=samples_key,
        pI_norm=pI_norm,
        Nburn=Nburn,
        verbose=verbose
    )
    
    # Make summary plot
    print('Plotting...')
    fig = summary_plot(
        vis_data, hp_data, Nprior_inds=Nprior_inds, suptitle=suptitle,
        fd_ylim=fd_ylim, tapered=hp_data['args'].taper
    )

    if hp_data['args'].n_ps_prior_bins > 0:
        plot_posteriors_w_priors(
            vis_data['delays'], hp_data, suptitle=suptitle,
            tapered=hp_data['args'].taper
        )
    
    return vis_data, hp_data, fig


def get_data(
    res_path,
    vis_paths,
    samples_key="samples",
    gcr_key="signal_cr",
    dps_key="signal_ps",
    fg_key="fg_amps",
    vis_unit="Jy",
    norm=False,
    taper=False,
    freq_axis=-1,
    Nburn=0,
    conf_intervals=[68, 95],
    frequencies=None,
    pI_norm=1.0,
    verbose=False
):
    # hydra-pspec data
    # Loads samples arrays and computes delay power spectra of Gaussian
    # Constrained Realizations (GCRs) and confidence intervals of GCRs and
    # Time-Averaged Delay Power Spectrum (TADPS) samples.
    hp_data = get_hp_data(
        res_path,
        samples_key=samples_key,
        gcr_key=gcr_key,
        dps_key=dps_key,
        fg_key=fg_key,
        vis_unit=vis_unit,
        norm=norm,
        taper=taper,
        freq_axis=freq_axis,
        Nburn=Nburn,
        conf_intervals=conf_intervals,
        freqs=frequencies,
        verbose=verbose
    )
    print()
    
    # Visibility data (inputs to hydra-pspec)
    # Loads and computes delay power spectra of the input visibilities.
    if verbose:
        print('Loading and processing visibilities...')
    vis_data = get_vis_data(
        vis_paths,
        ant_str=hp_data['args'].ant_str,
        frequencies=frequencies,
        pI_norm=pI_norm,
        norm=norm
    )

    return vis_data, hp_data


def get_vis_data(
    file_paths,
    ant_str=None,
    frequencies=None,
    pI_norm=1.0,
    norm=False
):
    """
    Load visibilities used in analysis and compute their delay spectra.
    
    Parameters
    ----------
    file_paths : dict
        Dictionary containing file paths to visibilities.
    ant_str : str
        Antenna string parsable by `pyuvdata.uvdata.UVData.select`.
    frequencies : array_like
        Frequencies to load.  Must be frequencies in the data file.
    pI_norm : float
        Pseudo stokes I normalization.  Scales the sum of XX and YY
        polarizations such as pI_norm * (XX + YY).  Defaults to 1.0.
    norm : bool
        If True, use dimensional normalizations in delay spectra calculations.
        Defaults to False.

    Returns
    -------
    data : dict
        Dictionary containing visibilities, their delay transforms, and
        metadata (frequencies and times).

    """
    vis = {}  # container for visibilities
    keys = file_paths.keys()
    bl = tuple([int(ant) for ant in ant_str.split('_')])
    
    # --- Visibilities ---    
    vis_unit = None
    for key in keys:
        if str(file_paths[key]).endswith(('.uvh5', '.uvfits')):
            uvd = UVData()
            uvd.read(file_paths[key], ant_str=ant_str, frequencies=frequencies)
            uvd.conjugate_bls()
            uvd = form_pseudo_stokes_vis(uvd, convention=pI_norm)
            vis[key] = uvd.get_data(bl + ("xx",), force_copy=True)
            if vis_unit is None:
                vis_unit = uvd.vis_units
        elif str(file_paths[key]).endswith('.npy'): 
            vis[key] = np.load(file_paths[key])
    if np.all([key in keys for key in ['sum', 'noise']]):
        # Sum of EoR + FG + noise
        vis['sum_noise'] = vis['sum'] + vis['noise']
            
    # Get metadata
    freqs = Quantity(uvd.freq_array[0], unit='Hz')
    df = freqs[1] - freqs[0]
    extent_freq = [
        (freqs.min() - df/2).to('MHz').value,
        (freqs.max() + df/2).to('MHz').value
    ]
    lsts = Quantity(np.unique(uvd.lst_array * 12 / np.pi), unit='h')
    dlst = np.diff(lsts).mean()
    extent_lst = [
        (lsts.min() - dlst/2).to('h').value,
        (lsts.max() + dlst/2).to('h').value
    ]
    
    # --- Delay Spectra ---
    # Containers for FFT delay spectra
    ds_fft = {}
    dps_fft = {}
    dps_fft_tavg = {}  # Time-averaged (tavg)
    # Containers for mean subtracted (ms) and tapered (tp) delay spectra
    ds_tp = {}
    dps_tp = {}
    dps_tp_tavg = {}  # Time-averaged (tavg)
    
    for key in keys:
        # FFT delay spectra
        ds_fft[key], dps_fft[key], delays = delay_power_spectrum(
            vis[key], freqs, vis_unit=vis_unit,
            mean_sub=False, taper=False, norm=norm
        )
        dps_fft_tavg[key] = dps_fft[key].mean(axis=0)
        # Mean-subtracted and tapered delay spectra
        ds_tp[key], dps_tp[key], _ = delay_power_spectrum(
            vis[key], freqs, vis_unit=vis_unit,
            mean_sub=False, taper=True, norm=norm
        )
        dps_tp_tavg[key] = dps_tp[key].mean(axis=0)
    if np.all([key in keys for key in ['sum', 'noise']]):
        # Delay spectrum of EoR + FG + noise
        ds_fft['sum_noise'], dps_fft['sum_noise'], _ = delay_power_spectrum(
            vis['sum'] + vis['noise'], freqs, vis_unit=vis_unit,
            mean_sub=False, taper=False, norm=norm
        )
        dps_fft_tavg['sum_noise'] = dps_fft['sum_noise'].mean(axis=0)
        ds_tp['sum_noise'], dps_tp['sum_noise'], _ = delay_power_spectrum(
            vis['sum'] + vis['noise'], freqs, vis_unit=vis_unit,
            mean_sub=False, taper=True, norm=norm
        )
        dps_tp_tavg['sum_noise'] = dps_tp['sum_noise'].mean(axis=0)
    
    ddelay = delays[1] - delays[0]
    extent_delay = [
        (delays.min() - ddelay/2).to('ns').value,
        (delays.max() + ddelay/2).to('ns').value
    ]
    
    data = {}
    data['freqs'] = freqs
    data['df'] = df
    data['extent_freq'] = extent_freq
    data['delays'] = delays
    data['ddelay'] = ddelay
    data['extent_delay'] = extent_delay
    data['lsts'] = lsts
    data['dlst'] = dlst
    data['extent_lst'] = extent_lst
    data['vis'] = vis
    data['ds'] = {'fft': ds_fft, 'tp': ds_tp}
    data['dps'] = {
        'fft': dps_fft,
        'fft_tavg': dps_fft_tavg,
        'tp': dps_tp,
        'tp_tavg': dps_tp_tavg
    }
    
    return data


def get_hp_data(
    res_path,
    post_as_weights=False,
    fftshift=False,
    samples_key='samples',
    gcr_key='signal_cr',
    dps_key='signal_ps',
    fg_key='fg_amps',
    vis_unit='Jy',
    norm=False,
    taper=None,
    iter_axis=0,
    time_axis=1,
    freq_axis=2,
    Nburn=0,
    conf_intervals=[68, 95],
    freqs=None,
    verbose=False
):
    """
    Extract delay spectra and confidence intervals from hydra-pspec outputs.
    
    Parameters
    ----------
    res_path : str or Path
        Path to directory or file (deprecated) containing analysis results from
        hydra-pspec.
    post_as_weights : bool
        If True, use the posterior to weight averages over the iteration axis.
        Defaults to False, i.e. calculate the sample mean.
    fftshift : bool
        Apply `numpy.fft.fftshift` to the delay axis of the data.  Required
        for old versions of the code which had the delay=0 mode as the 0th
        index entry of the delay axis.
    samples_key : str
        Key for accessing the samples from hydra-pspec in the old dictionary
        output format.
    gcr_key : str
        Key for Gaussian Constrained Realizations (GCRs) in `res`.  Defaults
        to 'signal_cr'.
    dps_key : str
        Key for Time-Averaged Delay Power Spectrum (TADPS) samples in `res`.
        Defaults to 'signal_ps'.
    fg_key : str
        Key for the foreground amplitude samples.  Defaults to `fg_amps`.
    vis_unit : str
        Visibility units for GCRs.  Defaults to 'Jy'.
    norm : bool
        If True, use dimensional normalizations in delay spectra calculations.
        Defaults to False.
    taper : bool
        If True, the sampler outputs are assumed to be tapered.
    iter_axis : int
        Axis along which the iterations are stored.  Defaults to 0.
    time_axis : int
        Axis along which the times are stored.  Defaults to 1.
    freq_axis : int
        Axis along which to compute the delay spectrum, i.e. the index of the
        frequency axis.  Defaults to 2.
    Nburn : int
        Number of "burn in" iterations to ignore in chains.  Defaults to 0,
        i.e. use entire chain.
    verbose : bool
        If `verbose` is True, print information and timing statements.
        Otherwise, execute silently.

    Returns
    -------
    data : dict
        Dictionary containing delay spectra and their confidence intervals.

    """    
    if not isinstance(res_path, Path):
        res_path = Path(res_path)

    # Load hydra-pspec output dictionary
    if verbose:
        print('Loading hydra-pspec results...')
    if res_path.is_file():
        # Old file structure (dictionary)
        res = np.load(res_path, allow_pickle=True).item()
        git_info = res['git']
        args = res['args']
        res = res[samples_key]
    else:
        # New file structure (directory)
        with open(res_path / 'git.json', 'r') as f:
            git_info = json.load(f)
        with open(res_path / 'args.json', 'r') as f:
            args = Namespace(**json.load(f))
        res = {
            gcr_key: np.load(res_path / 'gcr-eor.npy'),
            fg_key: np.load(res_path / 'fg-amps.npy'),
            dps_key: np.load(res_path / 'dps-eor.npy'),
            'chisq': np.load(res_path / 'chisq.npy'),
            'ln_post': np.load(res_path / 'ln-post.npy')
        }
    if verbose:
        print()
        
        print('Git Info\n' + '-'*len('git info'))
        pprint(git_info)
        print()
        print('Analysis Parameters\n' + '-'*len('Analysis Parameters'))
        pprint(args.__dict__)
        print()

    if taper is None:
        taper = 'taper' in args
        if taper:
            # If taper in args but False, set to False
            # This seems redundant but is important for backward compatibility
            taper = args.taper
        else:
            args.taper = False
    
    if not res[fg_key].shape[iter_axis] == args.Niter:
        cprint(
            f"[bold]WARNING:[/bold] Expected {args.Niter} iterations "
            f"but found {res[fg_key].shape[iter_axis]} iterations"
        )
        print()
    
    if verbose:
        print('Processing hydra-pspec data...')
    if freqs is None:
        if verbose:
            print("Getting frequencies from `args.file_paths`...")
        uvd = UVData()
        if isinstance(args.file_paths[0], Path_fr):
            fp = args.file_paths[0].absolute
        elif isinstance(args.file_paths[0], Path):
            fp = args.file_paths[0].absolute()
        else:
            fp = str(args.file_paths[0])
        uvd.read(fp, read_data=False)
        freqs = Quantity(uvd.freq_array[0], unit='Hz')
        # Prevent UVData warnings from mucking up the print formatting below
        sys.stderr.flush()
    else:
        if not isinstance(freqs, Quantity):
            freqs = Quantity(freqs, unit='Hz')
    Nfreqs = freqs.size
    
    # Gaussian Constrained Realizations (GCRs) of the EoR
    gcrs = res[gcr_key]
    # Delay Power Spectrum (DPS) samples
    dpss = res[dps_key]

    # GCRs of the FGs
    if 'fg_basis' in args:
        # Foreground model comprised of analytic functions
        if args.fg_basis.lower() == 'legendre':
            poly_func = scipy.special.legendre
        elif args.fg_basis.lower() == 'hermite':
            poly_func = scipy.special.hermite
        elif args.fg_basis.lower() == 'chebyshev':
            poly_func = scipy.special.chebyu
        # fg_model_vecs should have shape (Nfreqs, Nfgmodes)
        fg_model_vecs = np.array([
            poly_func(i)(np.linspace(-1., 1., Nfreqs))
            for i in range(args.Nfgmodes)
        ]).T
    else:
        bl_str = args.ant_str.replace("_", "-")
        # Load foreground model basis vectors from disk
        if 'fg_eig_dir' in args:
            # Legacy command line argument name (deprecated)
            fg_model_dir = Path(args.fg_eig_dir)
            fg_model_path = fg_model_dir / bl_str / args.fg_eig_file
            fg_model_vecs = np.load(fg_model_path)
            fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]
        else:
            if Path(args.fgmodes).is_dir():
                fg_model_dir = Path(args.fgmodes)
                fg_model_file = args.fgmodes_file
                if bl_str is not None:
                    fg_model_path = fg_model_dir / bl_str / fg_model_file
                else:
                    fg_model_path = fg_model_dir / fg_model_file
            else:
                fg_model_path = Path(args.fgmodes)
            fg_model_vecs = np.load(fg_model_path)
            fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]

    fg_amps = res[fg_key]  # shape (Niter, Nfreqs, Nfgmodes)
    fgs = np.einsum('ijk,kl->ijl', fg_amps, fg_model_vecs.T)

    # Get posterior-weighted mean quantities
    if 'ln_post' in res and post_as_weights:
        weights = res['ln_post']
    else:
        weights = np.ones(gcrs.shape[iter_axis])
    gcrs_mean = np.average(
        gcrs[Nburn:], weights=weights[Nburn:], axis=iter_axis
    )
    dpss_mean = np.average(
        dpss[Nburn:], weights=weights[Nburn:], axis=iter_axis
    )
    fgs_mean = np.average(
        fgs[Nburn:], weights=weights[Nburn:], axis=iter_axis
    )

    # Containers for FFT delay spectra (ds) and delay power spectra (dps)
    ds_fft = {}
    dps_fft = {}
    dps_fft_tavg = {}  # Time-averaged (tavg)
    # Containers for mean subtracted (ms) and tapered (tp) delay spectra
    ds_tp = {}
    dps_tp = {}
    dps_tp_tavg = {}  # Time-averaged (tavg)

    if verbose:
        print('Computing delay spectra...', end=' ')
        start = time.time()
    
    # Gaussian Constrained Realizations (GCRs)
    # GCR array has shape (Niter, Ntimes, Nfreqs)
    if not taper:
        # --- EoR ---
        # FFT delay spectra
        ds_fft['gcr'], dps_fft['gcr'], _ = delay_power_spectrum(
            gcrs, freqs, vis_unit=vis_unit,
            mean_sub=False, taper=False, norm=norm, axis=freq_axis
        )
        dps_fft_tavg['gcr'] = dps_fft['gcr'].mean(axis=time_axis)
        dps_fft_tavg['gcr_mean'] = np.average(
            dps_fft_tavg['gcr'], weights=weights, axis=iter_axis
        )
        # Tapered delay spectra
        ds_tp['gcr'], dps_tp['gcr'], _ = delay_power_spectrum(
            gcrs, freqs, vis_unit=vis_unit,
            mean_sub=False, taper=True, norm=norm, axis=freq_axis
        )
        dps_tp_tavg['gcr'] = dps_tp['gcr'].mean(axis=time_axis)

        # --- FGs ---
        # FFT delay spectra
        ds_fft['fgs'], dps_fft['fgs'], _ = delay_power_spectrum(
            fgs, freqs, vis_unit=vis_unit,
            mean_sub=False, taper=False, norm=norm, axis=freq_axis
        )
        dps_fft_tavg['fgs'] = dps_fft['fgs'].mean(axis=time_axis)
        dps_fft_tavg['fgs_mean'] = np.average(
            dps_fft_tavg['fgs'], weights=weights, axis=iter_axis
        )
        # Tapered delay spectra
        ds_tp['fgs'], dps_tp['fgs'], _ = delay_power_spectrum(
            fgs, freqs, vis_unit=vis_unit,
            mean_sub=False, taper=True, norm=norm, axis=freq_axis
        )
        dps_tp_tavg['fgs'] = dps_tp['fgs'].mean(axis=time_axis)
    else:
        # The GCR solutions are derived from the tapered data and therefore
        # do not need to be tapered.  Tapering the GCR solutions in this case
        # would result in multipliclation by the taper squared.

        # --- EoR ---
        # Tapered delay spectra
        ds_tp['gcr'], dps_tp['gcr'], _ = delay_power_spectrum(
            gcrs, freqs, vis_unit=vis_unit,
            mean_sub=False, taper=False, norm=norm, axis=freq_axis
        )
        dps_tp_tavg['gcr'] = dps_tp['gcr'].mean(axis=time_axis)

        # --- FGs ---
        # Tapered delay spectra
        ds_tp['fgs'], dps_tp['fgs'], _ = delay_power_spectrum(
            fgs, freqs, vis_unit=vis_unit,
            mean_sub=False, taper=False, norm=norm, axis=freq_axis
        )
        dps_tp_tavg['fgs'] = dps_tp['fgs'].mean(axis=time_axis)

    dps_tp_tavg['gcr_mean'] = np.average(
        dps_tp_tavg['gcr'], weights=weights, axis=iter_axis
    )
    dps_tp_tavg['fgs_mean'] = np.average(
        dps_tp_tavg['fgs'], weights=weights, axis=iter_axis
    )
    
    # Time-Averaged Delay Power Spectra (TADPS)
    if freq_axis > 1:
        axes = (freq_axis - 1,)
    else:
        axes = (freq_axis,)
    if taper:
        if fftshift:
            dps_tp_tavg['dps'] = np.fft.fftshift(dpss, axes=axes)
            dps_tp_tavg['dps_mean'] = np.fft.fftshift(dpss_mean, axes=axes)
        else:
            dps_tp_tavg['dps'] = dpss
            dps_tp_tavg['dps_mean'] = dpss_mean
    else:
        if fftshift:
            dps_fft_tavg['dps'] = np.fft.fftshift(dpss, axes=axes)
            dps_fft_tavg['dps_mean'] = np.fft.fftshift(dpss_mean, axes=axes)
        else:
            dps_fft_tavg['dps'] = dpss
            dps_fft_tavg['dps_mean'] = dpss_mean
    dps_unit = vis_unit + '^2'
    if norm:
        dps_unit += ' Hz'
    if taper:
        dps_tp_tavg['dps'] = Quantity(dps_tp_tavg['dps'], unit=dps_unit)
        dps_tp_tavg['dps_mean'] = Quantity(
            dps_tp_tavg['dps_mean'], unit=dps_unit
        )
    else:
        dps_fft_tavg['dps'] = Quantity(dps_fft_tavg['dps'], unit=dps_unit)
        dps_fft_tavg['dps_mean'] = Quantity(
            dps_fft_tavg['dps_mean'], unit=dps_unit
        )
    
    if verbose:
        print(f'({format_time(time.time() - start):.2f})')
    
    
    # Confidence intervals
    # GCRs
    if not taper:
        gcr_fft_cis = {}
        fgs_fft_cis = {}
    gcr_tp_cis = {}
    fgs_tp_cis = {}
    # TADPS
    if taper:
        dps_tp_cis = {}
    else:
        dps_fft_cis = {}
    
    if verbose:
        print('Computing confidence intervals...', end=' ')
        start = time.time()
    
    for conf_interval in conf_intervals:
        percentile = conf_interval/2 + 50

        # Container for tapered GCR DPS
        # EoR
        gcr_tp_bounds = {}
        gcr_tp_bounds['lbound'] = Quantity(
            np.zeros(gcrs.shape[freq_axis]), unit=dps_unit
        )
        gcr_tp_bounds['ubound'] = Quantity(
            np.zeros(gcrs.shape[freq_axis]), unit=dps_unit
        )
        # FGs
        fgs_tp_bounds = {}
        fgs_tp_bounds['lbound'] = Quantity(
            np.zeros(fgs.shape[freq_axis]), unit=dps_unit
        )
        fgs_tp_bounds['ubound'] = Quantity(
            np.zeros(fgs.shape[freq_axis]), unit=dps_unit
        )
        if taper:
            # Container for tapered DPS samples
            # EoR
            dps_tp_bounds = {}
            dps_tp_bounds['lbound'] = Quantity(
                np.zeros(gcrs.shape[freq_axis]), unit=dps_unit
            )
            dps_tp_bounds['ubound'] = Quantity(
                np.zeros(gcrs.shape[freq_axis]), unit=dps_unit
            )
        else:
            # --- EoR ---
            # Container for untapered GCR DPS
            gcr_fft_bounds = {}
            gcr_fft_bounds['lbound'] = Quantity(
                np.zeros(gcrs.shape[freq_axis]), unit=dps_unit
            )
            gcr_fft_bounds['ubound'] = Quantity(
                np.zeros(gcrs.shape[freq_axis]), unit=dps_unit
            )
            # Container for untapered DPS samples
            dps_fft_bounds = {}
            dps_fft_bounds['lbound'] = Quantity(
                np.zeros(gcrs.shape[freq_axis]), unit=dps_unit
            )
            dps_fft_bounds['ubound'] = Quantity(
                np.zeros(gcrs.shape[freq_axis]), unit=dps_unit
            )
            # --- FGs ---
            # Container for untapered GCR DPS
            fgs_fft_bounds = {}
            fgs_fft_bounds['lbound'] = Quantity(
                np.zeros(fgs.shape[freq_axis]), unit=dps_unit
            )
            fgs_fft_bounds['ubound'] = Quantity(
                np.zeros(fgs.shape[freq_axis]), unit=dps_unit
            )

        # Get weighted percentiles for each delay bin
        for i_dly in range(gcrs.shape[freq_axis]):
            # EoR
            gcr_tp_bounds['lbound'][i_dly] = weighted_quantile(
                dps_tp_tavg['gcr'][Nburn:, i_dly], 1-percentile/100,
                weights=weights[Nburn:]
            )
            gcr_tp_bounds['ubound'][i_dly] = weighted_quantile(
                dps_tp_tavg['gcr'][Nburn:, i_dly], percentile/100,
                weights=weights[Nburn:]
            )
            # FGs
            fgs_tp_bounds['lbound'][i_dly] = weighted_quantile(
                dps_tp_tavg['fgs'][Nburn:, i_dly], 1-percentile/100,
                weights=weights[Nburn:]
            )
            fgs_tp_bounds['ubound'][i_dly] = weighted_quantile(
                dps_tp_tavg['fgs'][Nburn:, i_dly], percentile/100,
                weights=weights[Nburn:]
            )

            if taper:
                dps_tp_bounds['lbound'][i_dly] = weighted_quantile(
                    dps_tp_tavg['dps'][Nburn:, i_dly], 1-percentile/100,
                    weights=weights[Nburn:]
                )
                dps_tp_bounds['ubound'][i_dly] = weighted_quantile(
                    dps_tp_tavg['dps'][Nburn:, i_dly], percentile/100,
                    weights=weights[Nburn:]
                )
            else:
                # --- EoR ---
                gcr_fft_bounds['lbound'][i_dly] = weighted_quantile(
                    dps_fft_tavg['gcr'][Nburn:, i_dly], 1-percentile/100,
                    weights=weights[Nburn:]
                )
                gcr_fft_bounds['ubound'][i_dly] = weighted_quantile(
                    dps_fft_tavg['gcr'][Nburn:, i_dly], percentile/100,
                    weights=weights[Nburn:]
                )

                dps_fft_bounds['lbound'][i_dly] = weighted_quantile(
                    dps_fft_tavg['dps'][Nburn:, i_dly], 1-percentile/100,
                    weights=weights[Nburn:]
                )
                dps_fft_bounds['ubound'][i_dly] = weighted_quantile(
                    dps_fft_tavg['dps'][Nburn:, i_dly], percentile/100,
                    weights=weights[Nburn:]
                )

                # --- FGs ---
                fgs_fft_bounds['lbound'][i_dly] = weighted_quantile(
                    dps_fft_tavg['fgs'][Nburn:, i_dly], 1-percentile/100,
                    weights=weights[Nburn:]
                )
                fgs_fft_bounds['ubound'][i_dly] = weighted_quantile(
                    dps_fft_tavg['fgs'][Nburn:, i_dly], percentile/100,
                    weights=weights[Nburn:]
                )
        
        gcr_tp_cis[conf_interval] = gcr_tp_bounds
        fgs_tp_cis[conf_interval] = fgs_tp_bounds
        if taper:
            dps_tp_cis[conf_interval] = dps_tp_bounds
        else:
            gcr_fft_cis[conf_interval] = gcr_fft_bounds
            dps_fft_cis[conf_interval] = dps_fft_bounds
            fgs_fft_cis[conf_interval] = fgs_fft_bounds

    if verbose:
        print(f'({format_time(time.time() - start):.2f})')
    
    if not taper:
        dps_fft_tavg['gcr_ci'] = gcr_fft_cis
        dps_fft_tavg['fgs_ci'] = fgs_fft_cis
    dps_tp_tavg['gcr_ci'] = gcr_tp_cis
    dps_tp_tavg['fgs_ci'] = fgs_tp_cis
    if taper:
        dps_tp_tavg['dps_ci'] = dps_tp_cis
    else:
        dps_fft_tavg['dps_ci'] = dps_fft_cis
    
    ds = dict(
        fft=ds_fft,
        tp=ds_tp
    )
    dps = dict(
        fft=dps_fft,
        fft_tavg=dps_fft_tavg,
        tp=dps_tp,
        tp_tavg=dps_tp_tavg
    )
    
    mean_type = 'sample mean'
    if post_as_weights:
        mean_type = 'posterior-weighted ' + mean_type
    data = dict(
        args=args,
        mean_type=mean_type,
        gcr=gcrs,
        gcr_mean=gcrs_mean,
        fg_amps=fg_amps,
        fg_vecs=fg_model_vecs,
        fgs=fgs,
        fgs_mean=fgs_mean,
        ds=ds,
        dps=dps
    )
    if 'chisq' in res:
        data.update({'chisq': res['chisq']})
    if 'ln_post' in res:
        data.update({'ln_post': res['ln_post']})

    return data


def summary_plot(
    vis_data,
    hp_data,
    weight_by_posterior=False,
    suptitle=None,
    plot_fgs=False,
    Nprior_inds=0,
    plot_prior=False,
    conf_interval=95,
    plot_tapered=False,
    tapered=False,
    figsize=(12*0.7, 14*0.7*2/3),
    ps_ylim=None,
    fd_ylim=(-0.35, 0.35),
    gcr_color='C0',
    gcr_marker='',
    fg_color='C1',
    fg_marker='',
    ylabel_units=None,
    print_avg_fd=False
):
    """
    Create a figure comparing the input data and hydra-pspec samples with
    two subplots:
    
    1. Top subplot shows the tapered delay power spectra of the true EoR+FGs,
       the noise, and the mean and `conf_interval`% confidence interval of the
       Gaussian Constrained Realization (GCR) samples.

    3. Bottom subplot computes the fractional error of the time-averaged delay
       power spectrum samples (draws from the inverse gamma distribution) and 
       the GCRs relative to the true EoR only delay power spectrum.  The
       errorbars represent the `conf_interval`% confidence interval of the
       fractional error.

    Parameters
    ----------
    vis_data : dict
        Dictionary generated by `get_vis_data`.
    hp_data : dict
        Dictionary generated by `get_hp_data`.
    suptitle : str
        Figure suptitle.
    plot_fgs : bool
        If True, plot delay power spectra of the FG model.
    Nprior_inds : int
        Number of bins around delay=0 which are ignored in the errorbar
        calculation due to the prior placed on the power spectrum in
        hydra-pspec.  In total, 2*Nprior_inds bins are ignored, i.e. any bins
        which satisfy |\tau| <= Nprior_inds * \Delta\tau are ignored as these
        bins are dominated by FG emission and sampling from this region is
        computationally expensive due to the large dynamic range dynamic range
        between the EoR and FG signals.  Defaults to 0 which plots all
        errorbars.
    plot_prior : bool
        Plot the delay bins affected by the delay power spectrum priors as a
        grey shaded region.
    conf_interval : int
        Confidence interval to plot.  Must be a key within `hp_data`.
    plot_tapered : bool
        If True, plot tapered delay power spectra.  Defaults to False.
    tapered : bool
        If True, assume tapering has been used in the analysis.  The fractional
        difference plots will compared tapered delay power spectra.  Otherwise,
        compare untapered delay power spectra.
    figsize : array-like
        matplotlib figure size.
    ps_ylim : array-like
        y-axis limits for the delay power spectrum plot.
    fd_ylim : array-like
        y-axis limits for the fractional difference plot.
    gcr_color : str
        matplotlib compatible color string for the GCR data.
    gcr_marker : str
        matplotlib compatible marker string for the GCR data.
    ylabel_units : str
        Units to include in the y-axis label of the power spectrum plot.
    print_avg_fd : bool
        If True, print the weighted average fractional difference between the
        DPS samples and the true EoR DPS.  The weights are computed as
        `1 / (ubound + lbound)**2`.

    """
    delays = vis_data['delays']
    Nfreqs = delays.size
    delay_mask = np.ones(Nfreqs, dtype=bool)
    delay_mask[Nfreqs//2-Nprior_inds:Nfreqs//2+Nprior_inds] = False
    
    fig, axs = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={'hspace': 0, 'height_ratios':[1, 0.6]}
    )
    legend_kwargs = dict(
        frameon=True, fontsize=14, labelspacing=0.3, loc='upper right',
        framealpha=0.9
    )

    if plot_tapered:
        ps_key = 'tp_tavg'
    else:
        ps_key = 'fft_tavg'
    
    # Tapered delay power spectra subplot
    ax = axs[0]
    if not plot_fgs:
        ax.plot(
            delays.to('ns'),
            vis_data['dps'][ps_key]['sum'],
            label='True EoR + Foregrounds',
            color='k',
            ls='-',
            alpha=0.7
        )
    else:
        ax.plot(
            delays.to('ns'),
            vis_data['dps'][ps_key]['fgs'],
            label='True Foregrounds',
            color='k',
            ls='-',
            alpha=0.7
        )
    ax.plot(
        delays.to('ns'),
        vis_data['dps'][ps_key]['eor'],
        label='True EoR',
        color='k',
        ls=':',
        alpha=0.7
    )
    ax.plot(
        delays.to('ns'),
        vis_data['dps'][ps_key]['noise'],
        label='Injected noise',
        color='k',
        ls='-.'
    )
    ax.plot(
        delays.to('ns'),
        hp_data['dps'][ps_key]['gcr_mean'],
        label=f"EoR GCR {hp_data['mean_type']}",
        color=gcr_color,
        marker=gcr_marker,
        ls='-',
        alpha=1.0
    )
    ax.fill_between(
        delays.to('ns').value,
        hp_data['dps'][ps_key]['gcr_ci'][conf_interval]['lbound'].value,
        hp_data['dps'][ps_key]['gcr_ci'][conf_interval]['ubound'].value,
        label='EoR GCR 95% conf.',
        color=gcr_color,
        alpha=0.35
    )
    if plot_fgs:
        ax.plot(
            delays.to('ns'),
            hp_data['dps'][ps_key]['fgs_mean'],
            label=f"FG GCR {hp_data['mean_type']}",
            color=fg_color,
            marker=fg_marker,
            ls='-',
            alpha=1.0
        )
        ax.fill_between(
            delays.to('ns').value,
            hp_data['dps'][ps_key]['fgs_ci'][conf_interval]['lbound'].value,
            hp_data['dps'][ps_key]['fgs_ci'][conf_interval]['ubound'].value,
            label='FG GCR 95% conf.',
            color=fg_color,
            alpha=0.35
        )
    ax.set_yscale('log')
    if ps_ylim is not None:
        ax.set_ylim(ps_ylim)
    ylabel = r'$P(\tau)$'
    if ylabel_units is not None:
        ylabel += fr' [{ylabel_units}]'
    ax.set_ylabel(ylabel)
    ax.set_title(suptitle)

    if not tapered:
        ps_key = 'fft_tavg'
    else:
        ps_key = 'tp_tavg'

    # Fractional difference subplot
    ax = axs[1]
    # Time-averaged delay power spectrum sample confidence intervals
    fd, fd_err = calc_frac_diff(
        hp_data['dps'][ps_key]['dps_mean'],
        vis_data['dps'][ps_key]['eor'],
        x_lbound=hp_data['dps'][ps_key]['dps_ci'][conf_interval]['lbound'],
        x_ubound=hp_data['dps'][ps_key]['dps_ci'][conf_interval]['ubound']
    )
    fd_err = fix_neg_errbars(fd_err, message="DPS Samples vs True")
    ax.errorbar(
        delays.to('ns'),
        fd,
        yerr=fd_err,
        color='k',
        alpha=0.6,
        label=f'DPS samples {conf_interval}% conf.',
        elinewidth=1.5,
        capsize=1.5,
        fmt='.',
        zorder=1
    )
    if print_avg_fd:
        weights = 1 / (frac_diff_dps_ubound + frac_diff_dps_lbound)**2
        wavg = np.average(frac_diff_dps_mean, weights=weights)
        print(
            f'(Weighted) Average fractional difference = {wavg:.2e}'
        )
    # GCR confidence intervals
    fd, fd_err = calc_frac_diff(
        hp_data['dps'][ps_key]['gcr_mean'],
        vis_data['dps'][ps_key]['eor'],
        x_lbound=hp_data['dps'][ps_key]['gcr_ci'][conf_interval]['lbound'],
        x_ubound=hp_data['dps'][ps_key]['gcr_ci'][conf_interval]['ubound']
    )
    fd_err = fix_neg_errbars(fd_err, message="EoR GCR DPS vs True")
    ax.plot(
        delays.to('ns'),
        fd,
        color=gcr_color,
        marker=gcr_marker,
        ls='-',
        alpha=1.0,
        zorder=10
    )
    ax.fill_between(
        delays.to('ns').value,
        fd - fd_err[0],
        fd + fd_err[1],
        color=gcr_color,
        alpha=0.35,
        zorder=10
    )
    if plot_fgs:
        fd, fd_err = calc_frac_diff(
            hp_data['dps'][ps_key]['fgs_mean'],
            vis_data['dps'][ps_key]['fgs'],
            x_lbound=hp_data['dps'][ps_key]['fgs_ci'][conf_interval]['lbound'],
            x_ubound=hp_data['dps'][ps_key]['fgs_ci'][conf_interval]['ubound']
        )
        fd_err = fix_neg_errbars(fd_err, message="FG GCR DPS vs True")
        ax.plot(
            delays.to('ns'),
            fd,
            color=fg_color,
            marker=fg_marker,
            ls='-',
            alpha=1.0,
            zorder=0
        )
        ax.fill_between(
            delays.to('ns').value,
            fd - fd_err[0],
            fd + fd_err[1],
            color=fg_color,
            alpha=0.35,
            zorder=0
        )
    ax.axhline(0, color='k', ls='--', alpha=0.7)
    ax.set_ylabel(f'Fractional Difference')  #, labelpad=5)
    ax.set_ylim(fd_ylim)
    ax.set_yticks(ax.get_yticks()[1:-1])
    ax.set_xlabel(r'$\tau$ [ns]')

    for ax in axs:
        legend = ax.legend(**legend_kwargs)
        legend.set(zorder=100)
        ax.grid(which='major', ls=':')
        if plot_prior:
            args = hp_data['args']
            n_ps_prior_bins = args.n_ps_prior_bins
            ps_prior_lo = args.ps_prior_lo
            ps_prior_hi = args.ps_prior_hi
            Ndlys = delays.size
            delays_priors = delays[
                Ndlys//2 - n_ps_prior_bins : Ndlys//2 + n_ps_prior_bins + 1
            ]
            ax.axvspan(
                delays_priors[0].to('ns').value,
                delays_priors[-1].to('ns').value,
                color='0.5',
                alpha=0.2,
                label='Uniform prior',
                zorder=0
            )

    fig.tight_layout()
    
    plt.show()

    return fig


def compare_data(
    data_a,
    data_b,
    frac_diff=False,
    labels=None,
    sym_cbar=False,
    comp_type='re_im',
    plot_size=4,
    force_clim=False,
    suptitle=None,
    share_all=True,
    aspect=False,
    axes_pad=(1.0, 0.2),
    cbar_mode='each',
    cbar_pad=0.01,
    extent=None,
    origin='upper',
    ax_labels=None
):
    """
    Compare data via imshow.

    """
    funcs = []
    func_lbls = []
    cmaps = []
    if 're' in comp_type:
        funcs.append(np.real)
        func_lbls.append('Re')
        cmaps.append('viridis')
    if 'im' in comp_type:
        funcs.append(np.imag)
        func_lbls.append('Im')
        cmaps.append('magma')
    if 'abs' in comp_type:
        funcs.append(np.abs)
        func_lbls.append('Abs')
        cmaps.append('viridis')
    if 'ph' in comp_type:
        funcs.append(np.angle)
        func_lbls.append('Phase')
        cmaps.append('twilight')
    
    if labels is None:
        labels = [None] * 2
    if ax_labels is None:
        ax_labels = [None] * 2
    
    nrows = len(funcs)
    ncols = 3
    figsize = (ncols*plot_size, 0.75*nrows*plot_size)
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig, 111, (nrows, ncols), share_all=share_all, aspect=aspect,
        axes_pad=axes_pad, cbar_mode=cbar_mode, cbar_pad=cbar_pad
    )
    
    im_kwargs = dict(origin=origin, extent=extent, aspect='auto')
    zip_obj = zip(grid.axes_row, funcs, func_lbls, cmaps)
    for i_row, (axs, func, func_lbl, cmap) in enumerate(zip_obj):
        if func == np.angle:
            vmin = -np.pi
            vmax = np.pi
        elif force_clim and not sym_cbar:
            vmin = np.min((func(data_a).min(), func(data_b).min()))
            vmax = np.max((func(data_a).max(), func(data_b).max()))
        else:
            vmin = None
            vmax = None
        
        ax = axs[0]
        if i_row == 0:
            ax.set_title(labels[0])
        if sym_cbar:
            norm = CenteredNorm
            vmin = None
            vmax = None
            cmap = 'RdGy_r'
        else:
            norm = lambda: None
        im = ax.imshow(
            func(data_a), cmap=cmap, norm=norm(), vmin=vmin, vmax=vmax, **im_kwargs
        )
        cb = fig.colorbar(im, cax=ax.cax, label=func_lbl)
        
        ax = axs[1]
        if i_row == 0:
            ax.set_title(labels[1])
        im = ax.imshow(
            func(data_b), cmap=cmap, norm=norm(), vmin=vmin, vmax=vmax, **im_kwargs
        )
        cb = fig.colorbar(im, cax=ax.cax, label=func_lbl)
        
        ax = axs[2]
        if i_row == 0:
            if frac_diff:
                ax.set_title(f'1 - {labels[1]}/{labels[0]}')
            else:
                ax.set_title(f'{labels[0]} - {labels[1]}')
        if frac_diff:
            diff = 1 - func(data_b) / func(data_a)
        else:
            diff = func(data_a) - func(data_b)
        clim = np.abs(diff).max()
        im = ax.imshow(
            diff, cmap='RdBu_r', vmin=-clim, vmax=clim, **im_kwargs
        )
        cb = fig.colorbar(im, cax=ax.cax, label=func_lbl)
        
        for ax in axs:
            ax.set_xlabel(ax_labels[0])
            ax.set_ylabel(ax_labels[1])
        
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()

    return fig


def fraser_summary_plot(
    vis_data,
    hp_data,
    suptitle=None,
    Nprior_inds=0
):
    """
    Create a figure comparing the input data and hydra-pspec samples with
    three subplots:
    
    1. Top subplot shows the mean-subtracted and tapered delay power spectra of
       the true EoR+FGs, the noise, and the mean and 95% confidence interval of
       the Gaussian Constrained Realization (GCR) samples.

    2. Middle subplot computes the fractional error of the GCR delay power
       spectrum relative to the true EoR only delay power spectrum.  The
       errorbars represent the 68% confidence interval of the fractional error.
       This computation uses mean-subtracted and tapered delay power spectra.

    3. Bottom subplot computes the fractional error of the time-averaged delay
       power spectrum samples (draws from the inverse gamma distribution)
       relative to the true EoR only delay power spectrum.  The errorbars
       represent the 68% confidence interval of the fractional error.  This
       computation does _not_ use mean-subtracted or tapered delay power
       spectra, i.e. the delay transform is computed as an FFT of the frequency
       axis.

    Parameters
    ----------
    vis_data : dict
        Dictionary generated by `get_vis_data`.
    hp_data : dict
        Dictionary generated by `get_hp_data`.
    suptitle : str
        Figure suptitle.
    Nprior_inds : int
        Number of bins around delay=0 which are ignored in the errorbar
        calculation due to the prior placed on the power spectrum in
        hydra-pspec.  In total, 2*Nprior_inds bins are ignored, i.e. any bins
        which satisfy |\tau| <= Nprior_inds * \Delta\tau are ignored as these
        bins are dominated by FG emission and sampling from this region is
        computationally expensive due to the large dynamic range dynamic range
        between the EoR and FG signals.  Defaults to 0 which plots all
        errorbars.

    """
    delays = vis_data['delays']
    Nfreqs = delays.size
    delay_mask = np.ones(Nfreqs, dtype=bool)
    delay_mask[Nfreqs//2-Nprior_inds:Nfreqs//2+Nprior_inds] = False
    
    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=(12*0.7, 14*0.7),
        gridspec_kw={'hspace': 0, 'height_ratios':[1, 0.6, 0.6]}
    )
    legend_kwargs = dict(
        frameon=False, fontsize=14, labelspacing=0.3, loc='upper right',
        framealpha=0.9
    )
    
    # Mean-subtracted, tapered delay power spectra subplot
    ax = axs[0]
    ax.plot(
        delays.to('ns'),
        vis_data['dps']['tp_tavg']['sum'],
        label='True EoR + Foreground',
        color='k',
        ls='-',
        alpha=0.7
    )
    ax.plot(
        delays.to('ns'),
        vis_data['dps']['tp_tavg']['eor'],
        label='True EoR',
        color='k',
        ls=':',
        alpha=0.7
    )
    ax.plot(
        delays.to('ns'),
        vis_data['dps']['tp_tavg']['noise'],
        label='Injected noise',
        color='k',
        ls='-.'
    )
    ax.plot(
        delays.to('ns'),
        hp_data['dps']['tp_tavg']['gcr'].mean(axis=0),
        label=r'EoR sample mean',
        color='b',
        ls='-',
        alpha=0.6
    )
    ax.fill_between(
        delays.to('ns').value,
        hp_data['dps']['tp_tavg']['gcr_ci'][95]['lbound'].value,
        hp_data['dps']['tp_tavg']['gcr_ci'][95]['ubound'].value,
        label=r'EoR sample 95% conf.',
        color='g',
        alpha=0.3
    )
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(\tau)$ [Jy$^2$]')

    # GCR errorbar subplot
    ax = axs[1]
    true = vis_data['dps']['tp_tavg']['eor']
    gcr_mean = hp_data['dps']['tp_tavg']['gcr'].mean(axis=0)
    gcr_lbound = hp_data['dps']['tp_tavg']['gcr_ci'][68]['lbound']
    gcr_ubound = hp_data['dps']['tp_tavg']['gcr_ci'][68]['ubound']
    frac_diff = gcr_mean/true - 1
    frac_diff[~delay_mask] = 0.0
    # Errorbars
    errbar_crs = np.zeros((2, Nfreqs))
    # Lower bounds
    errbar_crs[0] = frac_diff - (gcr_lbound/true - 1)
    if np.any(errbar_crs[0] < 0):
        print('GCRs: Negative errorbar values found in lower bounds')
        inds = np.where(errbar_crs[0] < 0)[0]
        pprint(inds)
        pprint(errbar_crs[0, inds])
        errbar_crs[0, inds] = np.abs(errbar_crs[0, inds])
    # Upper bounds
    errbar_crs[1] = gcr_ubound/true - 1 - frac_diff
    if np.any(errbar_crs[1] < 0):
        print('GCRs: Negative errorbar values found in upper bounds')
        inds = np.where(errbar_crs[1] < 0)[0]
        pprint(inds)
        pprint(errbar_crs[1, inds])
        errbar_crs[1, inds] = np.abs(errbar_crs[1, inds])
    ax.errorbar(
        delays.to('ns')[delay_mask],
        frac_diff[delay_mask],
        yerr=errbar_crs[:, delay_mask],
        color='k',
        alpha=0.75,
        label='Residual, 68% conf.',
        elinewidth=1.5,
        capsize=2.5,
        fmt='.'
    )
    ax.axhline(0, color='k', ls='--', alpha=0.7)
    ax.set_ylabel(
        r'$\langle \tilde e_{cr,i}^\dag ~\tilde e_{cr,i} \rangle$ frac. resid.'
    )
    ax.set_ylim([-0.2, 0.2])
    ax.set_yticks(ax.get_yticks()[1:-1])

    # Time-averaged delay power spectrum sample errorbar subplot
    ax = axs[2]
    true = vis_data['dps']['fft_tavg']['eor']
    dps_mean = hp_data['dps']['fft_tavg']['dps'].mean(axis=0)
    dps_lbound = hp_data['dps']['fft_tavg']['dps_ci'][68]['lbound']
    dps_ubound = hp_data['dps']['fft_tavg']['dps_ci'][68]['ubound']
    frac_diff = dps_mean/true - 1
    # Errorbars
    errbar_est = np.zeros((2, Nfreqs))
    # Lower bounds
    errbar_est[0] = frac_diff - (dps_lbound/true - 1)
    if np.any(errbar_est[0] < 0):
        print('TADPS: Negative errorbar values found in lower bounds')
        inds = np.where(errbar_est[0] < 0)[0]
        pprint(inds)
        pprint(errbar_est[0, inds])
        errbar_est[0, inds] = np.abs(errbar_est[0, inds])
    # Upper bounds
    errbar_est[1] = dps_ubound/true - 1 - frac_diff
    if np.any(errbar_est[1] < 0):
        print('TADPS: Negative errorbar values found in upper bounds')
        inds = np.where(errbar_est[1] < 0)[0]
        pprint(inds)
        pprint(errbar_est[1, inds])
        errbar_est[1, inds] = np.abs(errbar_est[1, inds])
    ax.errorbar(
        delays.to('ns')[delay_mask],
        frac_diff[delay_mask],
        yerr=errbar_est[:, delay_mask],
        color='k',
        alpha=0.75,
        label='Residual, 68% conf.',
        elinewidth=1.5,
        capsize=2.5,
        fmt='.'
    )
    ax.axhline(0, color='k', ls='--', alpha=0.7)
    ax.set_xlabel(r'$\tau$ [ns]')
    ax.set_ylabel(r'$\sigma^2_i$ frac. resid.')
    ax.set_ylim([-0.2, 0.2])
    ax.set_yticks(ax.get_yticks()[1:-1])

    for ax in axs:
        ax.legend(**legend_kwargs)
        ax.grid(which='major', ls=':')

    fig.suptitle(suptitle)
    fig.tight_layout()
    
    plt.show()

    return fig


def plot_posteriors_w_priors(
    delays,
    hp_data,
    suptitle=None,
    tapered=False,
    Nhistbins=31,
    xlim=None,
    log=False
):
    if not isinstance(delays, Quantity):
        delays = Quantity(delays, unit='ns')
    else:
        delays = delays.to('ns')
    args = hp_data['args']
    Nfreqs = delays.size
    ps_prior_inds = slice(
        Nfreqs//2 - args.n_ps_prior_bins,
        Nfreqs//2 + args.n_ps_prior_bins + 1
    )
    
    nplots = 2 * args.n_ps_prior_bins + 1
    plot_height = 1
    figsize = (12, nplots*plot_height)
    fig, axs = plt.subplots(
        nplots, 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.1}
    )
    if args.ps_prior_lo == 0:
        bins_min = 1e-16
    else:
        bins_min = args.ps_prior_lo
    bins_max = args.ps_prior_hi
    bins = np.logspace(np.log10(bins_min), np.log10(bins_max), Nhistbins)
    hist_kwargs = dict(histtype='step', lw=2, density=True, bins=bins, log=log)
    if tapered:
        ps_key = 'tp_tavg'
    else:
        ps_key = 'fft_tavg'
    if xlim is None:
        xlim = (
            0.9*hp_data['dps'][ps_key]['dps'][:, ps_prior_inds].min().value,
            1.1*hp_data['dps'][ps_key]['dps'][:, ps_prior_inds].max().value
        )
    for i_ax, ax in enumerate(axs):
        _ = ax.hist(
            hp_data['dps'][ps_key]['dps'][:, ps_prior_inds][:, i_ax].value,
            **hist_kwargs
        )
        # ax.set_ylabel(fr'$\tau$ = {delays[ps_prior_inds][i_ax]:.1f}')
        ax.annotate(
            fr'$\tau$ = {delays[ps_prior_inds][i_ax]:.1f}',
            (0.015, 0.85),
            xycoords='axes fraction',
            ha='left',
            va='top',
            bbox=dict(facecolor='w', alpha=0.75)
        )
        ax.axvline(args.ps_prior_lo, ls='--', color='k')
        ax.axvline(args.ps_prior_hi, ls='--', color='k')
        ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.grid(ls=':', which='both')
        # ax.set_yticks([])
    ax.set_xlabel(fr"$P(\tau)$ {hp_data['dps'][ps_key]['dps'].unit}")
    fig.suptitle(suptitle)


def fg_chains_plot(
    hp_data,
    vis_fgs,
    bl_str,
    i_t=0,
    Nburn=0,
    Nsigma=5,
    log=False,
    bins=31,
    plot_size=4,
    suptitle=None
):
    args = hp_data['args']
    if 'fg_eig_file' in args:
        fg_model_path = Path(args.fg_eig_dir)
    else:
        fg_model_path = Path(args.fgmodes)
    if fg_model_path.is_dir():
        if 'fg_eig_file' in args:
            fg_model_file = args.fg_eig_file
        else:
            fg_model_file = args.fgmodes_file
        fg_model_path /= bl_str
    else:
        fg_model_file = fg_model_path.name
        fg_model_path = fg_model_path.parent

    fg_model_evecs = np.load(
        fg_model_path / fg_model_file
    )
    fg_model_evals = np.load(
        fg_model_path / fg_model_file.replace('evecs', 'evals')
    )
    Nfgmodes = hp_data['args'].Nfgmodes
    fg_model_evecs = fg_model_evecs[:, :Nfgmodes]
    fg_model_evals = fg_model_evals[:Nfgmodes]

    fg_amps = hp_data['fg_amps'].copy()
    Niter = fg_amps.shape[0]

    nrows = Nfgmodes
    if 'ln_post' in hp_data:
        # Add a row for plotting the log posterior
        nrows += 1
    ncols = 4

    figsize = (ncols*plot_size*1.5, 0.6*nrows*plot_size)
    fig = plt.figure(figsize=figsize)
    gs_re = GridSpec(
        nrows, 2, figure=fig, right=0.48, wspace=0.02,
        width_ratios=[3, 1]
    )
    gs_im = GridSpec(
        nrows, 2, figure=fig, left=0.52, wspace=0.02,
        width_ratios=[3, 1]
    )

    hist_kwargs = dict(
        histtype='step', bins=bins, density=True, lw=2,
        orientation='horizontal'
    )

    if 'ln_post' in hp_data:
        ln_post_axs = [
            fig.add_subplot(gs_re[0, 0]),
            fig.add_subplot(gs_im[0, 0])
        ]
        for ax in ln_post_axs:
            ax.plot(hp_data['ln_post'], 'k-')
            ax.set_xticklabels([])
        ln_post_axs[0].set_ylabel('Log Posterior')

    for i_fg in range(Nfgmodes):
        if 'ln_post' in hp_data:
            i_ax = i_fg + 1
        else:
            i_ax = i_fg

        mean_re = fg_amps[Nburn:, i_t, i_fg].real.mean()
        mean_im = fg_amps[Nburn:, i_t, i_fg].imag.mean()
        mean = mean_re + 1j*mean_im
        std_re = fg_amps[Nburn:, i_t, i_fg].real.std()
        std_im = fg_amps[Nburn:, i_t, i_fg].imag.std()
        std = std_re + 1j*std_im
        data_lims = np.array([mean - Nsigma*std, mean + Nsigma*std])

        ax = fig.add_subplot(gs_re[i_ax, 0])
        # Plot chain
        if i_ax == int('ln_post' in hp_data):
            label = 'chain'
        else:
            label = None
        ax.plot(fg_amps[:, i_t, i_fg].real, label=label)
        if Nburn > 0:
            if i_ax == int('ln_post' in hp_data):
                label += f'[{Nburn}:]'
            ax.plot(
                np.arange(Niter)[Nburn:],
                fg_amps[Nburn:, i_t, i_fg].real,
                label=label
            )
        # Plot the projection coefficient of the data onto the model
        proj_coeff = fg_model_evecs.conj().T @ vis_fgs[i_t]
        if i_ax == int('ln_post' in hp_data):
            label = 'Projection Coefficient'
            ax.set_title('Real')
        ax.axhline(
            proj_coeff[i_fg].real,
            ls='-',
            color='k',
            label=label
        )
        if log:
            ax.set_yscale('log')
        if i_ax == int('ln_post' in hp_data):
            ax.legend(loc='lower right', ncols=3, fontsize=9)
        elif i_ax == nrows - 1:
            ax.set_xlabel('Iteration')
        ax.set_ylim(data_lims.real)
        ax.set_ylabel('Sample Amplitude')
        props = dict(boxstyle='square', facecolor='white', alpha=0.7)
        ax.annotate(
            rf"$i_{{\rm{{FG}}}}$ = {i_fg}", (0.02, 0.925),
            xycoords='axes fraction', ha='left', va='top',
            bbox=props
        )
        ax.grid()

        ax = fig.add_subplot(gs_re[i_ax, 1])
        _ = ax.hist(fg_amps[:, i_t, i_fg].real, **hist_kwargs)
        if Nburn > 0:
            _ = ax.hist(fg_amps[Nburn:, i_t, i_fg].real, **hist_kwargs)
        ax.axhline(
            proj_coeff[i_fg].real,
            ls='-',
            color='k'
        )
        if i_ax == int('ln_post' in hp_data):
            ax.set_title('Real')
        ax.set_ylim(data_lims.real)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid()

        # Imaginary component
        ax = fig.add_subplot(gs_im[i_ax, 0])
        # Plot chain
        ax.plot(fg_amps[:, i_t, i_fg].imag)
        if Nburn > 0:
            ax.plot(
                np.arange(Niter)[Nburn:],
                fg_amps[Nburn:, i_t, i_fg].imag
            )
        # Plot true eigenvalue
        ax.axhline(
            proj_coeff[i_fg].imag,
            ls='-',
            color='k'
        )
        if log:
            ax.set_yscale('log')
        if i_ax == int('ln_post' in hp_data):
            ax.set_title('Imag')
        elif i_ax == nrows - 1:
            ax.set_xlabel('Iteration')
        ax.set_ylim(data_lims.imag)
        ax.grid()

        ax = fig.add_subplot(gs_im[i_ax, 1])
        _ = ax.hist(fg_amps[:, i_t, i_fg].imag, **hist_kwargs)
        if Nburn > 0:
            _ = ax.hist(fg_amps[Nburn:, i_t, i_fg].imag, **hist_kwargs)
        ax.axhline(
            proj_coeff[i_fg].imag,
            ls='-',
            color='k'
        )
        if i_ax == int('ln_post' in hp_data):
            ax.set_title('Imag')
        ax.set_ylim(data_lims.imag)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid()

    if suptitle is not None:
        suptitle += fr' ($i_t = {i_t}$)'
        fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.95)

    plt.show()

    return fig


def dps_chains_plot(
    delays,
    dps_chains,
    dps_true=None,
    delay_inds=None,
    delay_vals=None,
    Nburn=0,
    Nsigma=5,
    plot_size=4,
    log=False,
    bins=51,
    suptitle=None
):
    """
    Plot chains as a time-series and a histogram to monitor convergence.
    
    Parameters
    ----------
    delays : array-like or astropy.units.Quantity
        Delays in nanoseconds with shape (Ndelays,).
    dps_chains : array-like
        Array-like of chains delay power spectrum (DPS) amplitudes with shape
        (Niter, Ndelays).
    dps_true : array-like
        Array-like of true DPS amplitudes with shape (Ndelays,).
    delay_inds : array-like
        Indices of delay bins to plot chains.
    delay_vals : array-like or astropy.units.Quantity
        Delays (in nanoseconds) to plot chains.
    
    """
    Niter, Ndelays = dps_chains.shape
    
    if not isinstance(delays, Quantity):
        delays = Quantity(delays, unit='ns')
    else:
        delays = delays.to('ns')
    if delay_vals is not None:
        if not isinstance(delay_vals, Quantity):
            delay_vals = Quantity(delay_vals, unit='ns')
        else:
            delay_vals = delay_vals.to('ns')
    
    if delay_vals is not None:
        delay_inds = np.zeros(len(delay_vals), dtype=int)
        for dly_ind, delay in enumerate(delay_vals):
            delay_inds[dly_ind] = np.argmin(np.abs((delays - delay).value))
    elif delay_inds is not None:
        pass
    else:
        print('Must pass delay_inds or delays.  Exiting.')
        return
    nrows = len(delay_inds)
    ncols = 2
    
    figsize = (ncols*plot_size, 0.6*nrows*plot_size)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize,
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.02}
    )
    
    for i_row, dly_ind in enumerate(delay_inds):
        mean = dps_chains[Nburn:, dly_ind].mean().value
        std = dps_chains[Nburn:, dly_ind].std().value
        data_lims = (mean - Nsigma*std, mean + Nsigma*std)
        
        ax = axs[i_row, 0]
        if i_row == 0:
            label = 'chain'
        else:
            label = None
        ax.plot(dps_chains[:, dly_ind].value, label=label)
        if Nburn > 0:
            if i_row == 0:
                label += f'[{Nburn}:]'                
            ax.plot(
                np.arange(Niter)[Nburn:],
                dps_chains[Nburn:, dly_ind].value,
                label=label
            )
        if dps_true is not None:
            if i_row == 0:
                label = 'True'
            ax.axhline(
                dps_true[dly_ind].value,
                ls='--',
                color='k',
                label=label
            )
        if log:
            ax.set_yscale('log')
        if i_row == 0:
            ax.legend(loc='lower right', ncols=3, fontsize=9)
        elif i_row == nrows - 1:
            ax.set_xlabel('Iteration')
        if i_row < nrows - 1:
            ax.set_xticklabels([])
        ax.set_ylim(data_lims)
        ax.set_ylabel('Sample Amplitude', fontsize=12)
        ax.annotate(
            fr"$\tau$ = {delays[dly_ind]:.2f}", (0.05, 0.95),
            xycoords='axes fraction', ha='left', va='top'
        )
        
        ax = axs[i_row, 1]
        hist_kwargs = dict(
            histtype='step', bins=bins, density=True, lw=2,
            orientation='horizontal'
        )
        _ = ax.hist(dps_chains[:, dly_ind], **hist_kwargs)
        if Nburn > 0:
            _ = ax.hist(dps_chains[Nburn:, dly_ind], **hist_kwargs)
        if dps_true is not None:
            ax.axhline(
                dps_true[dly_ind].value,
                ls='--',
                color='k'
            )
        ax.set_ylim(data_lims)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    for ax in axs.flatten():
        ax.grid(ls=':')
    
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()

    plt.show()
    
    return fig


def fix_neg_errbars(errbars, message=None):
    lbounds = errbars[0]
    ubounds = errbars[1]
    
    neg_lbounds = np.any(lbounds < 0)
    neg_ubounds = np.any(ubounds < 0)
    if message is not None and neg_lbounds or neg_ubounds:
        print(message)
    
    if neg_lbounds:
        print('Negative values found in lower bounds')
        inds = np.where(lbounds < 0)[0]
        pprint(inds)
        pprint(lbounds[inds])
        lbounds[inds] = np.abs(lbounds[inds])
        print()
    if neg_ubounds:
        print('Negative values found in upper bounds')
        inds = np.where(ubounds < 0)[0]
        pprint(inds)
        pprint(ubounds[inds])
        ubounds[inds] = np.abs(ubounds[inds])
        print()
    errbars[0] = lbounds
    errbars[1] = ubounds
    
    return errbars


def inv_gamma_draws_plot(
    vis_data,
    hp_data,
    suptitle=None,
    ci=68,
    ylim_dps=None,  # [0.07, 4963.378888338189]
    ylim_fd=[-0.2, 0.2]
):
    """
    Compare the input signal with the GCRs and inverse gamma samples.
    
    Each Time Averaged Delay Power Spectrum (TADPS) sample per delay bin is
    drawn from an inverse gamma distribution.  The invgamma distribution for
    each delay bin uses the number of observations as the shape parameter and
    the sum of the absolute value squared of the delay transform of the
    Gaussian Constrained Realizations (GCRs) as the scale parameter.  If the
    GCRs agree in amplitude with the input data, and the TADPS samples agree
    with the GCRs, then the TADPS should agree with the input data.  This plot
    checks this agreement and compares invgamma draws using the GCRs with the
    TADPS from hydra-pspec.

    """
    delays = vis_data['delays']
    true_props = dict(color='k', ls='--', alpha=0.7)
    noise_props = dict(color='k', ls='-.')
    gcrs_props = dict(color='C0', ls='-', alpha=0.6)
    dpss_props = dict(color='C3', ls=':', alpha=0.6)
    
    # Delay spectra
    # GCRs, shape (Niter, Ntimes, Ndelays)
    ds_gcrs = hp_data['ds']['fft']['gcr']
    
    Niter, Ntimes, Ndelays = ds_gcrs.shape
    
    # Delay power spectra
    # Input EoR data i.e. the "true" signal, shape (Ndelays,)
    dps_true = vis_data['dps']['fft_tavg']['eor']
    # Noise, shape (Ndelays,)
    dps_noise = vis_data['dps']['fft_tavg']['noise']
    # GCRs, shape (Niter, Ndelays)
    # FIXME: Do I need to account for sample variance (which is accounted for
    # by the inverse gamma sampling for the delay power spectrum) which means
    # multiplying the time average by Ntimes / (Ntimes - 1) since the time
    # average is formed by dividing by Ntimes?
    dps_gcrs = hp_data['dps']['fft_tavg']['gcr']  # * Ntimes / (Ntimes - 1)
    dps_gcrs_sample_mean = dps_gcrs.mean(axis=0)
    dps_gcrs_lbound = hp_data['dps']['fft_tavg']['gcr_ci'][ci]['lbound']
    dps_gcrs_ubound = hp_data['dps']['fft_tavg']['gcr_ci'][ci]['ubound']
    # TADPS samples, shape (Niter, Ndelays)
    dps_dpss = hp_data['dps']['fft_tavg']['dps']
    dps_dpss_sample_mean = dps_dpss.mean(axis=0)
    dps_dpss_lbound = hp_data['dps']['fft_tavg']['dps_ci'][ci]['lbound']
    dps_dpss_ubound = hp_data['dps']['fft_tavg']['dps_ci'][ci]['ubound']
    
    nrows = 2
    ncols = 3
    plot_width = 6
    plot_height = 4
    figsize = (ncols*plot_width, nrows*plot_height)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Compare input data and GCRs
    ax = axs[0, 0]
    ax.set_title('GCRs vs True')
    ax.plot(
        delays.to('ns'), dps_true, label='True', **true_props
    )
    ax.plot(
        delays.to('ns'), dps_noise, label='Injected noise', **noise_props
    )
    ax.plot(
        delays.to('ns'), dps_gcrs_sample_mean, label=r'GCR sample mean',
        **gcrs_props
    )
    ax.fill_between(
        delays.to('ns').value, dps_gcrs_lbound.value, dps_gcrs_ubound.value,
        label=f'GCR sample {ci}% conf.', color=gcrs_props['color'], alpha=0.1
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylabel(r'$P(\tau)$ [Jy$^2$]')
    
    # Fractional difference
    ax = axs[1, 0]
    fd, fd_err = calc_frac_diff(
        dps_gcrs_sample_mean, dps_true,
        x_lbound=dps_gcrs_lbound, x_ubound=dps_gcrs_ubound
    )
    fd_err = fix_neg_errbars(
        fd_err, message='GCRs vs True\n'+'-'*len('GCRs vs True')
    )
    mean_fd = np.average(fd, weights=np.diff(fd_err, axis=0).squeeze())
    ax.plot(
        delays.to('ns'), fd, label=f'Mean fractional diff.\n({mean_fd:.2f})',
        color='k', ls='-', alpha=0.6
    )
    ax.fill_between(
        delays.to('ns').value,
        fd - fd_err[0],
        fd + fd_err[1],
        label=f'{ci}% conf.',
        color='k',
        alpha=0.1
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylabel('Fractional Difference')
    
    
    # Compare GCRs and TADPS
    ax = axs[0, 1]
    ax.set_title('GCRs vs TADPSs')
    ax.plot(
        delays.to('ns'), dps_noise, label='Injected noise', **noise_props
    )
    ax.plot(
        delays.to('ns'), dps_gcrs_sample_mean, label=r'GCR sample mean',
        **gcrs_props
    )
    ax.fill_between(
        delays.to('ns').value, dps_gcrs_lbound.value, dps_gcrs_ubound.value,
        label=f'GCR sample {ci}% conf.', color=gcrs_props['color'], alpha=0.1
    )
    ax.plot(
        delays.to('ns'), dps_dpss_sample_mean, label='TADPS sample mean',
        **dpss_props
    )
    ax.fill_between(
        delays.to('ns').value, dps_dpss_lbound.value, dps_dpss_ubound.value,
        label=f'TADPS sample {ci}% conf.', color=dpss_props['color'], alpha=0.1
    )
    ax.legend(loc='upper right', fontsize=10)
    
    # Fractional difference
    ax = axs[1, 1]
    fd, fd_err = calc_frac_diff(
        dps_dpss_sample_mean, dps_gcrs_sample_mean,
        x_lbound=dps_dpss_lbound, x_ubound=dps_dpss_ubound,
        y_lbound=dps_gcrs_lbound, y_ubound=dps_gcrs_ubound
    )
    fd_err = fix_neg_errbars(
        fd_err, message='TADPSs vs GCRs\n'+'-'*len('TADPSs vs GCRs')
    )
    mean_fd = np.average(fd, weights=np.diff(fd_err, axis=0).squeeze())
    ax.plot(
        delays.to('ns'), fd, label=f'Mean fractional diff.\n({mean_fd:.2e})',
        color='k', ls='-', alpha=0.6
    )
    ax.fill_between(
        delays.to('ns').value,
        fd - fd_err[0],
        fd + fd_err[1],
        label=f'{ci}% conf.',
        color='k',
        alpha=0.1
    )
    ax.legend(loc='upper right', fontsize=10)
    
    
    # Compare input data and TADPS
    ax = axs[0, 2]
    ax.set_title('TADPSs vs True')
    ax.plot(
        delays.to('ns'), dps_true, label='True', **true_props
    )
    ax.plot(
        delays.to('ns'), dps_noise, label='Injected noise', **noise_props
    )
    ax.plot(
        delays.to('ns'), dps_dpss_sample_mean, label=r'TADPS sample mean',
        **dpss_props
    )
    ax.fill_between(
        delays.to('ns').value, dps_dpss_lbound.value, dps_dpss_ubound.value,
        label=f'TADPS sample {ci}% conf.', color=dpss_props['color'], alpha=0.1
    )
    ax.legend(loc='upper right', fontsize=10)
    
    # Fractional difference
    ax = axs[1, 2]
    fd, fd_err = calc_frac_diff(
        dps_dpss_sample_mean, dps_true,
        x_lbound=dps_dpss_lbound, x_ubound=dps_dpss_ubound
    )
    fd_err = fix_neg_errbars(
        fd_err, message='TADPSs vs True\n'+'-'*len('TADPSs vs True')
    )
    mean_fd = np.average(fd, weights=np.diff(fd_err, axis=0).squeeze())
    ax.plot(
        delays.to('ns'), fd, label=f'Mean fractional diff.\n({mean_fd:.2e})',
        color='k', ls='-', alpha=0.6
    )
    ax.fill_between(
        delays.to('ns').value,
        fd - fd_err[0],
        fd + fd_err[1],
        label=f'{ci}% conf.',
        color='k',
        alpha=0.1
    )
    ax.legend(loc='upper right', fontsize=10)
    
    for ax in axs[0]:
        ax.set_yscale('log')
        ax.set_ylim(ylim_dps)
        ax.grid(ls=':')
    for ax in axs[1]:
        # ax.set_yticks(
        #     [-0.1, 0, 0.1], labels=[-0.1, 0, 0.1], minor=False
        # )
        # ax.set_yticks(
        #     [-0.15, -0.05, 0.05, 0.15], labels=None, minor=True
        # )
        # ax.set_ylim([-0.2, 0.2])
        ax.set_xlabel(r'$\tau$ [ns]')
        ax.set_ylim(ylim_fd)
        ax.grid(ls=':', which='both')

    if not suptitle is None:
        fig.suptitle(suptitle)
    fig.tight_layout()

    plt.show()
    
    return fig


def compare_cov(
    cov,
    hp_data,
    iteration=None,
    Nburn=0,
    conf_intervals=[68, 95],
    comp_type='re',
    force_clim=False,
    suptitle=''
):
    dps_samples = hp_data['dps']['fft_tavg']['dps'].copy()
    Niter, Nfreqs = dps_samples.shape
    
    dft_mat = fourier_operator(Nfreqs)
    if iteration is None:
        cov_samples = np.zeros((Niter, Nfreqs, Nfreqs), dtype=complex)
        for i_iter in range(Niter):
            cov_samples[i_iter] = covariance_from_pspec(
                dps_samples[i_iter] / Nfreqs**2, dft_mat
            )
        cov_sample = np.mean(cov_samples[Nburn:], axis=0)
    else:
        cov_sample = covariance_from_pspec(
            dps_samples[iteration] / Nfreqs**2, dft_mat
        )
    # cov_samples_ci = {}
    # for conf_interval in conf_intervals:
    #     percentile = conf_interval/2 + 50
    #     ci_bounds = {}
    #     ci_bounds['lbound'] = np.percentile(
    #         cov_samples[Nburn:], 100-percentile, axis=0
    #     )
    #     ci_bounds['ubound'] = np.percentile(
    #         cov_samples[Nburn:], percentile, axis=0
    #     )
    #     cov_samples_ci[conf_interval] = ci_bounds
    
    labels = ['Input']
    if suptitle is not None:
        suptitle += ', '
    if iteration is None:
        suptitle += f'Nburn = {Nburn}'
        labels.append('Sample Mean')
    else:
        suptitle += f'Iteration {iteration}'
        labels.append('Sample')
        
    fig = compare_data(
        cov,
        cov_sample,
        labels=labels,
        suptitle=suptitle,
        comp_type=comp_type,
        force_clim=force_clim
    )

    plt.show()


def error_correlation_plot(vis_data, hp_data, suptitle=None):
    delays = vis_data['delays']
    ddelay = delays[1] - delays[0]
    extent_delay = [
        (delays.min() - ddelay/2).to('ns').value,
        (delays.max() + ddelay/2).to('ns').value
    ]
    true = vis_data['dps']['tp_tavg']['eor']
    
    fig_height = 5
    if suptitle is not None:
        fig_height += 0.5*suptitle.count('\n') + 1
    fig = plt.figure(figsize=(12, fig_height))
    grid = ImageGrid(
        fig, 111, (1, 2), share_all=True, axes_pad=0.1,
        cbar_mode='single', cbar_pad=0.1
    )
    im_kwargs = dict(
        vmin=-1, vmax=1, origin='upper', cmap='RdBu_r',
        extent=extent_delay+extent_delay[::-1]
    )
    
    
    # GCRs
    gcrs = hp_data['dps']['tp_tavg']['gcr']
    frac_diff = gcrs / true[None, :] - 1
    corr_frac_diff = np.corrcoef(frac_diff.value.T)
    
    ax = grid.axes_all[0]
    # ax.set_title(
    #     r'$\langle \tilde e_{cr,i}^\dag ~\tilde e_{cr,i} \rangle$ frac. resid.'
    # )
    ax.set_title('GCR Fractional Difference')
    im = ax.imshow(corr_frac_diff, **im_kwargs)
    _ = fig.colorbar(im, cax=ax.cax, label='Correlation')
    
    
    # TADPSs
    dpss = hp_data['dps']['fft_tavg']['dps']
    frac_diff = dpss / true[None, :] - 1
    corr_frac_diff = np.corrcoef(frac_diff.value.T)
    
    ax = grid.axes_all[1]
    # ax.set_title(r'$\sigma^2_i$ frac. resid.')
    ax.set_title('DPS Fractional Difference')
    im = ax.imshow(corr_frac_diff, **im_kwargs)
    _ = fig.colorbar(im, cax=ax.cax, label='Correlation')
    
    for ax in grid.axes_all:
        ax.set_xlabel(r'$\tau$ [ns]')
        ax.set_ylabel(r'$\tau$ [ns]')
    
    if suptitle is not None:
        fig.suptitle(suptitle)
    
    plt.show()
    
    return fig


def plot_log_posterior(
    vis_data,
    hp_data,
    bl_str,
    suptitle=None,
    inv_plot=True
):
    """
    Plot ln Pr(...) ~ -(d - m)^\dagger N^-1 (d - m) - e^\dagger E^-1 e.

    Parameters
    ----------
    vis_data : dict
        Dictionary from `get_vis_data`.
    hp_data : dict
        Dictionary from `get_hp_data`.
    bl_str : str
        Antenna pair string of the form f'{ant1}-{ant2}'.
    suptitle : str
        Figure suptitle.

    """
    if not 'ln_post' in hp_data:
        # Load foreground model
        fg_model_dir = Path(hp_data['args'].fg_eig_dir)
        fg_model_path = fg_model_dir / bl_str / hp_data['args'].fg_eig_file
        fg_model_vecs = np.load(fg_model_path)
        fg_model_vecs = fg_model_vecs[:, :hp_data['args'].Nfgmodes]
        
        # Load noise covariance matrix
        noise_cov_dir = Path(hp_data['args'].noise_cov)
        noise_cov_path = (
            noise_cov_dir / bl_str / hp_data['args'].noise_cov_file
        )
        N = np.load(noise_cov_path)
        # N is diagonal, so np.linalg.inv is fine
        Ninv = np.linalg.inv(N)

        # data vector
        data = vis_data['vis']['sum_noise']  # shape (Ntimes, Nfreqs)
        # Gaussian constrained realizations
        eor_model = hp_data['gcr']  # shape (Niter, Ntimes, Nfreqs)
        Niter, Ntimes, Nfreqs = eor_model.shape
        # Foreground model
        fg_amps = hp_data['fg_amps']
        fg_model = np.einsum('ijk,kl->ijl', fg_amps, fg_model_vecs.T)
        # Combined EoR+FG model
        model = eor_model + fg_model
        # Delay power spectrum samples
        dpss = hp_data['dps']['fft_tavg']['dps']
        # Fourier operator for freq -> delay transform
        fourier_op = fourier_operator(Nfreqs)
        
        ln_post = np.zeros(Niter)
        if inv_plot:
            inv_test = np.zeros_like(ln_post)
        for i_it in range(Niter):
            # EoR frequency-frequency covariance
            E = covariance_from_pspec(dpss[i_it] / Nfreqs**2, fourier_op)
            # Using np.linalg.inv is not advisable, but because E is typically
            # approximately diagonal it should be okay.  The bottom subplot in
            # the generated figure checks that Einv @ E is the identity matrix.
            Einv = np.linalg.inv(E)
            if inv_plot:
                inv_test[i_it] = np.mean(np.abs(Einv @ E - np.eye(Nfreqs)))
            
            # Compute log posterior at each time and sum over time axis
            # We're assuming that each time sample is an independent observation.
            # The probability of getting all time samples is thus the product of
            # each probability. In log space, this corresponds to the sum of each
            # time's posterior probability.
            ln_post[i_it] = np.sum(np.diagonal(
                -(data - model[i_it]).conj() @ Ninv @ (data - model[i_it]).T
                - eor_model[i_it].conj() @ Einv @ eor_model[i_it].T
            )).real
    else:
        inv_plot = False
        ln_post = hp_data['ln_post']
        Niter = ln_post.size
    
    iterations = np.arange(Niter) + 1
    nrows = 1 + inv_plot
    plot_height = 3
    fig, axs = plt.subplots(
        nrows, 1, figsize=(12, nrows*plot_height), sharex=True,
        gridspec_kw={'hspace': 0.1}
    )
    
    if inv_plot:
        ax = axs[0]
    else:
        ax = axs
    ax.plot(iterations, ln_post, 'k-')
    if np.abs(ln_post).max() / np.abs(ln_post).min() > 10:
        ax.set_yscale('symlog', linthresh=1e-1)
        ax.set_ylim((4*ln_post.min(), 0.25*ln_post.max()))
    ax.set_ylabel(r'$\ln$ Posterior')
    if suptitle is not None:
        ax.set_title(suptitle)
    
    if inv_plot:
        ax = axs[1]
        ax.plot(iterations, inv_test.real, 'k.', label='Real')
        ax.set_ylabel(r'|$\mathbf{E}^{-1}\mathbf{E} - \mathbf{I}|$')
        ax.set_xlabel('Iteration')
    
        for ax in axs:
            ax.grid(ls=':')
    else:
        ax.set_xlabel('Iteration')
        ax.grid(ls=':')
    
    plt.show()
    
    return fig


def autocorrelation_plot(
    delays,
    hp_data,
    delay_inds=None,
    delay_vals=None,
    Nburn=0,
    plot_size=4,
    suptitle=None
):
    """
    Parameters
    ----------
    delays : array-like or astropy.units.Quantity
        Delays in nanoseconds with shape (Ndelays,).
    hp_data : dict
        Dictionary from `get_hp_data`.
    delay_inds : array-like
        Indices of delay bins to plot chains.
    delay_vals : array-like or astropy.units.Quantity
        Delays (in nanoseconds) to plot chains.
    plot_size : float
        Height of each subplot.
    suptitle : str
        Figure suptitle.

    """
    dpss = hp_data['dps']['fft_tavg']['dps'].copy()
    dpss -= dpss[Nburn:].mean(axis=0)[None, :]
    # dpss = np.random.normal(0, 1, dpss.shape)
    start = time.time()
    print('Computing autocorrelation...', end=' ')
    dpss_autocorr = fftconvolve(dpss, dpss, mode='full', axes=(0,))
    dpss_autocorr /= np.sum(dpss.value**2, axis=0)
    print(f'({format_time(time.time() - start):.1f})')
    Niter, Nfreqs = dpss.shape
    lags = np.arange(2*Niter-1) - Niter
    
    if not isinstance(delays, Quantity):
        delays = Quantity(delays, unit='ns')
    else:
        delays = delays.to('ns')
    if delay_vals is not None:
        if not isinstance(delay_vals, Quantity):
            delay_vals = Quantity(delay_vals, unit='ns')
        else:
            delay_vals = delay_vals.to('ns')
    
    if delay_vals is not None:
        delay_inds = np.zeros(len(delay_vals), dtype=int)
        for dly_ind, delay in enumerate(delay_vals):
            delay_inds[dly_ind] = np.argmin(np.abs((delays - delay).value))
    elif delay_inds is not None:
        pass
    else:
        print('Must pass delay_inds or delays.  Exiting.')
        return
    nrows = len(delay_inds)
    ncols = 1
    
    figsize = (2.5*ncols*plot_size, 0.4*nrows*plot_size)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=True,
        gridspec_kw={'hspace': 0.05}
    )
    
    for i_row, dly_ind in enumerate(delay_inds):
        ax = axs[i_row]
        ax.plot(lags, dpss_autocorr[:, dly_ind], 'k.')
        ax.annotate(
            fr"$\tau$ = {delays[dly_ind].to('ns'):.1f}", (0.025, 0.95),
            xycoords='axes fraction', ha='left', va='top'
        )
    ax.set_xlabel('Lag [Iterations]')
    
    for ax in axs:
        ax.grid(ls=':')
        
    fig.supylabel('Autocorrelation')
    fig.suptitle(suptitle)
    
    return fig


def dps_inspection(
    vis_data, hp_data, xlim=None, ylim=None, prior_lo=5000, prior_hi=20000,
    suptitle=None
):
    delays = vis_data['delays'].to('ns')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        delays, vis_data['dps']['fft_tavg']['sum'],
        ls='-', color='k', label='True EoR + Foreground'
    )
    ax.plot(
        delays, vis_data['dps']['fft_tavg']['eor'],
        ls=':', color='k', label='True EoR'
    )
    ax.plot(
        delays, vis_data['dps']['fft_tavg']['noise'],
        ls='-.', color='k', label='Injected noise'
    )
    ax.plot(
        delays, hp_data['dps']['fft_tavg']['gcr_mean'],
        ls='-', color='C0', label='EoR sample mean', zorder=15
    )
    ax.fill_between(
        delays.value,
        hp_data['dps']['fft_tavg']['gcr_ci'][95]['lbound'].value,
        hp_data['dps']['fft_tavg']['gcr_ci'][95]['ubound'].value,
        color='C0',
        alpha=0.35,
        zorder=10
    )
    yerr = np.zeros((2, delays.size))
    yerr[0] = (
        hp_data['dps']['fft_tavg']['dps_mean'].value
        - hp_data['dps']['fft_tavg']['dps_ci'][95]['lbound'].value
    )
    if np.any(yerr[0] < 0):
        print('TADPS: Negative errorbar values found in lower bounds')
        inds = np.where(yerr[0] < 0)[0]
        pprint(inds)
        pprint(yerr[0, inds])
        yerr[0, inds] = np.abs(yerr[0, inds])
    yerr[1] = (
        hp_data['dps']['fft_tavg']['dps_ci'][95]['ubound'].value
        - hp_data['dps']['fft_tavg']['dps_mean'].value
    )
    if np.any(yerr[1] < 0):
        print('TADPS: Negative errorbar values found in upper bounds')
        inds = np.where(yerr[1] < 0)[0]
        pprint(inds)
        pprint(yerr[1, inds])
        yerr[1, inds] = np.abs(yerr[1, inds])
    ax.errorbar(
        delays.value,
        hp_data['dps']['fft_tavg']['dps_mean'].value,
        yerr=yerr,
        color='k',
        marker='o',
        capsize=3,
        ls='',
        lw=1.5,
        label='DPS Samples'
    )
    if hp_data['args'].n_ps_prior_bins > 0:
        prior_inds = slice(
            delays.size//2 - hp_data['args'].n_ps_prior_bins,
            delays.size//2 + hp_data['args'].n_ps_prior_bins + 1
        )
        delays_priors = delays[prior_inds]
        ax.hlines(
            [prior_lo, prior_hi],
            delays_priors[0].to('ns').value,
            delays_priors[-1].to('ns').value,
            color='0.5',
            ls='--',
            label='Prior'
        )
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_yscale('log')
    ax.grid(ls=':')
    legend = ax.legend(loc='upper right')
    legend.set(zorder=100)
    ax.set_xlabel(r'$\tau$ [ns]')
    ax.set_ylabel(fr"$P(\tau)$ [{vis_data['ds']['fft']['eor'].unit}$^2$]")
    fig.suptitle(suptitle)
    fig.tight_layout()


def get_chisq(
    vis_data, hp_data, suptitle=None, iterations=None, chisq_bad=None,
    log=False, plot=True
):
    args = hp_data['args']
    
    N = np.load(
        Path(args.noise_cov)
        / args.ant_str.replace('_', '-')
        / args.noise_cov_file
    )
    Ninv = np.linalg.inv(N)
    
    fg_basis_vecs = np.load(
        Path(args.fg_eig_dir)
        / args.ant_str.replace('_', '-')
        / args.fg_eig_file
    )
    fg_basis_vecs = fg_basis_vecs[:, :args.Nfgmodes]
    fg_amps = hp_data['fg_amps']
    fg_model = np.einsum('ijk,lk->ijl', fg_amps, fg_basis_vecs)
    Niter = fg_amps.shape[0]
    iter_nums = np.arange(Niter) + 1
    
    eor_model = hp_data['gcr']
    
    vis = vis_data['vis']['sum_noise']
    
    model = eor_model + fg_model
    chisqs = (
        np.abs(vis[None, :, :] - model)**2 * Ninv.diagonal()[None, None, :]
    )
    if 'chisq' not in hp_data:
        # Reduced chi-squared per iteration (averaged over time and frequency)
        red_chisq_per_iter = chisqs.mean(axis=(1, 2))
    else:
        red_chisq_per_iter = hp_data['chisq']

    if iterations is None and chisq_bad is not None:
        iter_inds = np.where(red_chisq_per_iter > chisq_bad)[0]
    else:
        if isinstance(iterations, int):
            iterations = [iterations]
        iter_inds = np.where(np.in1d(iter_nums, iterations))[0]
    
    if not plot:
        return chisqs
    else:
        fig, ax = plt.subplots()
        ax.plot(np.arange(Niter) + 1, red_chisq_per_iter, 'ko')
        if red_chisq_per_iter.max() / red_chisq_per_iter.min() > 10:
            ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'Reduced $\chi^2$')
        ax.grid(ls=':')
        fig.suptitle(suptitle)
        fig.tight_layout()
        
        plot_size = 4
        ncols = iter_inds.size
        if ncols*(1 + plot_size) > 100:
            dpi = 25
        else:
            dpi = 100

        # # Data
        # fig = plt.figure(figsize=(ncols*(1 + plot_size), plot_size), dpi=dpi)
        # grid = ImageGrid(
        #     fig, 111, (1, ncols), share_all=True, axes_pad=1.0,
        #     cbar_mode='each', cbar_pad=0.01
        # )
        # for i_ax, ax in enumerate(grid.axes_all):
        #     if log:
        #         norm = LogNorm()
        #     else:
        #         norm = None
        #     im = ax.imshow(np.abs(vis), norm=norm)
        #     ax.set_xlabel('Frequency Index')
        #     ax.set_ylabel('Time Index')
        #     _ = fig.colorbar(im, cax=ax.cax)
        #     ax.set_title(fr'$\mathbf{{d}}$, Iter. {iter_inds[i_ax]+1}')

        # # Model
        # fig = plt.figure(figsize=(ncols*(1 + plot_size), plot_size), dpi=dpi)
        # grid = ImageGrid(
        #     fig, 111, (1, ncols), share_all=True, axes_pad=1.0,
        #     cbar_mode='each', cbar_pad=0.01
        # )
        # for i_ax, ax in enumerate(grid.axes_all):
        #     if log:
        #         norm = LogNorm()
        #     else:
        #         norm = None
        #     im = ax.imshow(np.abs(model[iter_inds[i_ax]]), norm=norm)
        #     ax.set_xlabel('Frequency Index')
        #     ax.set_ylabel('Time Index')
        #     _ = fig.colorbar(im, cax=ax.cax)
        #     ax.set_title(fr'$\mathbf{{m}}$, Iter. {iter_inds[i_ax]+1}')
        
        # # Data - Model
        # fig = plt.figure(figsize=(ncols*(1 + plot_size), plot_size), dpi=dpi)
        # grid = ImageGrid(
        #     fig, 111, (1, ncols), share_all=True, axes_pad=1.0,
        #     cbar_mode='each', cbar_pad=0.01
        # )
        # for i_ax, ax in enumerate(grid.axes_all):
        #     im = ax.imshow(
        #         np.abs(vis) - np.abs(model[iter_inds[i_ax]]), cmap='magma'
        #     )
        #     ax.set_xlabel('Frequency Index')
        #     ax.set_ylabel('Time Index')
        #     _ = fig.colorbar(im, cax=ax.cax)
        #     ax.set_title(
        #         r'$\mathbf{d}-\mathbf{m}$, '
        #         + f'Iter. {iter_inds[i_ax]+1}'
        #     )

        # Chi-squared waterfalls
        fig = plt.figure(figsize=(ncols*(1 + plot_size), plot_size), dpi=dpi)
        grid = ImageGrid(
            fig, 111, (1, ncols), share_all=True, axes_pad=1.0,
            cbar_mode='each', cbar_pad=0.01
        )
        
        for i_ax, ax in enumerate(grid.axes_all):
            if log:
                norm = LogNorm()
            else:
                norm = None
            im = ax.imshow(chisqs[iter_inds[i_ax]], norm=norm)
            ax.set_xlabel('Frequency Index')
            ax.set_ylabel('Time Index')
            _ = fig.colorbar(im, cax=ax.cax)
            ax.set_title(fr'$\chi^2$, Iter. {iter_inds[i_ax]+1}')


def get_fg_model(
    hp_data, Nburn=0, return_vec=True, bl_str="0-1", Nfreqs=None
):
    """
    Generate foreground model visibilities from hydra-pspec outputs.

    Parameters
    ----------
    hp_data : dict
        Dictionary containing the results from a hydra-pspec analysis
        including the foreground model amplitudes `hp_data['fg_amps']`
        and the command line arguments `hp_data['args']` containing paths
        to the foreground model basis vectors.
    Nburn : int
        Number of samples to skip (to avoid burn in).
    return_vec : bool
        If True, return a column matrix of foreground model basis vectors.
    bl_str : str
        Hyphen-separated antenna pair string, e.g. "ant1-ant2" for directory
        indexing.  Defaults to "0-1".
    Nfreqs : int
        Number of frequency channels.  Used if generating analytic polynomials.

    Returns
    -------
    fg_model : ndarray
        Array containing foreground model visibilities with shape
        (Niter, Ntimes, Nfreqs).
    fg_model_vecs : ndarray
        Array containing the foreground model basis vectors as columns with
        shape (Nfreqs, Nfgmodes).

    """
    args = hp_data['args']

    # Get foreground basis vectors
    if 'fg_basis' in args:
        # Foreground model comprised of analytic functions
        if args.fg_basis.lower() == 'legendre':
            poly_func = scipy.special.legendre
        elif args.fg_basis.lower() == 'hermite':
            poly_func = scipy.special.hermite
        elif args.fg_basis.lower() == 'chebyshev':
            poly_func = scipy.special.chebyu
        # fg_model_vecs should have shape (Nfreqs, Nfgmodes)
        fg_model_vecs = np.array([
            poly_func(i)(np.linspace(-1., 1., Nfreqs))
            for i in range(args.Nfgmodes)
        ]).T
    else:
        # Load foreground model basis vectors from disk
        if 'fg_eig_dir' in args:
            # Legacy command line argument name (deprecated)
            fg_model_dir = Path(args.fg_eig_dir)
            fg_model_path = fg_model_dir / bl_str / args.fg_eig_file
            fg_model_vecs = np.load(fg_model_path)
            fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]
        else:
            if Path(args.fgmodes).is_dir():
                fg_model_dir = Path(args.fgmodes)
                fg_model_file = args.fgmodes_file
                if bl_str is not None:
                    fg_model_path = fg_model_dir / bl_str / fg_model_file
                else:
                    fg_model_path = fg_model_dir / fg_model_file
            else:
                fg_model_path = Path(args.fgmodes)
            fg_model_vecs = np.load(fg_model_path)
            fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]

    fg_amps = hp_data['fg_amps']  # shape (Niter, Nfreqs, Nfgmodes)
    fg_model = np.einsum('ijk,kl->ijl', fg_amps, fg_model_vecs.T)

    if return_vec:
        return fg_model, fg_model_vecs
    else:
        return fg_model


def get_models(hp_data, Nburn=0, bl_str=None, fg_model_dir=None):
    args = hp_data['args']
    if bl_str is None:
        bl_str = args.ant_str.replace('_', '-')
    
    # EoR
    # Gaussian constrained realizations
    eor_model = hp_data['gcr']  # shape (Niter, Ntimes, Nfreqs)

    # Foregrounds
    # Load foreground model basis vectors
    if 'fg_basis' in args:
        if args.fg_basis.lower() == 'legendre':
            poly_func = scipy.special.legendre
        elif args.fg_basis.lower() == 'hermite':
            poly_func = scipy.special.hermite
        elif args.fg_basis.lower() == 'chebyshev':
            poly_func = scipy.special.chebyu
        # fg_model_vecs should have shape (Nfreqs, Nfgmodes)
        fg_model_vecs = np.array([
            poly_func(i)(np.linspace(-1., 1., eor_model.shape[-1]))
            for i in range(args.Nfgmodes)
        ]).T
    elif fg_model_dir is None:
        # Load foreground model basis vectors
        if 'fg_eig_dir' in args:
            fg_model_dir = Path(args.fg_eig_dir)
            fg_model_path = fg_model_dir / bl_str / args.fg_eig_file
            fg_model_vecs = np.load(fg_model_path)
            fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]
        else:
            if Path(args.fgmodes).is_dir():
                fg_model_dir = Path(args.fgmodes)
                fg_model_file = args.fgmodes_file
                if bl_str is not None:
                    fg_model_path = fg_model_dir / bl_str / fg_model_file
                else:
                    fg_model_path = fg_model_dir / fg_model_file
            else:
                fg_model_path = Path(args.fgmodes)
            fg_model_vecs = np.load(fg_model_path)
            fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]
    # Foreground model amplitudes
    fg_amps = hp_data['fg_amps']  # shape (Niter, Nfreqs, Nfgmodes)
    fg_model = np.einsum('ijk,kl->ijl', fg_amps, fg_model_vecs.T)

    # EoR + Foregrounds
    model_pwm = np.average(
        (eor_model + fg_model)[Nburn:],
        weights=hp_data['ln_post'][Nburn:],  # Posterior-weighted mean
        axis=0
    )
    
    return eor_model, fg_model, model_pwm


def compare_data_and_model(
    vis_data,
    hp_data,
    Nburn=0,
    taper=False,
    comp_type='reim',
    fg_model_dir=None
):
    args = hp_data['args']

    # DATA
    data = vis_data['vis']['sum_noise'].copy()  # shape (Ntimes, Nfreqs)
    if taper:
        data *= blackmanharris(data.shape[1])[None, :]

    # MODEL
    eor_model, fg_model, model = get_models(
        hp_data, Nburn=Nburn, fg_model_dir=fg_model_dir
    )
    
    fig = compare_data(
        data, model, labels=['Data', 'Model'], comp_type=comp_type
    )


def model_comparison(
    args, freqs, d, S_initial, signal_cr, signal_S, signal_ps, fg_amps,
    taper=False, suptitle='', i_iter=0
):
    bl_str = args.ant_str.replace('_', '-')
    bl = (int(bl_str.split('-')[0]), int(bl_str.split('-')[1]))
    Ntimes, Nfreqs = d.shape

    # EoR
    eor_dir = Path(args.sigcov0).parent.parent
    uvd = UVData()
    uvd.read(eor_dir / 'vis-circ.uvh5')
    uvd.conjugate_bls()
    vis_eor = (
        uvd.get_data(bl + ('xx',), force_copy=True)
        + uvd.get_data(bl + ('yy',), force_copy=True)
    )

    if taper:
        # Taper matrix
        T = np.diag(blackmanharris(Nfreqs))
        d = np.einsum('ii,ji->ji', T, d.copy())
        vis_eor = np.einsum('ii,ji->ji', T, vis_eor)
        S_initial = T @ S_initial.copy() @ T
    else:
        T = np.eye(Nfreqs)

    # EoR
    # ---------
    
    # EoR model
    eor_model = signal_cr[i_iter]

    # Visibility comparison
    fig = compare_data(
        vis_eor, eor_model, labels=['Data', 'Model'], comp_type='re,im',
        suptitle=suptitle+'EoR Visibility Comparison'
    )

    # Delay power spectrum comparison
    dps_kwargs = dict(mean_sub=False, taper=False, norm=False)
    vis_eor_ds, vis_eor_dps, delays = delay_power_spectrum(
        vis_eor, freqs, **dps_kwargs
    )
    eor_model_ds, eor_model_dps, _ = delay_power_spectrum(
        eor_model, freqs, **dps_kwargs
    )

    fig = compare_data(
        vis_eor_dps.value, eor_model_dps.value, labels=['Data', 'Model'],
        comp_type='re', plot_size=5, frac_diff=False,
        suptitle=suptitle+'EoR Delay Power Spectrum Comparison'
    )

    # Time-averaged delay power spectrum
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True, gridspec_kw={'hspace': 0.1}
    )
    
    ax = axs[0]
    ax.set_title(suptitle+'Time-Averaged EoR Delay Power Spectrum Comparison')
    ax.semilogy(
        delays.to('ns'), vis_eor_dps.mean(axis=0), 'k-', label='Data'
    )
    ax.semilogy(
        delays.to('ns'), eor_model_dps.mean(axis=0), 'C0--', label='GCR Sample'
    )
    ax.semilogy(
        delays.to('ns'), signal_ps[i_iter], 'C1--', label='DPS Sample'
    )
    ax.legend(loc='upper right')
    ax.set_ylabel(r'$P(\tau)$ [Jy$^2$]')

    ax = axs[1]
    ax.plot(
        delays.to('ns'),
        vis_eor_dps.mean(axis=0) - eor_model_dps.mean(axis=0),
        'C0--'
    )
    ax.plot(
        delays.to('ns'),
        vis_eor_dps.mean(axis=0).value - signal_ps[i_iter],
        'C1--'
    )
    ax.set_xlabel(r'$\tau$ [ns]')
    ax.set_ylabel('Data - Model')

    for ax in axs:
        ax.grid()


    # EoR + FGs
    # ---------
    
    # FG model
    fg_model_dir = Path(args.fg_eig_dir)
    fg_model_path = fg_model_dir / bl_str / args.fg_eig_file
    fg_model_vecs = np.load(fg_model_path)
    fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]  # shape (Nfreqs, Nfgmodes)
    # Foreground model amplitudes
    fg_model = np.einsum('ijk,kl->ijl', fg_amps, fg_model_vecs.T)[i_iter]

    # EoR+FG model
    model = eor_model + fg_model

    # Visibility comparison
    fig = compare_data(
        d, model, labels=['Data', 'Model'], comp_type='re,im',
        suptitle=suptitle+'EoR + FG Visibility Comparison'
    )

    # Delay power spectrum comparison
    d_ds, d_dps, delays = delay_power_spectrum(d, freqs, **dps_kwargs)
    model_ds, model_dps, _ = delay_power_spectrum(model, freqs, **dps_kwargs)

    fig = compare_data(
        d_dps.value, model_dps.value, labels=['Data', 'Model'], comp_type='re',
        plot_size=5, frac_diff=True,
        suptitle=suptitle+'EoR + FG Delay Power Spectrum Comparison'
    )

    # Time-averaged delay power spectrum
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True, gridspec_kw={'hspace': 0.1}
    )
    
    ax = axs[0]
    ax.set_title(
        suptitle+'Time-Averaged EoR + FG Delay Power Spectrum Comparison'
    )
    ax.semilogy(
        delays.to('ns'), d_dps.mean(axis=0), 'k-', label='Data'
    )
    ax.semilogy(
        delays.to('ns'), model_dps.mean(axis=0), 'C0--', label='Model'
    )
    ax.legend(loc='upper right')
    ax.set_ylabel(r'$P(\tau)$ [Jy$^2$]')

    ax = axs[1]
    ax.plot(
        delays.to('ns'),
        d_dps.mean(axis=0) - model_dps.mean(axis=0),
        'k-'
    )
    ax.set_xlabel(r'$\tau$ [ns]')
    ax.set_ylabel('Data - Model')

    for ax in axs:
        ax.grid()

    # Covariance matrix comparison
    fig = compare_data(
        S_initial, signal_S, labels=['Input', 'MAP Est.'],
        comp_type='re', plot_size=5,
        suptitle=suptitle+'EoR Covariance Comparison'
    )


def fg_model_inspection(
    vis_data, hp_data, vis_fgs, i_t=0, conf_interval=95, fgcolor='C3',
    plot_size=3.5, suptitle='', bl_str='0-1'
):
    args = hp_data['args']
    
    # ------------
    # Get FG model
    # ------------
    # Load foreground model basis vectors
    if 'fg_eig_dir' in args:
        fg_model_dir = Path(args.fg_eig_dir)
        fg_model_path = fg_model_dir / bl_str / args.fg_eig_file
        fg_model_vecs = np.load(fg_model_path)
        fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]  # shape (Nfreqs, Nfgmodes)
    else:
        if Path(args.fgmodes).is_dir():
            fg_model_dir = Path(args.fgmodes)
            fg_model_file = args.fgmodes_file
            if bl_str is not None:
                fg_model_path = fg_model_dir / bl_str / fg_model_file
            else:
                fg_model_path = fg_model_dir / fg_model_file
        else:
            fg_model_path = Path(args.fgmodes)
        fg_model_vecs = np.load(fg_model_path)
        fg_model_vecs = fg_model_vecs[:, :args.Nfgmodes]
    # Foreground model amplitudes
    fg_amps = hp_data['fg_amps'][:, i_t]  # shape (Niter, Nfgmodes)
    fg_model = fg_amps @ fg_model_vecs.T  # shape (Niter, Nfreqs)
    # fg_model = np.einsum('ijk,kl->ijl', fg_amps, fg_model_vecs.T)

    # Get FG amp confidence intervals
    fg_amps_ci = {}
    for i_fg in range(args.Nfgmodes):
        percentile = conf_interval/2 + 50
        
        lbound = (
            np.percentile(fg_amps[:, i_fg].real, 100-percentile, axis=0)
            + 1j*np.percentile(fg_amps[:, i_fg].imag, 100-percentile, axis=0)
        )
        ubound = (
            np.percentile(fg_amps[:, i_fg].real, percentile, axis=0)
            + 1j*np.percentile(fg_amps[:, i_fg].imag, percentile, axis=0)
        )
        fg_amps_ci[i_fg] = dict(lbound=lbound, ubound=ubound)

    # FG model confidence interval
    lbound = (
        np.percentile(fg_model.real, 100-percentile, axis=0)
        + 1j*np.percentile(fg_model.imag, 100-percentile, axis=0)
    )
    ubound = (
        np.percentile(fg_model.real, percentile, axis=0)
        + 1j*np.percentile(fg_model.imag, percentile, axis=0)
    )
    fg_model_ci = dict(lbound=lbound, ubound=ubound)


    # Plot FG model and confidence intervals
    freqs = vis_data['freqs'].to('MHz')
    
    fig, axs = plt.subplots(
        2, 2, figsize=(12, 6), sharex=True,
        gridspec_kw={'hspace': 0.1}
    )

    # Real component
    ax = axs[0, 0]
    ax.set_title('Real')
    ax.plot(
        freqs,
        vis_fgs[i_t].real,
        'k-',
        label='True'
    )
    ax.plot(
        freqs,
        fg_model.mean(axis=0).real,
        f'{fgcolor}-',
        label='Sample mean'
    )
    ax.fill_between(
        freqs.value,
        fg_model_ci['lbound'].real,
        fg_model_ci['ubound'].real,
        color=fgcolor,
        alpha=0.5,
        label='95% conf.'
    )
    ax.legend()
    ax.set_ylabel('FG Spectrum')

    ax = axs[1, 0]
    diff, diff_err = calc_frac_diff(
        fg_model.mean(axis=0).real,
        vis_fgs[i_t].real,
        x_lbound=fg_model_ci['lbound'].real,
        x_ubound=fg_model_ci['ubound'].real
    )
    ax.plot(freqs, diff, f'{fgcolor}-')
    ax.fill_between(
        freqs.value,
        diff - diff_err[0],
        diff + diff_err[1],
        color=fgcolor,
        alpha=0.5
    )
    ax.set_ylabel('Fractional Difference')
    ax.set_xlabel('Frequency [MHz]')

    # Imaginary component
    ax = axs[0, 1]
    ax.set_title('Imaginary')
    ax.plot(
        freqs,
        vis_fgs[i_t].imag,
        'k-',
        label='True'
    )
    ax.plot(
        freqs,
        fg_model.mean(axis=0).imag,
        f'{fgcolor}-',
        label='Sample mean'
    )
    ax.fill_between(
        freqs.value,
        fg_model_ci['lbound'].imag,
        fg_model_ci['ubound'].imag,
        color=fgcolor,
        alpha=0.5,
        label='95% conf.'
    )

    ax = axs[1, 1]
    diff, diff_err = calc_frac_diff(
        fg_model.mean(axis=0).imag,
        vis_fgs[i_t].imag,
        x_lbound=fg_model_ci['lbound'].imag,
        x_ubound=fg_model_ci['ubound'].imag
    )
    ax.plot(freqs, diff, f'{fgcolor}-')
    ax.fill_between(
        freqs.value,
        diff - diff_err[0],
        diff + diff_err[1],
        color=fgcolor,
        alpha=0.5
    )
    ax.set_xlabel('Frequency [MHz]')

    for ax in axs.flatten():
        ax.grid()
    fig.suptitle(suptitle+fr'FG Model Comparison ($i_{{t}}$ = {i_t})')


    # Plot FG model basis vectors with 95% confidence intervals    
    Nfgmodes = fg_amps.shape[1]
    nrows = Nfgmodes
    ncols = 2
    figsize = (1.2 * ncols * plot_size, 0.5*nrows * plot_size)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=True,
        gridspec_kw={'hspace': 0.1, 'wspace': 0.5}
    )

    for i_fg in range(Nfgmodes):
        basis_vec = fg_model_vecs[:, i_fg]

        # Real component
        ax = axs[i_fg, 0]
        if i_fg == 0:
            ax.set_title('Real')
        ax.plot(
            freqs,
            fg_amps[:, i_fg].mean().real * basis_vec.real,
            'k-',
            label='Sample mean'
        )
        ax.fill_between(
            freqs.value,
            fg_amps_ci[i_fg]['lbound'].real * basis_vec.real,
            fg_amps_ci[i_fg]['ubound'].real * basis_vec.real,
            color='k',
            alpha=0.5,
            label='95% conf.'
        )

        # Imaginary component
        ax = axs[i_fg, 1]
        if i_fg == 0:
            ax.set_title('Imaginary')
        ax.plot(
            freqs,
            fg_amps[:, i_fg].mean().imag * basis_vec.imag,
            'k-',
            label='Sample mean'
        )
        ax.fill_between(
            freqs.value,
            fg_amps_ci[i_fg]['lbound'].imag * basis_vec.imag,
            fg_amps_ci[i_fg]['ubound'].imag * basis_vec.imag,
            color='k',
            alpha=0.5,
            label='95% conf.'
        )

    for ax in axs.flatten():
        ax.grid()
    for ax in axs[-1]:
        ax.set_xlabel('Frequency [MHz]')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.suptitle(suptitle + 'FG Model Basis Vectors')
    fig.subplots_adjust(top=0.925)
    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
        ncol=2, frameon=False
    )


def fg_model_dps_plot(
    dps_fgs,
    hp_data,
    freqs, 
    post_as_weights=False,
    Nburn=0,
    taper=True,
    figsize=(10, 8),
    conf_interval=95,
    fg_color='C1',
    fd_ylim=[-1, 1],
    xlim=None,
    suptitle=None
):
    """
    Parameters
    ----------
    dps_fgs : array_like
        Array of foreground delay power spectrum amplitudes with shape
        (Ntimes, Nfreqs).
    hp_data : dict
        Dictionary containing the results from a hydra-pspec analysis
        including the foreground model amplitudes `hp_data['fg_amps']`
        and the command line arguments `hp_data['args']` containing paths
        to the foreground model basis vectors.
    freqs : array_like or `astropy.units.Quantity`
        Array of frequencies in Hz (or Hz compatible unit if Quantity).
    post_as_weights : bool
        If True, use the posterior to weight averages over the iteration axis.
        Defaults to False, i.e. calculate the sample mean.
    Nburn : int
        Number of iterations to ignore to avoid e.g. burn in.
    taper : bool
        If True, apply a Blackman-Harris tapering function prior to delay
        transformation.
    figsize : tuple
        Figure size as (width, height).
    conf_interval : float
        Confidence interval to calculate and plot.
    fd_ylim : array-like
        y-axis limits for the fractional difference subplot.  Defaults to
        [-1, 1].
    xlim : array-like
        x-axis limits.  Defaults to displaying the whole x axis.
    suptitle : str
        Figure suptitle.

    Returns
    -------
    fig : `matplotlib.pyplot.figure`
        Figure instance.

    """
    Nfreqs = freqs.size
    if not isinstance(freqs, Quantity):
        freqs = Quantity(freqs, unit='Hz')

    # Incoherent average of the true foreground delay power spectrum
    dps_fgs_tavg = dps_fgs.mean(axis=0)

    if not 'fgs' in hp_data:
        # Foreground model
        fg_model, fg_model_vecs = get_fg_model(hp_data, Nburn=Nburn)
        _, dps_fg_model, delays = delay_power_spectrum(
            fg_model, freqs, axis=2, mean_sub=False, taper=taper
        )
        # Incoherent averages
        dps_fg_model_tavg = dps_fg_model.mean(axis=1)
        # Posterior-weighted mean foreground delay power spectrum
        if 'ln_post' in hp_data and post_as_weights:
            weights = hp_data['ln_post']
        else:
            weights = np.ones(fg_model.shape[0])
        dps_fg_model_tavg_mean = np.average(
            dps_fg_model_tavg[Nburn:], weights=weights[Nburn:], axis=0
        )
        # Calculate the confidence interval
        percentile = conf_interval/2 + 50
        dps_fg_model_tavg_lbound = np.zeros(Nfreqs, dtype=dps_fg_model.dtype)
        dps_fg_model_tavg_ubound = np.zeros(Nfreqs, dtype=dps_fg_model.dtype)
        for i_dly in range(Nfreqs):
            dps_fg_model_tavg_lbound[i_dly] = weighted_quantile(
                dps_fg_model_tavg[Nburn:, i_dly].value,
                1-percentile/100,
                weights=hp_data['ln_post']
            )
            dps_fg_model_tavg_ubound[i_dly] = weighted_quantile(
                dps_fg_model_tavg[Nburn:, i_dly].value,
                percentile/100,
                weights=hp_data['ln_post']
            )
    else:
        delays = np.fft.fftshift(
            np.fft.fftfreq(Nfreqs, d=(freqs[1]-freqs[0]).to('1/ns'))
        )
        
        if taper:
            ps_key = 'tp_tavg'
        else:
            ps_key = 'fft_tavg'
        dps_fg_model_tavg_mean = hp_data['dps'][ps_key]['fgs_mean']
        dps_fg_model_tavg_lbound = (
            hp_data['dps'][ps_key]['fgs_ci'][conf_interval]['lbound']
        ).value
        dps_fg_model_tavg_ubound = (
            hp_data['dps'][ps_key]['fgs_ci'][conf_interval]['ubound']
        ).value

    fig, axs = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'hspace': 0.1, 'height_ratios': [1.5, 1]}
    )

    ax = axs[0]
    ax.set_ylabel(r"$P(\tau)$ [arb. units]")
    ax.plot(delays.to('ns'), dps_fgs.mean(axis=0), 'k-', label='True FGs')
    ax.plot(
        delays.to('ns'),
        dps_fg_model_tavg_mean,
        ls='-',
        marker='o',
        color=fg_color,
        alpha=0.8,
        label='FG Model'
    )
    ax.fill_between(
        delays.to('ns').value,
        dps_fg_model_tavg_lbound,
        dps_fg_model_tavg_ubound,
        color='C1',
        alpha=0.3,
        label=r"{conf_interval}% conf."
    )
    ax.set_yscale("log")

    ax = axs[1]
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel("Fractional Diff.")

    fd, fd_err = calc_frac_diff(
        dps_fg_model_tavg_mean.value,
        dps_fgs_tavg,
        x_lbound=dps_fg_model_tavg_lbound,
        x_ubound=dps_fg_model_tavg_ubound
    )
    fd_err = fix_neg_errbars(fd_err, message="GCR FGs vs True FGs")
    ax.plot(
        delays.to('ns'),
        fd,
        ls='-',
        marker='o',
        color=fg_color,
        alpha=0.8
    )
    ax.fill_between(
        delays.to('ns').value,
        fd - fd_err[0],
        fd + fd_err[1],
        color=fg_color,
        alpha=0.3,
        label=f"{conf_interval}% conf."
    )
    ax.legend(loc='upper right')
    ax.set_ylim(fd_ylim)

    for ax in axs:
        ax.grid()
        if xlim is not None:
            ax.set_xlim(xlim)

    fig.suptitle(suptitle)

    return fig
