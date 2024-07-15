import argparse
import copy
import itertools
import os
import re
import time
import yaml
import time

import numpy as np
from astropy import units
from scipy import stats
from scipy.interpolate import RectBivariateSpline, interp1d
from warnings import warn

from data import DATA_PATH
from pyuvdata import UVData
from pyuvdata.utils import polstr2num
from hera_cal.abscal import get_d2m_time_map
from hera_cal.io import HERAData, to_HERAData
from hera_cal.utils import lst_rephase
from hera_sim import Simulator
from hera_sim.noise import thermal_noise

import hera_sim
if hera_sim.__version__.startswith('0'):
    from hera_sim.rfi import _listify
else:
    from hera_sim.utils import _listify

# ------- Functions for adding systematics ------- #

def add_noise(sim, Trx=100, seed=None, ret_cmp=True):
    """
    Add thermal noise to a UVData object, based on its auto-correlations.

    Parameters
    ----------
    sim : :class:`pyuvdata.UVData` or :class:`hera_sim.Simulator` instance
        The simulation data to add noise to
    Trx : float, optional
        The receiver temperature, in Kelvin.
    seed : int, optional
        The random seed. Not used if not specified. Use value 'random' 
        to have a seed automatically generated.
    """
    sim = _sim_to_uvd(sim)

    original_lsts = copy.deepcopy(sim.lst_array)
    unwrapped_lsts = sim.lst_array
    unwrapped_lsts[unwrapped_lsts < unwrapped_lsts[0]] += 2 * np.pi
    lsts = np.unique(unwrapped_lsts)
    sim.lst_array = unwrapped_lsts
    freqs = np.unique(sim.freq_array)
    freqs_GHz = freqs / 1e9  # GHz

    # Find out which antenna has the autocorrelation data, in case we
    # add noise before inflating.
    antpair = next(ants for ants in sim.get_antpairs() if ants[0] == ants[1])

    # Set up to use the autos to set the noise level
    beam_poly = np.load(os.path.join(DATA_PATH, "RIMEz_beam_poly.npy"))
    omega_p = np.polyval(beam_poly, freqs_GHz)
    Jy_to_K = hera_sim.noise.jy2T(freqs_GHz, omega_p) / 1000

    if seed is not None:
        seed = _gen_seed(seed)
        np.random.seed(seed)

    noise = np.zeros_like(sim.data_array, dtype=complex) if ret_cmp else sim.data_array

    for pol in ('xx', 'yy'):
        autos = sim.get_data(*antpair, pol) * Jy_to_K[None, :]
        interp = RectBivariateSpline(lsts, freqs_GHz, autos.real)

        for bl in sim.get_antpairs():
            blts, _, indx = sim._key2inds(bl + (pol,))
            
            # Cross-correlations get this treatment
            if bl[0] != bl[1]:
                noise[blts, 0, :, indx[0]] += thermal_noise(
                    lsts=lsts, fqs=freqs_GHz, Tsky_mdl=interp, Trx=Trx, omega_p=omega_p,
                )
            else:
                # ... but the autos need to be *real*
                noise[blts, 0, :, indx[0]] += Trx / Jy_to_K

    if ret_cmp:
        sim.data_array += noise

    # Return simulation LSTs back to their original form, then return results.
    sim.lst_array = original_lsts
    
    if ret_cmp:
        return sim, noise
    else:
        return sim


def add_gains(
    sim, 
    seed=None, 
    time_vary_params=None,
    ret_cmp=True,
    **gain_params
):
    """
    Add per-antenna bandpass gains to the simulation.

    Parameters
    ----------
    sim : :class:`hera_sim.Simulator` or :class:`pyuvdata.UVData`
        The object containing the simulation data and metadata.
    seed : int, optional
        The random seed. Not used if not specified. Use value 'random' 
        to have a seed automatically generated.
    time_vary_params : dict, optional
        Parameters for adding time variation to the gains. Keys should 
        be any of ('amp', 'amplitude', 'phs', 'phase'). Values should 
        be the set of variation parameters (parameters prefixed by 
        'variation' in the ``vary_gains_in_time`` function) to be used.
        Default behavior is to leave the gains constant in time.
        See ``vary_gains_in_time`` function for details on parameters.
    **gain_params
        The parameters for simulating bandpass gains.

    Returns
    -------
    corrupted_sim : :class:`pyuvdata.UVData`
        Simulation with bandpass gains applied. 
    gains : dict
        Dictionary mapping antenna numbers to bandpass gains as a 
        function of frequency (and potentially time).
    """
    # Setup
    sim = _sim_to_uvd(sim)
    freqs_GHz = np.unique(sim.freq_array) / 1e9
    ants = sim.antenna_numbers
    seed = _gen_seed(seed)

    # Simulate and apply the gains
    if seed is not None:
        np.random.seed(seed)
    gains = hera_sim.sigchain.gen_gains(freqs_GHz, ants, **gain_params)
    apply_gains(sim, gains, time_vary_params)

    if ret_cmp:
        return sim, gains
    else:
        return sim

def add_reflections(
    sim, 
    seed=None, 
    dly=1200, 
    dly_spread=0, 
    amp=1e-3, 
    amp_scale=0, 
    time_vary_params=None,
    ret_cmp=True
):
    """
    Add per-antenna reflection gains to the simulation.

    Parameters
    ----------
    sim : :class:`hera_sim.Simulator` or :class:`pyuvdata.UVData`
        The object containing the simulation data and metadata.
    seed : int, optional
        The random seed. Not used if not specified. Use value 'random' 
        to have a seed automatically generated.
    dly : float, optional
        Delay at which the reflection appears, in nanoseconds. 
        Default is 1200 ns.
    dly_spread : float, optional
        Absolute amount the delays vary between antennas, in ns. 
        Default is 0 ns.
    amp : float, optional
        Amplitude of the reflection. Default is 1e-3.
    amp_scale : float, optional
        Fractional variation in the reflection amplitude between antennas.
        Default is 0. 
    time_vary_params : dict, optional
        Parameters for adding time variation to the gains. Keys should 
        be any of ('amp', 'amplitude', 'phs', 'phase'). Values should 
        be the set of variation parameters (parameters prefixed by 
        'variation' in the ``vary_gains_in_time`` function) to be used.
        Default behavior is to leave the gains constant in time.
        See ``vary_gains_in_time`` function for details on parameters.

    Returns
    -------
    corrupted_sim : :class:`pyuvdata.UVData`
        Simulation with reflection gains applied. 
    gains : dict
        Dictionary mapping antenna numbers to reflection gains as a 
        function of frequency (and potentially time).
    """
    # Setup
    sim = _sim_to_uvd(sim)
    freqs_GHz = np.unique(sim.freq_array) / 1e9
    ants = sim.antenna_numbers
    Nants = len(ants)
    seed = _gen_seed(seed)
    gains = {ant: np.ones(freqs_GHz.size, dtype=complex) for ant in ants}

    # Support multi-reflections.
    amps = _listify(amp)
    dlys = _listify(dly)
    seeds = _listify(seed)
    dly_spreads = _listify(dly_spread)
    amp_scales = _listify(amp_scale)
    if len(seeds) == 1:
        seeds *= len(amps)
    if len(dly_spreads) == 1:
        dly_spreads *= len(amps)
    if len(amp_scales) == 1:
        amp_scales *= len(amps)
    if any(
        len(pair[0]) != len(pair[1])
        for pair in itertools.combinations(
            (amps, dlys, seeds, dly_spreads, amp_scales), 2
        )
    ):
        raise ValueError(
            "It appears as if you are trying to simulate multiple reflections "
            "but have provided parameters with different lengths. Please check "
            "your input to the reflection simulation."
        )

    iterator = zip(amps, dlys, seeds, dly_spreads, amp_scales)
    for amp, dly, seed, dly_spread, amp_scale in iterator:
        # Randomize parameters in a realistic way.
        if seed is not None:
            np.random.seed(int(seed))
        delays_ns = stats.norm.rvs(dly, dly_spread, Nants)
        phases = stats.uniform.rvs(0, 2*np.pi, Nants)
        amplitudes = amp * stats.norm.rvs(1, amp_scale, Nants)

        # Simulate and apply the gains.
        reflections = hera_sim.sigchain.gen_reflection_gains(
            freqs_GHz, ants, amp=amplitudes, dly=delays_ns, phs=phases
        )
        gains = {ant: gain * reflections[ant] for ant, gain in gains.items()}

    apply_gains(sim, gains, time_vary_params)

    if ret_cmp:
        return sim, gains
    else:
        return sim


def add_reflection_spectrum(
    sim,
    seed=None,
    Ncopies=20,
    amp_range=(-3, -4),
    amp_scale=0.0001,
    dly_rng=(200, 1000),
    dly_spread=30,
    time_vary_params=None,
    ret_cmp=True,
):
    """
    Add reflections at a range of delays.

    Parameters
    ----------
    sim: :class:`hera_sim.Simulator` or :class:`pyuvdata.UVData`
        The object containing the simulation data and metadata.
    seed: int, optional
        The random seed. Not used if not specified. Use value 'random' 
        to have a seed automatically generated.
    Ncopies: int, optional
        Number of reflections to introduce. Default is 20.
    amp_range: tuple of float, optional
        Range of reflection amplitudes, on a base-10 log scale. Default is
        to have reflection amplitudes range from 10^-3 to 10^-4.
    amp_scale: float, optional
        Amount of jitter to apply to amplitudes, as a fraction of the
        original amplitudes. Default is 0.01% jitter.
    dly_rng: tuple of float, optional
        Range of delays, inclusive, at which to insert reflections, in ns.
        Default is to insert reflections from 200 ns to 1000 ns.
    dly_spread: float, optional
        Amount of jitter to apply to delays, in ns. Default is to introduce
        jitter with a 30 ns standard deviation.
    time_vary_params: dict, optional
        Dictionary of parameters to use for introducing time variation to the
        reflection gains. Default is to not vary the gains in time.
    ret_cmp: bool, optional
        Whether to return the simulated reflection gains alongside the modified
        simulation. Default is to return the gains.

    Returns
    -------
    corrupted_sim: :class:`pyuvdata.UVData`
        Simulation with reflection gains applied. 
    gains: dict
        Dictionary mapping antenna numbers to reflection gains as a 
        function of frequency (and potentially time). Only returned if ``ret_cmp``
        is set to True.
    """
    # Make sure each reflection gets its own jitter.
    seed = _gen_seed(seed)
    if seed is not None:
        np.random.seed(seed)
        seed = np.random.uniform(0, 2**32, Ncopies)
    # Just a thin wrapper around add_reflections.
    amps = np.logspace(*amp_range, Ncopies)
    dlys = np.linspace(*dly_rng, Ncopies)
    return add_reflections(
        sim=sim,
        seed=seed,
        amp=amps,
        amp_scale=amp_scale,
        dly=dlys,
        dly_spread=dly_spread,
        time_vary_params=time_vary_params,
        ret_cmp=ret_cmp,
    )


def add_xtalk(
    sim, 
    seed=None, 
    Ncopies=10, 
    amp_range=(-4,-6), 
    dly_rng=(900,1300),
    ret_cmp=True
):
    """
    Add cross-coupling crosstalk to the simulation's cross-correlations.

    Parameters
    ----------
    sim : :class:`hera_sim.Simulator` or :class:`pyuvdata.UVData`
        The object containing the simulation data and metadata.
    seed : int, optional
        The random seed. Not used if not specified. Use value 'random' 
        to have a seed automatically generated.
    Ncopies : int, optional
        Number of delays at which to generate crosstalk. Default is 
        to use 10 copies.
    amp_range : tuple of float, optional
        Base-10 logarithm of the amplitude of the crosstalk at the 
        first and last delays where crosstalk is injected. Default 
        is to use -4 and -6.
    dly_rng : tuple of float, optional
        Minmum and maximum delays, in ns, at which to inject 
        crosstalk. Crosstalk is inserted at ``Ncopies`` delays spaced 
        linearly between the minimum and maximum delays. 
        Default is 900-1300 ns.

    Returns
    -------
    corrupted_sim : :class:`pyuvdata.UVData`
        Simulation with crosstalk applied.
    xtalk : np.ndarray
        Data for the crosstalk visibilities.
    """
    # Setup
    sim = _sim_to_uvd(sim)
    freqs_GHz = np.unique(sim.freq_array) / 1e9
    amps = np.logspace(*amp_range, Ncopies)
    dlys = np.linspace(*dly_rng, Ncopies)
    seed = _gen_seed(seed)

    if ret_cmp:
        xtalk = np.zeros_like(sim.data_array, dtype=complex)
    if seed is not None:
        np.random.seed(seed)
    for antpairpol in sim.get_antpairpols():
        print(antpairpol)
        ai, aj, pol = antpairpol
        if ai == aj:
            continue
        blt_inds, _, pol_inds = sim._key2inds(antpairpol)
        this_slice = (blt_inds, 0, slice(None), pol_inds[0])

        # Add some jitter to the delays and amplitudes.
        ddlys = stats.norm.rvs(1, 1e-2, Ncopies)
        damps = stats.norm.rvs(1, 1e-4, Ncopies)

        # Generate random phases and pull the autocorrelation.
        phs = stats.uniform.rvs(0, 2*np.pi, Ncopies)
        autovis = sim.get_data(ai, aj, pol)
        if ret_cmp:
            xtalk[this_slice] = gen_xtalk(
                autovis, freqs_GHz, amps * damps, dlys + ddlys, phs
            )
        else:
            sim.data_array[this_slice] += gen_xtalk(
                autovis, freqs_GHz, amps * damps, dlys + ddlys, phs
            )
        del autovis

    if ret_cmp:
        sim.data_array += xtalk
        return sim, xtalk
    else:
        return sim

def apply_gains(sim, gains, time_vary_params=None):
    """Apply per-antenna gains to a simulation."""
    # Support gain variation in time.
    times = np.unique(sim.time_array)
    time_vary_params = time_vary_params or {}
    for mode, vary_params in time_vary_params.items():
        gains = vary_gains_in_time(gains, times, mode, **vary_params)
    for antpairpol in sim.get_antpairpols():
        blt_inds, _, pol_inds = sim._key2inds(antpairpol)
        this_slice = (blt_inds, 0, slice(None), pol_inds[0])
        vis = sim.get_data(antpairpol)
        sim.data_array[this_slice] = hera_sim.sigchain.apply_gains(
            vis, gains, antpairpol[:2]
        )
        del vis
    return

def vary_gains_in_time(
    gains, 
    times, 
    mode='amp',
    variation_ref_times=None, 
    variation_timescales=None,
    variation_amps=(0.05,),
    variation_modes=('linear',),
    clip_gains=True
):
    """Vary gain amplitudes or phases in time.
    
    Parameters
    ----------
    gains : dict
        Dictionary containing the per-antenna gains to make vary in time.
    times : array-like of float
        Times at which to simulate time variation. Should match the unique 
        times in the visibility metadata.
    mode : str, optional
        Whether to vary the amplitudes or the phases. Accepts 'amp', 
        'amplitude', 'phs', or 'phase' as a value. Default is 'amp'.
    variation_ref_times : float or array-like of float, optional
        Reference time(s) to use for generating time variation. If not 
        using the 'noiselike' option, then this is the time where the 
        gains are unchanged from their original values. Default is to 
        use the middle time. This should have the same units as the 
        `times` parameter.
    variation_timescales : float or array-like of float, optional
        Timescale on which the variation is supposed to take place. For 
        sinusoidal variation, this is the period of the variation. This 
        should be in the same units as the `times` parameter. Default 
        is to use the entire length of `times`.
    variation_amps : float or array-like of float, optional
        Amplitude of the variation, as a fraction of the input gains. 
        This is *not* the peak-to-peak amplitude! Default is 5%.
    variation_modes : str or array-like of str, optional
        Which type(s) of time variation to simulate. Currently supported 
        options are 'linear', 'sinusoidal', and 'noiselike'. Default is 
        to vary gains linearly over the given times.
    clip_gains : bool, optional
        Whether to clip the gains so that they do not vary by more than 
        100% of their original values. Default is True. Raises a warning 
        if gains are clipped.

    Returns
    -------
    time_varied_gains : dict
        Dictionary of gains with time variation applied.
    """
    # Setup for handling multiple modes of variation at once.
    if variation_ref_times is None:
        variation_ref_times = (np.median(times),)
    if variation_timescales is None:
        variation_timescales = (times[-1] - times[0],)
    variation_modes = _listify(variation_modes)
    variation_amps = _listify(variation_amps)
    variation_ref_times = _listify(variation_ref_times)
    variation_timescales = _listify(variation_timescales)
    variation_params = (
        variation_modes, variation_amps, variation_ref_times, variation_timescales
    )
    Nmodes = max(len(param) for param in variation_params)
    if any(Nmodes % len(param) != 0 for param in variation_params):
        raise ValueError(
            "Input variation parameters are coprime in length. "
            "Please ensure variation parameters are either all "
            "the same length, or can be broadcast to the same "
            "length as the longest variation parameter."
        )
    iterator = zip(
        variation_ref_times * (Nmodes // len(variation_ref_times)),
        variation_timescales * (Nmodes // len(variation_timescales)),
        variation_modes * (Nmodes // len(variation_modes)),
        variation_amps * (Nmodes // len(variation_amps)),
    )

    # Generate an envelope for adding time variation.
    envelope = 1
    for ref_time, timescale, variation_mode, variation_amp in iterator:
        phases = (times - ref_time) / timescale
        if variation_mode == 'linear':
            envelope *= 1 + 2 * variation_amp * phases
        elif variation_mode == 'sinusoidal':
            envelope *= 1 + variation_amp * np.sin(2 * np.pi * phases)
        elif variation_mode == 'noiselike':
            envelope *= stats.norm.rvs(1, variation_amp, times.size)
    if np.all(envelope == 1):
        warn("Time variation method not supported; returning.")
        return gains
    if clip_gains and not np.allclose(envelope, np.clip(envelope, 0, 2)):
        warn(
            "Gains vary in time by an amount more than their time-"
            "independent values. Clipping time-variation envelope."
            "Check your parameters if this is unexpected."
        )
        envelope = np.clip(envelope, 0, 2)
    gain_shape = list(gains.values())[0].shape
    envelope = np.outer(envelope, np.ones(gain_shape[-1]))
    if mode in ('phs', 'phase'):
        envelope = np.exp(1j*envelope)

    # Actually apply the time variation, but do it in place.
    for ant, gain in gains.items():
        if len(gain_shape) == 1:
            new_gain = gain[None, :] * envelope
        else:
            new_gain = gain * envelope
        gains[ant] = new_gain
    return gains

def gen_xtalk(autovis, freqs, xamps, xdlys, xphs):
    """Generate a series of cross-coupling crosstalk visibilities."""
    xtalk = np.zeros_like(autovis, dtype=complex)
    _gen_xtalk = hera_sim.sigchain.gen_cross_coupling_xtalk
    for amp, dly, phs in zip(xamps, xdlys, xphs):
        xtalk += _gen_xtalk(freqs, autovis) #, amp, dly, phs)
        xtalk += _gen_xtalk(freqs, autovis) #, amp, -dly, phs)
    return xtalk

SYSTEMATICS_SIMULATORS = {
    'noise' : add_noise,
    'gains' : add_gains,
    'reflection_spectrum': add_reflection_spectrum,
    'reflections' : add_reflections,
    'xtalk' : add_xtalk
}

def apply_systematics(
    sim, 
    seed=None,
    noise=None, 
    gains=None, 
    reflection_spectrum=None,
    reflections=None,
    xtalk=None,
    return_systematics=False,
    verbose=False
):
    """One-stop shop for applying systematics to a simulation.

    This function handles the application of a handful of systematic 
    effects, provided appropriate parameters for simulating the 
    desired set of systematics. For more information on how the 
    systematics are simulated, please see the lower-level ``add_x`` 
    functions and their documentation.

    Parameters
    ----------
    sim : :class:`pyuvdata.UVData`
        Simulation object containing simulation data/metadata. May be 
        a subclass thereof, or something that may be converted to a 
        :class:`pyuvdata.UVData` object (such as a path to a file).
    seed : {int, None, or 'random'}, optional
        The random seed. If None, then this parameter is ignored. If 
        'random', then a seed is generated from the system time. See 
        :func:`_gen_seed` for details. Default is to not use a seed. 
        If this is specified, then it is used as the default seed for 
        any systematics that do not have their `seed` parameter defined. 
    noise : dict, optional
        Dictionary of parameters for simulating noise. See :func:`add_noise` 
        for further information. Default is to not simulate noise.
    gains : dict, optional
        Dictionary of parameters for simulating bandpass gains. See 
        :func:`add_gains` for further information. Default is to not 
        simulate bandpass gains.
    reflections : dict, optional
        Dictionary of parameters for simulating cable reflections. See 
        :func:`add_reflections` for further information. Default is to 
        not simulate cable reflections.
    xtalk : dict, optional
        Dictionary of parameters for simulating cross-coupling crosstalk. 
        See :func:`add_xtalk` for further information. Default is to not 
        simulate crosstalk.
    return_systematics : bool, optional
        Whether to return the simulated systematics. If True, then 
        systematics are returned as a dictionary, with the names of the 
        systematics as the keys. The values are either data arrays (in the 
        format of :class:`pyuvdata.UVData` data arrays) or dictionaries 
        mapping antennas to gains, depending on the systematic. (Noise 
        and crosstalk are visibility-like, so they are returned as data 
        arrays; reflections and bandpass gains are per-antenna quantities, 
        which are optionally time-dependent, and so are returned as mappings 
        from antenna numbers to gains.) Default is to not return systematics. 
        !! WARNING !! If you are simulating effects for large data arrays, 
        then enabling this feature may potentially result in a memory 
        overflow!
    verbose : bool, optional
        Whether to print statements tracking the progress of the systematics 
        simulation and application. Primarily used for debugging. Default 
        is to not print updates.

    Returns
    -------
    corrupted_sim : :class:`pyuvdata.UVData`
        Simulation with systematics applied.
    systematics : dict
        Dictionary mapping systematics to their data arrays or gain 
        dictionaries. None is returned if ``return_systematics`` is False.
    parameters : dict
        Dictionary mapping systematic names to simulation parameters 
        required to reconstruct the simulated systematic effects.
    """
    if verbose:
        print("Extracting systematics parameters...\n")

    # Collect all of the parameters into a dictionary.
    parameters = {
        'noise' : noise,
        'gains' : gains,
        'reflection_spectrum': reflection_spectrum,
        'reflections' : reflections,
        'xtalk' : xtalk
    }
  
    # Update random seeds in case user wants to "randomly" generate seeds.
    for params in parameters.values():
        if params is not None:
            params['seed'] = _gen_seed(params.get('seed', seed))

    # Apply the systematics and track the results.
    # Simulating this and keeping all the systematics may be very 
    # memory-intensive, so sometimes we might not want to keep 
    # track of the intermediate products.
    systematics = {} if return_systematics else None
    sim = _sim_to_uvd(sim)
    for systematic, params in parameters.items():
        if params is None:
            continue
        if verbose:
            print(f"Simulating {systematic}:")
            print(f"-----------"+"-"*len(systematic))
            print(f"Params:")
            for param, value in params.items():
                print(f"\t{param} : {value}")
            print("Min/Mean/Max before: ", np.min(sim.data_array),
                  np.mean(sim.data_array), np.max(sim.data_array))

        # This is a bit of a hack, but I can't think of a better way...
        t = time.time()
        add_systematic = SYSTEMATICS_SIMULATORS[systematic]

        if return_systematics:
            sim, systematics[systematic] = add_systematic(sim, ret_cmp=True, **params)
        else:
            sim = add_systematic(sim, ret_cmp=False, **params)

        if verbose:
            print("Min/Mean/Max after: ", np.min(sim.data_array),
                  np.mean(sim.data_array), np.max(sim.data_array))

            print(f"Done in {time.time() - t} sec.")
            print()
    return sim, systematics, parameters

# ------- Functions for preparing files ------- #

def adjust_sim_to_data(sim_file, data_files, verbose=False):
    """
    Modify simulated data to be consistent with an observation's metadata.
    
    Parameters
    ----------
    sim_file : str or path-like object
        Path to simulation file to modify.
    data_files : array-like of str or path-like objects
        Collection of paths to observed data.

    Returns
    -------
    modified_sim_data : :class:`pyuvdata.UVData`
        Simulation data that has been modified to match the metadata 
        contents of ``data_files``. See notes for details.
        
    Notes
    -----
    The modified simulation data has its antennas relabeled and repositioned to match 
    the labels and positions of the observational data. Any antenna from the 
    simulation that does not correspond to an antenna used in observation is excluded 
    from the modified simulation data. See ``downselect_antennas`` to learn more 
    about how the antenna-related metadata is modified. Note that this process 
    requires that the data be fully inflated; in the interest of keeping memory 
    consumption low, this step is performed after LST rephasing.
    
    It is also worth noting that the modified simulation data has not had its flag or 
    nsamples arrays modified in any way, but its data array has been modified. The 
    modified data has been downselected in time, rephased to have its LSTs match the 
    observed LSTs, and its time and LST arrays have been updated to match the observed 
    data.
    
    The modified simulation data is chunked into files containing the same number of 
    integrations as the observation files (assuming a uniform number of integration 
    per file) and written to disk. 
    """
    # Ensure the data files are a list and load in their metadata.
    data_files = _listify(data_files)
    ref_uvd = UVData()
    ref_uvd.read(data_files, read_data=False)

    # Load in the simulation data, but only the linear vispols.
    use_pols = [polstr2num(pol) for pol in ('xx', 'yy')]
    sim_uvd = UVData()
    sim_uvd.read(sim_file, polarizations=use_pols)

    # Downselect in time and rephase LSTs to match the reference data.
    if verbose:
        print("Rephasing simulation to reference data...")
    sim_uvd = interpolate_to_reference(sim_uvd, ref_uvd)

    # Inflate the simulation so antenna downselection can actually be done.
    if verbose:
        print("Inflating simulation by redundancy...")
    sim_uvd.inflate_by_redundancy()
    
    # Find and use the intersection of the RIMEz and H1C arrays.
    if verbose:
        print("Choosing subset of antennas to keep...")
    sim_uvd = downselect_antennas(sim_uvd, ref_uvd)
    
    # Make sure the data is conjugated properly so redcal doesn't break.
    if verbose:
        print("Conjugating simulation to ant1<ant2 convention...")
    sim_uvd.conjugate_bls('ant1<ant2')
    
    return sim_uvd

def prepare_sim_files(
    sim_file, 
    data_files, 
    save_dir, 
    sky_cmp=None,
    systematics_params=None, 
    save_truth=True,
    clobber=True,
    verbose=True
):
    """
    Modify simulation data to match reference metadata and apply systematics.
    
    Parameters
    ----------
    sim_file : str or path-like object
        Path to simulation file to modify.
    data_files : array-like of str or path-like objects
        Collection of paths to observed data.
    save_dir : str or path-like object
        Path to directory where modified simulation files should be saved.
    sky_cmp : str, optional
        Sky component simulated in ``sim_file``. If not provided, then the component 
        is inferred from the file name.
    systematics_params : dict, optional
        Dictionary mapping systematics names (i.e. 'noise', 'gains') to 
        a dictionary of parameters to use for generating the systematic.
        Default is to not simulate/apply systematics.
    save_truth : bool, optional
        Whether to save the true visibilities. Default is True.
    clobber : bool, optional
        Whether to overwrite files that may share the same name as the new files to 
        be saved in ``save_dir``. Default is to overwrite files.
    """
    if verbose:
        print("Beginning file preparation...")

    # Get the sky component if not specified.
    sky_cmp = sky_cmp or _parse_filename_for_cmp(sim_file)

    # Don't do anything if the files already exist and clobber is False.
    true_files_exist = _sim_files_exist(data_files, save_dir, sky_cmp, 'true')
    corrupt_files_exist = _sim_files_exist(data_files, save_dir, sky_cmp, 'corrupt')
    save_corrupt = bool(systematics_params)
    write_truth = save_truth and (clobber or not true_files_exist)
    write_corrupt = save_corrupt and (clobber or not corrupt_files_exist)
    if not (write_truth or write_corrupt):
        print("No new files to make; returning.")
        return

    # Modify the simulation data to match the reference metadata.
    sim_uvd = adjust_sim_to_data(sim_file, data_files, verbose=verbose)

    # Chunk the simulation data and write to disk.
    if write_truth:
        if verbose:
            print("Writing true visibilities to disk...")
        chunk_sim_and_save(
            sim_uvd, save_dir, ref_files=data_files, 
            sky_cmp=sky_cmp, state='true', clobber=clobber
        )

    # Apply systematics if desired.
    if write_corrupt:
        # TODO: update this to handle being able to save the systematics
        # but maybe raise a warning if the task may cause a MemoryError
        if verbose:
            print("Simulating and applying systematics.")
            print("====================================")
        sim_uvd, systematics, params = apply_systematics(
            sim_uvd, return_systematics=False, verbose=verbose, 
            **systematics_params
        )
        print('==================================')
        
        if verbose:
            print("Writing corrupted visibilities to disk...", end='')

        t = time.time()
        chunk_sim_and_save(
            sim_uvd, save_dir, ref_files=data_files, 
            sky_cmp=sky_cmp, state='corrupt', clobber=clobber
        )
        print(f" done in {time.time() - t} sec")
        
        if verbose:
            print("Writing sysytematics parameters to disk...")
        save_config(params, data_files[0], save_dir, sky_cmp, clobber)
    return

def downselect_antennas(sim_uvd, ref_uvd, tol=1.0):
    """
    Downselect simulated array to match unique baselines from reference.
    
    Parameters
    ----------
    sim_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the simulation data.
    ref_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the reference data. 
        Only needs to have the metadata loaded.
    tol : float, optional
        The maximum allowable discrepancy in antenna position components.
        Default is 1 meter.
        
    Returns
    -------
    new_sim_uvd : :class:`pyuvdata.UVData`
        Simulation data adjusted to contain a subset of the simulated array
        such that, after appropriately translating the antenna positions, 
        the maximum number of antennas/unique baselines remain in the 
        intersection of the simulation subarray and the reference array. 
        The relevant attributes of the simulation data are updated 
        accordingly.

    Notes
    -----
    See the memo "Antennas, Baselines, and Maximum Overlap" for details on 
    the antenna selection process. It is worth noting that the algorithm 
    implemented here is *not* appropriate for general arrays, but the 
    general case should reduce to this case for an array with appropriate 
    symmetry, as we have for the RIMEz simulation.
    """
    sim_uvd = _sim_to_uvd(sim_uvd)
    ref_uvd_meta = ref_uvd.copy(metadata_only=True)

    # Find the optimal intersection, get the map between antennas, and downselect the data.
    sim_antpos = _get_antpos(sim_uvd, ENU=False)
    ref_antpos = _get_antpos(ref_uvd_meta, ENU=False)
    array_intersection = _get_array_intersection(sim_antpos, ref_antpos, tol)
    sim_to_ref_ant_map = _get_antenna_map(array_intersection, ref_antpos, tol)
    ref_to_sim_ant_map = {
        ref_ant : sim_ant for sim_ant, ref_ant in sim_to_ref_ant_map.items()
    }
    sim_uvd.select(antenna_nums=list(sim_to_ref_ant_map.keys()), keep_all_metadata=False)
    ref_uvd_meta.select(
        antenna_nums=list(sim_to_ref_ant_map.values()), keep_all_metadata=False
    )
    
    # Update some of the antenna metadata in a copy of the simulation.
    # This is to ensure that slicing is performed correctly.
    sim_uvd_copy = copy.deepcopy(sim_uvd)
    sim_uvd_copy.ant_1_array = np.asarray(
        [sim_to_ref_ant_map[sim_ant] for sim_ant in sim_uvd.ant_1_array]
    )
    sim_uvd_copy.ant_2_array = np.asarray(
        [sim_to_ref_ant_map[sim_ant] for sim_ant in sim_uvd.ant_2_array]
    )
    attrs_to_update = (
        "antenna_numbers",
        "antenna_names",
        "telescope_location",
        "telescope_location_lat_lon_alt",
        "telescope_location_lat_lon_alt_degrees",
    )
    for attr in attrs_to_update:
        setattr(sim_uvd_copy, attr, getattr(ref_uvd_meta, attr))

    # Update the antenna positions array to use shifted simulation positions.
    # array_intersection has simulation antenna numbers as keys and shifted 
    # simulation positions as values.
    sim_uvd_copy.antenna_positions = np.array([
        array_intersection[ref_to_sim_ant_map[ref_ant]]
        for ref_ant in ref_uvd_meta.antenna_numbers
    ])
    sim_uvd_copy.history += "\nAntennas adjusted to optimally match H1C antennas."

    # Prepare the new data array.
    for antpairpol, vis in sim_uvd.antpairpol_iter():
        ai, aj, pol = antpairpol
        ref_antpairpol = (sim_to_ref_ant_map[ai], sim_to_ref_ant_map[aj], pol)
        ref_bl = ref_antpairpol[:2]
        blts, conj_blts, pol_inds = sim_uvd_copy._key2inds(ref_antpairpol)
        
        # Correctly choose which slice to use depending on whether the 
        # reference baseline corresponding to (ai, aj) is conjugated.
        # (If it's conjugated in the reference, then blts is empty.)
        if len(blts) > 0:
            this_slice = (blts, 0, slice(None), pol_inds[0])
        else:
            this_slice = (conj_blts, 0, slice(None), pol_inds[1])
            vis = vis.conj()
            ref_bl = ref_bl[::-1]

        # Update the data-like parameters.
        sim_uvd_copy.data_array[this_slice] = vis
        sim_uvd_copy.flag_array[this_slice] = sim_uvd.get_flags(antpairpol)
        sim_uvd_copy.nsample_array[this_slice] = sim_uvd.get_nsamples(antpairpol)
        # Update the baseline array.
        old_bl_int = sim_uvd.antnums_to_baseline(ai, aj)
        new_bl_int = sim_uvd.antnums_to_baseline(*ref_bl)
        sim_uvd_copy.baseline_array[sim_uvd.baseline_array==old_bl_int] = new_bl_int
        
    # Update the last of the metadata.
    sim_uvd_copy.set_uvws_from_antenna_positions()

    return sim_uvd_copy

def rephase_to_reference(sim_uvd, ref_uvd):
    """
    Rephase simulation LSTs to reference LSTs after downselection in time.
    
    After downselection and rephasing, this function overwrites the 
    simulation LST and time arrays to match the reference LSTs and times.
    
    Parameters
    ----------
    sim_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the simulation data.
    ref_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the reference data. 
        Only the metadata for this object needs to be read.

    Returns
    -------
    new_sim_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the modified 
        simulation data. The modified data has been rephased to match 
        the reference LSTs and has had its time and LST arrays adjusted 
        to contain only the times and LSTs that are present in the 
        reference data.
    """
    # Convert the simulation to a HERAData object. 
    sim_uvd = to_HERAData(sim_uvd)

    # Load in times and LSTs; get a mapping between reference/sim times.
    ref_time_to_lst_map = {
        ref_time : ref_lst 
        for ref_time, ref_lst in zip(ref_uvd.time_array, ref_uvd.lst_array)
    }
    sim_time_to_lst_map = {
        sim_time : sim_lst
        for sim_time, sim_lst in zip(sim_uvd.time_array, sim_uvd.lst_array)
    }
    ref_times = np.array(list(ref_time_to_lst_map.keys()))
    ref_lsts = np.array(list(ref_time_to_lst_map.values()))
    sim_times = np.array(list(sim_time_to_lst_map.keys()))
    sim_lsts = np.array(list(sim_time_to_lst_map.values()))
    ref_to_sim_time_map = get_d2m_time_map(ref_times, ref_lsts, sim_times, sim_lsts)

    # Clip reference times/lsts in case the reference bounds exceed the simulation's.
    if any(sim_time is None for sim_time in ref_to_sim_time_map.values()):
        warn("Some reference LSTs not in simulation; clipping reference times/LSTs.")
        key = [
            ref_times.tolist().index(ref_time)
            for ref_time, sim_time in ref_to_sim_time_map.items()
            if sim_time is not None
        ]
        ref_times = ref_times[key]
        ref_lsts = ref_lsts[key]

    # Figure out how much to rephase for each integration.
    selected_sim_lsts = np.array(
        list(
            sim_lsts[sim_times.tolist().index(sim_time)] 
            for sim_time in ref_to_sim_time_map.values()
            if sim_time is not None
        )
    )
    dlst = ref_lsts - selected_sim_lsts
    dlst = np.where(np.isclose(dlst, 0, atol=0.1), dlst, dlst - 2 * np.pi)

    # Warn the user if there are any large jumps in rephasing amounts, since 
    # this will likely result in a time getting skipped.
    dlst_diff = np.diff(dlst)
    avg_diff = np.median(dlst_diff)
    if any(not np.isclose(diff, avg_diff, rtol=0.1) for diff in dlst_diff):
        msg = "There is a large jump in the rephasing amount for "
        msg += "at least one time. There may be a discontinuity in "
        msg += "the rephased visibilities."
        warn(msg)

    # Downselect in time and load data.
    sim_uvd.select(times=list(ref_to_sim_time_map.values()))
    data, _, _ = sim_uvd.build_datacontainers()
    data.select_or_expand_times(list(ref_to_sim_time_map.values()))

    # Build the antpair -> ENU baseline vector dictionary.
    antpos = data.antpos
    bls = {
        (ai,aj,pol) : antpos[aj] - antpos[ai]
        for ai, aj, pol in data.bls()
    }

    # Rephase and update the data.
    lst_rephase(
        data, bls, data.freqs, dlst, 
        lat=sim_uvd.telescope_location_lat_lon_alt_degrees[0]
    ) # XXX do we want to use the simulation or the reference position?
      # lat/lon is good to about 6 mas; alt off by about 2 m
    sim_uvd.update(data=data)
    new_sim_lsts = np.zeros_like(sim_uvd.lst_array)
    new_sim_times = np.zeros_like(sim_uvd.time_array)
    loop_iterable = zip(ref_to_sim_time_map.values(), ref_times, ref_lsts)
    for times, ref_lst in zip(ref_to_sim_time_map.items(), ref_lsts):
        ref_time, sim_time = times
        blt_slice = np.argwhere(sim_uvd.time_array == sim_time).flatten()
        new_sim_lsts[blt_slice] = ref_lst
        new_sim_times[blt_slice] = ref_time
    sim_uvd.time_array = new_sim_times
    sim_uvd.lst_array = new_sim_lsts

    # Ensure that we return a UVData object so that chunk_sim_and_save doesn't break.
    return _sim_to_uvd(sim_uvd)

def interpolate_to_reference(sim_uvd, ref_uvd):
    """
    Interpolate simulation data to reference LSTs.
    
    After downselection and rephasing, this function overwrites the 
    simulation LST and time arrays to match the reference LSTs and times.
    
    Parameters
    ----------
    sim_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the simulation data.
    ref_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the reference data. 
        Only the metadata for this object needs to be read.

    Returns
    -------
    new_sim_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the modified 
        simulation data. The modified data has been rephased to match 
        the reference LSTs and has had its time and LST arrays adjusted 
        to contain only the times and LSTs that are present in the 
        reference data.
    """
    # Ensure the simulation is a UVData object.
    sim_uvd = _sim_to_uvd(sim_uvd)

    # Load in times and LSTs; get a mapping between reference/sim times.
    # This is to ensure that the times/LSTs are mapped appropriately 
    # when there is a phase wrap, since np.unique sorts values.
    ref_time_to_lst_map = {
        ref_time : ref_lst 
        for ref_time, ref_lst in zip(ref_uvd.time_array, ref_uvd.lst_array)
    }
    sim_time_to_lst_map = {
        sim_time : sim_lst
        for sim_time, sim_lst in zip(sim_uvd.time_array, sim_uvd.lst_array)
    }
    ref_times = np.array(list(ref_time_to_lst_map.keys()))
    ref_lsts = np.array(list(ref_time_to_lst_map.values()))
    ref_freqs = np.unique(ref_uvd.freq_array)
    sim_times = np.array(list(sim_time_to_lst_map.keys()))
    sim_lsts = np.array(list(sim_time_to_lst_map.values()))
    sim_freqs = np.unique(sim_uvd.freq_array)

    # Currently don't know how to deal with phase wraps appropriately, so
    def iswrapped(lsts):
        return any(lst < lsts[0] for lst in lsts)
    if iswrapped(ref_lsts) or iswrapped(sim_lsts):
        msg = "Interpolation to reference currently not supported for "
        msg += "wrapped LSTs."
        raise NotImplementedError(msg)

    # Raise a warning if any of the reference LSTs are outside the bounds 
    # of the simulation LSTs.
    if any(lst < sim_lsts[0] or lst > sim_lsts[-1] for lst in ref_lsts):
        warn("Reference LSTs exceed bounds of simulation. Clipping reference LSTs.")
        key = np.logical_and(sim_lsts[0] <= ref_lsts, ref_lsts <= sim_lsts[-1])
        ref_times = ref_times[key]
        ref_lsts = ref_lsts[key]

    # Setup data/metadata objects that will need to be rewritten nontrivially.
    new_Nblts = ref_times.size * sim_uvd.Nbls
    new_time_array = np.empty(new_Nblts, dtype=float)
    new_lst_array = np.empty(new_Nblts, dtype=float)
    new_ant_1_array = np.empty(new_Nblts, dtype=int)
    new_ant_2_array = np.empty(new_Nblts, dtype=int)
    new_baseline_array = np.empty(new_Nblts, dtype=int)
    new_uvw_array = np.empty((new_Nblts, 3), dtype=float)
    new_data = np.zeros(
        (new_Nblts, 1, ref_uvd.Nfreqs, sim_uvd.Npols), dtype=complex
    )

    # Fill in the new metadata, sorting the same way HERA data is sorted.
    for i, bl in enumerate(sim_uvd.get_antpairs()):
        ant1, ant2 = bl
        this_slice = slice(i, None, sim_uvd.Nbls)
        old_blt = sim_uvd._key2inds(bl)[0][0] # Just need one for a reference
        this_uvw = sim_uvd.uvw_array[old_blt]
        this_baseline = sim_uvd.baseline_array[old_blt]
        
        # Actually update the metadata.
        new_ant_1_array[this_slice] = ant1
        new_ant_2_array[this_slice] = ant2
        new_baseline_array[this_slice] = this_baseline
        new_uvw_array[this_slice] = this_uvw
        new_time_array[this_slice] = ref_times
        new_lst_array[this_slice] = ref_lsts

        # Now update the data.
        # XXX this isn't super careful and will break if the reference 
        # frequencies aren't within the bounds of the simulation frequencies
        for pol_ind, pol in enumerate(sim_uvd.polarization_array):
            vis = sim_uvd.get_data(bl + (pol,))
            re_spline = interp1d(sim_lsts, vis.real, axis=0, kind='cubic')
            im_spline = interp1d(sim_lsts, vis.imag, axis=0, kind='cubic')
            new_data[this_slice, 0, :, pol_ind] += (
                re_spline(ref_lsts) + 1j * im_spline(ref_lsts)
            )

    # Finally, update all of the data/metadata.
    sim_uvd.Ntimes = ref_times.size # In case ref_times had to be truncated.
    sim_uvd.Nfreqs = ref_uvd.Nfreqs
    sim_uvd.time_array = new_time_array
    sim_uvd.lst_array = new_lst_array
    sim_uvd.ant_1_array = new_ant_1_array
    sim_uvd.ant_2_array = new_ant_2_array
    sim_uvd.baseline_array = new_baseline_array
    sim_uvd.uvw_array = new_uvw_array
    sim_uvd.flag_array = np.zeros(new_data.shape, dtype=bool)
    sim_uvd.nsample_array = np.ones(new_data.shape, dtype=float)
    sim_uvd.data_array = new_data
    sim_uvd.blt_order = None
    return sim_uvd

def chunk_sim_and_save(
    sim_uvd, 
    save_dir, 
    ref_files=None,
    Nint_per_file=None,
    sky_cmp=None, 
    state=None, 
    clobber=True
):
    """
    Chunk the simulation data to match the reference file and write to disk.
    
    Chunked files have the following naming convention:
    save_dir/zen.{jd_major}.{jd_minor}[.{sky_cmp}][.{state}].uvh5
    The entires in brackets are optional and may be omitted.

    Parameters
    ----------
    sim_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the simulation data 
        to chunk and write to disk.
    save_dir : str or path-like object
        Path to the directory where the chunked files will be saved.
    ref_files : iterable of str
        Iterable of filepaths to use for reference when chunking. This must 
        be specified if ``Nint_per_file`` is not specified. This determines 
        (and overrides, if also provided) ``Nint_per_file`` if provided.
    Nint_per_file : int, optional
        Number of integrations per chunked file. This must be specified 
        if ``ref_files`` is not specified.
    sky_cmp : str, optional
        String denoting which sky component has been simulated. Should 
        be one of the following: ('foregrounds', 'eor', 'sum').
    state : str, optional
        String denoting whether the file is the true sky or corrupted.
    clobber : bool, optional
        Whether to overwrite any existing files that share the new 
        filenames. Default is to overwrite files.
    """
    if ref_files is None and Nint_per_file is None:
        raise ValueError(
            "Either a glob of reference files or the number of integrations "
            "per file must be provided."
        )

    # Pull the number of integrations per file if needed.
    if ref_files is not None:
        uvd = UVData()
        uvd.read(ref_files[0], read_data=False)
        Nint_per_file = uvd.Ntimes
        jd_pattern = re.compile(r"\.(?P<major>[0-9]{7})\.(?P<minor>[0-9]{5}).")

    # Pull the simulation times, then start the chunking process.
    sim_times = np.unique(sim_uvd.time_array)
    Nfiles = int(np.ceil(sim_uvd.Ntimes / Nint_per_file))
    for Nfile in range(Nfiles):
        # Figure out filing and slicing information.
        if ref_files is not None:
            jd = re.search(jd_pattern, ref_files[Nfile]).groupdict()
            jd = float(f"{jd['major']}.{jd['minor']}")
            uvd = UVData()
            uvd.read(ref_files[Nfile], read_data=False)
            times = np.unique(uvd.time_array)
        else:
            start_ind = Nfile * Nint_per_file
            jd = np.round(sim_times[start_ind], 5)
            this_slice = slice(start_ind, start_ind + Nint_per_file)
            times = sim_times[this_slice]
        filename = f"zen.{jd:.5f}.uvh5"
        if sky_cmp is not None:
            filename = filename.replace(".uvh5", f".{sky_cmp}.uvh5")
        if state is not None:
            filename = filename.replace(".uvh5", f".{state}.uvh5")
        save_path = os.path.join(save_dir, filename)

        # Chunk it and write to disk.
        this_uvd = sim_uvd.select(times=times, inplace=False)
        this_uvd.write_uvh5(save_path, clobber=clobber)

        # Delete the temporary UVData object to speed things up a bit.
        del this_uvd
    return

def save_config(config, ref_file, save_dir, sky_cmp, clobber=True):
    """
    Write the configuration used for systematics simulation to disk.
    
    Parameters
    ----------
    config : dict
        Dictionary mapping names of systematics components (e.g. 'noise' 
        or 'gains') to parameters used to simulate the systematic.
    ref_file : str
        Path to a reference file, to be used for extracting the JD.
    save_dir : str or path-like object
        Path to the directory where the chunked files will be saved.
    sky_cmp : str
        String denoting which sky component has been simulated. Should 
        be one of the following: ('foregrounds', 'eor', 'sum').
    clobber : bool, optional
        Whether to overwrite any existing files that share the new 
        filenames. Default is to overwrite files.
    """
    jd_pattern = re.compile("[0-9]{7}.[0-9]{5}")
    jd = jd_pattern.findall(ref_file)[0]
    filename = f"{jd}.config.{sky_cmp}.yaml"
    save_path = os.path.join(save_dir, filename)
    if not clobber and os.path.exists(save_path):
        print("Config file exists and clobber set to False; returning.")
        return
    with open(save_path, 'w') as f:
        f.write(yaml.dump(config))
    return

# ------- Helper Functions ------- #

def _sim_to_uvd(sim):
    """Update simulation object type."""
    if isinstance(sim, hera_sim.Simulator):
        sim = sim.data
    elif isinstance(sim, HERAData):
        uvd = UVData()
        for attr in uvd:
            setattr(uvd, attr, getattr(sim, attr))
        return uvd
    elif isinstance(sim, str):
        sim_ = UVData()
        sim_.read(sim)
        sim = sim_
    return sim

def _gen_seed(seed):
    if seed is None or isinstance(seed, int):
        return seed
    elif isinstance(seed, str) and seed == 'random':
        return time.time_ns() % 2**32
    elif isinstance(seed, (list, np.ndarray)) and all(isinstance(x, int) for x in seed):
        # Handle list or numpy array of integers
        return seed
    else:
        # raise ValueError(f"Invalid seed value: {seed}")
        return np.array(seed)
        # return 42

# def _gen_seed(seed):
#     """Generate a random seed pseudo-randomly if desired."""
#     print("Seed shape: ",np.shape(seed)," seed value: ",seed)
#     if seed is None or type(seed) is int:
#         print("first triggered")
#         return seed
#     elif len(seed)>0 and type(all(seed)) is int:
#         print("second triggered")
#         return seed
#     elif seed == 'random':
#         print("Third triggered")
#         return time.time_ns() % 2**32

def _get_array_intersection(sim_antpos, ref_antpos, tol=1.0):
    """Find the optimal choice of simulation subarray and return it."""
    optimal_translation = _get_optimal_translation(sim_antpos, ref_antpos, tol)
    new_antpos = {ant : pos + optimal_translation for ant, pos in sim_antpos.items()}
    intersection = {
        ant : pos for ant, pos in new_antpos.items()
        if any(np.allclose(pos, ref_pos, atol=tol) for ref_pos in ref_antpos.values())
    }
    return intersection

def _get_antenna_map(sim_antpos, ref_antpos, tol=1.0):
    """Find a mapping from simulation antennas to reference antennas."""
    antenna_map = {}
    refants = list(ref_antpos.keys())
    for ant, pos in sim_antpos.items():
        refant_index = np.argwhere(
            [np.allclose(pos, ref_pos, atol=tol) for ref_pos in ref_antpos.values()]
        ).flatten()
        if refant_index.size == 0:
            continue
        antenna_map[ant] = refants[refant_index[0]]
    return antenna_map

def _get_antpos(uvd, ENU=True):
    """Retrieve the {ant : pos} dictionary from the data."""
    if ENU:
        pos, ant = uvd.get_ENU_antpos()
    else:
        ant = uvd.antenna_numbers
        pos = uvd.antenna_positions
    return dict(zip(ant, pos))

def _get_optimal_translation(sim_antpos, ref_antpos, tol=1.0):
    """Find the translation that maximizes the overlap between antenna arrays."""
    # Get a dictionary of translations; keys are sim_ant -> ref_ant.
    translations = _build_translations(sim_antpos, ref_antpos, tol)
    intersection_sizes = {}
    
    # Calculate the number of antennas in the intersection for each pair of arrays.
    for sim_ant_to_ref_ant, translation in translations.items():
        new_sim_antpos = {
            ant : pos + translation for ant, pos in sim_antpos.items()
        }
        Nintersections = sum(
            any(np.allclose(pos, ref_pos, atol=tol) 
                for ref_pos in ref_antpos.values()
            )
            for pos in new_sim_antpos.values()
        )
        intersection_sizes[sim_ant_to_ref_ant] = Nintersections
        
    # Choose the translation that has the most antennas in the intersection.
    sim_to_ref_keys = list(translations.keys())
    intersections_per_translation = np.array(list(intersection_sizes.values()))
    index = np.argmax(intersections_per_translation)
    return translations[sim_to_ref_keys[index]]

def _build_translations(sim_antpos, ref_antpos, tol=1.0):
    """Build all possible translations that overlap at least one antenna."""
    sim_ants = list(sim_antpos.keys())
    ref_ants = list(ref_antpos.keys())
    translations = {
        f"{sim_ant}->{ref_ant}" : ref_antpos[ref_ant] - sim_antpos[sim_ant]
        for sim_ant in sim_ants for ref_ant in ref_ants
    }
    unique_translations = {}
    for key, translation in translations.items():
        if not any(
            np.allclose(translation, unique_translation, atol=tol)
            for unique_translation in unique_translations.values()
        ):
            unique_translations[key] = translation
    return unique_translations

def _parse_filename_for_cmp(filename):
    """Infer the sky component from the provided filename."""
    cmp_re = re.compile("[a-z]+.uvh5")
    sky_cmp = cmp_re.findall(str(filename))
    if sky_cmp == []:
        raise ValueError(
            f"Simulation component could not be inferred from {filename}."
        )
    return sky_cmp[0][:-5]

def _sim_files_exist(data_files, save_dir, sky_cmp, state):
    """Check if the chunked simulation files already exist."""
    file_ext = os.path.splitext(data_files[0])[1]
    data_file_names = [os.path.split(dfile)[1] for dfile in data_files]
    sim_file_names = [data_file.replace("HH", f"{sky_cmp}") for data_file in data_file_names]
    # Catch cases where the data files don't have 'HH' in their name.
    if all(
        sim_file == data_file for sim_file, data_file in zip(sim_file_names, data_file_names)
    ):
        sim_file_names = [
            sim_file.replace(file_ext, f".{sky_cmp}" + file_ext)
            for sim_file in sim_file_names
        ]
    sim_files = [
        os.path.join(save_dir, sim_file.replace(file_ext, f".{state}" + file_ext))
        for sim_file in sim_file_names
    ]
    return all(os.path.exists(sim_file) for sim_file in sim_files)

def _apply_lst_cut(file_glob, lst_min, lst_max, uniformly_sampled=True):
    """
    Choose a subset of files so that only the specified range of LSTs remain.
    
    Parameters
    ----------
    file_glob : array-like of str
        Glob of file paths to trim. The files in the glob should be for a 
        single day of observation.
    lst_min : float
        Lower bound of LST range to keep, in units of hours.
    lst_max : float
        Upper bound of LST range to keep, in units of hours.
    uniformly_sampled : bool, optional
        Whether the files are uniformly sampled with the same number of 
        integrations per file. Default assumption is that they are uniform.
        If this is not true, then the file downselection may be very slow, 
        since the entire file glob is read.

    Returns
    -------
    trimmed_file_glob : list of str
        File glob trimmed to contain only files whose LSTs lie completely 
        within the range of LSTs desired.
    """
    # Ensure the files are sorted.
    data_files = sorted(file_glob)
    Nfiles = len(data_files)

    # Extract parameters necessary for determining the duration of a single file.
    uvd = UVData()
    uvd.read(data_files[0], read_data=False)
    dt = np.median(np.diff(np.unique(uvd.time_array)))
    Nint_per_file = uvd.Ntimes
    
    # Determine which times correspond to LSTs in the desired LST range.
    if uniformly_sampled:
        uvd = UVData()
        uvd.read(data_files[0], read_data=False)
        lst0 = uvd.lst_array[0] * units.day.to('hr') / (2 * np.pi)
        unique_lsts = np.unique(uvd.lst_array)
        unique_lsts[unique_lsts < unique_lsts[0]] += 2 * np.pi # unwrap LSTs
        dlst = np.median(np.diff(unique_lsts)) * units.day.to('hr') / (2 * np.pi)
        file_lst_span = dlst * uvd.Ntimes
        start_lsts = (lst0 + file_lst_span * np.arange(Nfiles)) % 24
        data_files = sorted([
            dfile for dfile, start_lst in zip(data_files, start_lsts)
            if lst_min <= start_lst and start_lst + file_lst_span <= lst_max
        ])
    else:
        # Need to do it slowly; check files individually in case unevenly spaced.
        files_to_keep = []
        for dfile in data_files:
            uvd = UVData()
            uvd.read(dfile, read_data=False)
            lsts_hr = np.unique(uvd.lst_array) * units.day.to('hr') / (2 * np.pi)
            keep_file = np.all(
                np.logical_and(lst_min <= lsts_hr, lsts_hr <= lst_max)
            )
            if keep_file:
                files_to_keep.append(dfile)
            del uvd
        data_files = sorted(files_to_keep)
    return data_files

# ------- Argparsers ------- #

def sim_prep_argparser():
    """Argparser for preparing simulation files and adding systematics."""
    desc = "Modify simulation files to match observation parameters "
    desc += "and optionally apply systematics."
    a = argparse.ArgumentParser(description=desc)
    a.add_argument("--verbose", default=False, action="store_true")
    file_opts = a.add_argument_group(title="File preparation parameters.")
    file_opts.add_argument("simfile", type=str, help="Simulation file to be modified.")
    file_opts.add_argument("obsdir", type=str, help="Directory containing observation files.")
    file_opts.add_argument("savedir", type=str, help="Destination to write modified files.")
    file_opts.add_argument(
        "--lst_min", type=float, default=0, help="Minimum LST to keep, in hours."
    )
    file_opts.add_argument(
        "--lst_max", type=float, default=24, help="Maximum LST to keep, in hours."
    )
    file_opts.add_argument(
        "--uniformly_sampled", default=False, action="store_true",
        help="Whether the reference files in ``obsdir`` uniformly sample in LST."
    )
    file_opts.add_argument(
        "--skip_truth", default=False, action="store_true",
        help="Skip writing the true visibilities to disk; only write corrupted."
    )
    file_opts.add_argument(
        "--clobber", default=False, action="store_true", 
        help="Overwrite existing modified simulation files."
    )
    file_opts.add_argument(
        "--Nchunks", default=1, type=int, 
        help="Number of chunks to break obsfiles glob into for file preparation."
    )
    systematics = a.add_argument_group(title="Options for simulating systematics.")
    systematics.add_argument("--seed", default=None, help="Random seed for all components.")
    systematics.add_argument(
        "--config", default=None, type=str, help="Configuration file for systematics."
    )
    args = a.parse_args()
    return args

def abscal_model_argparser():
    """Argparser for preparing the abscal model."""
    desc = "Prepare an absolute calibration model from a simulation file."
    a = argparse.ArgumentParser(description=desc)
    a.add_argument("simfile", type=str, help="Simulation file to use.")
    a.add_argument("reffile", type=str, help="Reference for array selection.")
    a.add_argument("savedir", type=str, help="Where to write files to disk.")
    a.add_argument("Nint_per_file", type=int, help="Number of integrations per file.")
    a.add_argument("lst_min", type=float, help="Lower bound of LST range, in hours.")
    a.add_argument("lst_max", type=float, help="Upper bound of LST range, in hours.")
    a.add_argument("--Trx", type=float, default=0, help="Receiver temperature for noise.")
    a.add_argument(
        "--Nchunks", type=int, default=1, help="Number of chunks for performing routine."
    )
    a.add_argument(
        "--add_noise", default=False, action="store_true", help="Whether to add noise."
    )
    a.add_argument("--clobber", default=False, action="store_true", help="Overwrite files.")
    a.add_argument("--verbose", default=False, action="store_true", help="Print progress.")
    args = a.parse_args()
    return args


def smooth_abscal_model_argparser():
    """Argparser for smoothing the abscal model."""
    desc = "Smooth abscal model."
    a = argparse.ArgumentParser(description=desc)
    a.add_argument("datadir", type=str, help="Data dir location and savedir")
    a.add_argument("datafile", type=str, help="Glob-parseable field for data files")
    a.add_argument("tol", type=float, help="clean tol")
    a.add_argument("gain", type=float, help="clean gain")
    a.add_argument("skip_wgt", type=float, help="clean skip_wgt")
    a.add_argument("edgecut_low", type=int, help="clean edgecut_low")
    a.add_argument("edgecut_hi", type=int, help="clean edgecut_hi")
    a.add_argument("maxiter", type=int, help="clean maxiter")
    a.add_argument("min_dly", type=float, help="clean min_dly")
    a.add_argument("horizon", type=float, help="clean horizon")
    a.add_argument("standoff", type=float, help="clean standoff")
    a.add_argument("window", type=str, help="clean window")
    a.add_argument("alpha", type=float, help="clean alpha")
    args = a.parse_args()
    return args
                     
