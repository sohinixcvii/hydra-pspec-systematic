import matplotlib.pyplot as plt
import numpy as np
from .config_plots import *
import os
import scipy.stats as sci_st
from astropy import units

def master_plotter(
    data_sets,
    col_labels=None,
    fig_title='Data comparison',
    plot_type='imshow',
    norm='linear',
    save_flag=True,
    cmap='seismic',
    dir=op_dir,
    imag_flag=True,
    vmin=None,
    vmax=None,
    show=False
):
    """
    Plot a list of 2D complex data sets in a grid of subplots, displaying real, imaginary,
    and absolute components (if `imag_flag` is True).

    Parameters
    ----------
    data_sets : list of 2D np.ndarray
        List of 2D complex-valued arrays to plot. Each entry will be shown in a column of subplots.

    col_labels : list of str, optional
        Labels for each data set to be shown above the corresponding column. If not provided, generic labels will be used.

    fig_title : str, optional
        Title for the entire figure. Also used as the filename if saving.

    plot_type : {'imshow', 'matshow'}, optional
        Type of matplotlib plot to use for each subplot. Default is 'imshow'.

    norm : {'linear', 'log'} or matplotlib.colors.Normalize, optional
        Color normalization to apply. Can be 'linear', 'log', or a custom Normalize object.

    save_flag : bool, optional
        Whether to save the resulting figure to file.

    cmap : str or matplotlib colormap, optional
        Colormap to use for the plots.

    dir : str, optional
        Directory to save the figure in, if `save_flag` is True. Default is `op_dir` specified in config_plots.

    imag_flag : bool, optional
        If True, plots all three of real, imaginary, and absolute parts. If False, only plots the real part.

    vmin : float or list of floats, optional
        Minimum value(s) for colormap normalization. Can be a scalar or list matching the number of data sets.

    vmax : float or list of floats, optional
        Maximum value(s) for colormap normalization. Can be a scalar or list matching the number of data sets.

    show : bool, optional
        If True, displays the figure using `plt.show()`. If False, closes the figure after saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plotted subplots.

    Raises
    ------
    ValueError
        If the number of column labels does not match the number of data sets, or if an unsupported `plot_type` or `norm` is provided.

    Notes
    -----
    - This function is useful for visually comparing multiple 2D complex-valued matrices in terms of their real, imaginary, and magnitude components.
    """

    num_sets = len(data_sets)
    # data_sets = np.array(data_sets)
    col_labels = col_labels or [f"Data {i}" for i in range(num_sets)]

    if len(col_labels) != num_sets:
        raise ValueError("Number of column labels must match number of data sets.")

    if isinstance(norm, str):
        if norm == 'linear':
            norm_fn = None
        elif norm == 'log':
            norm_fn = LogNorm()
        else:
            raise ValueError(f"Unknown norm '{norm}'")
    else:
        norm_fn = norm

    nrows = 3 if imag_flag else 1
    fig, ax = plt.subplots(nrows, num_sets, figsize=(num_sets * 5, nrows * 4), squeeze=False)
    ylabels = ['Real', 'Imaginary', 'Absolute']

    for i in range(num_sets):
        data = data_sets[i]
        vmin_i = vmin[i] if isinstance(vmin, (list, tuple, np.ndarray)) else vmin
        vmax_i = vmax[i] if isinstance(vmax, (list, tuple, np.ndarray)) else vmax
        for j, part in enumerate([np.real(data), np.imag(data), np.abs(data)] if imag_flag else [np.real(data)]):
            plot_ax = ax[j, i]
            if plot_type == 'imshow':
                im = plot_ax.imshow(part, origin='lower', cmap=cmap, norm=norm_fn, vmin=vmin_i, vmax=vmax_i)
            elif plot_type == 'matshow':
                im = plot_ax.matshow(part, cmap=cmap, norm=norm_fn, vmin=vmin_i, vmax=vmax_i, aspect='auto')
            else:
                raise ValueError("plot_type must be 'imshow' or 'matshow'")
            if i == 0:
                plot_ax.set_ylabel(ylabels[j], fontsize=14)
            plot_ax.set_title(col_labels[i], fontsize=14)
            plt.colorbar(im, ax=plot_ax, fraction=0.046, pad=0.04)

    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_flag:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(dir+fig_title+'.png',bbox_inches='tight',dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plot_dps(vis_eor_path, res_dir, dir=op_dir, Nburn=0, conf_interval=95, ):
    # Load in EoR visibilities
    uvd = UVData()
    uvd.read(vis_eor_path)
    uvd.conjugate_bls()
    uvd = form_pseudo_stokes_vis(uvd)
    # The test data only contains a single baseline (0, 1) and the pseudo-Stokes I
    # visibilities after `form_pseudo_stokes_vis` are stored in the XX polarization
    vis_eor = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)

    # Get freuqency metadata
    freqs = uvd.freq_array * units.Hz
    if uvd.use_future_array_shapes:
        freqs = freqs[0]
    df = freqs[1] - freqs[0]
    Nfreqs = freqs.size

    # Compute the delay power spectrum of the input EoR signal
    axes = (1,)
    ds_eor_true = np.fft.ifftshift(vis_eor, axes=axes)
    ds_eor_true = np.fft.fftn(ds_eor_true, axes=axes)
    ds_eor_true = np.fft.fftshift(ds_eor_true, axes=axes)
    dps_eor_true = (np.abs(ds_eor_true)**2).mean(axis=0)
    delays = np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=df.to("1/ns")))

    # Load in results from hydra_pspec
    dps_eor_hp = np.load(Path(res_dir) / "dps-eor.npy")
    ln_post = np.load(Path(res_dir) / "ln-post.npy")
    if Nburn > 0:
        dps_eor_hp = dps_eor_hp[Nburn:]
        ln_post = ln_post[Nburn:]
    # Posterior-weighted mean delay power spectrum
    dps_eor_hp_pwm = np.average(dps_eor_hp, weights=ln_post, axis=0)
    # Confidence interval of delay power spectrum posteriors
    percentile = conf_interval/2 + 50
    dps_eor_hp_ubound = np.percentile(dps_eor_hp, percentile, axis=0)
    dps_eor_hp_lbound = np.percentile(dps_eor_hp, 100-percentile, axis=0)
    dps_eor_hp_err = np.vstack((
        dps_eor_hp_pwm - dps_eor_hp_lbound,
        dps_eor_hp_ubound - dps_eor_hp_pwm
    ))

    # Plot the true and recovered delay power spectra
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(delays, dps_eor_true, "k:", label="True")
    ax.errorbar(
        delays,
        dps_eor_hp_pwm,
        yerr=np.abs(dps_eor_hp_err),
        color="k",
        # ls="",
        marker="o",
        capsize=3,
        label=f"Recovered ({conf_interval}% Confidence)"
    )
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"$P(\tau)$ [arb. units]")
    ax.set_title("EoR Delay Power Spectrum Comparison (systematics)")
    ax.set_yscale("log")
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir+'EoR_DPS_comparison.png',bbox_inches='tight',dpi=300)

    res=dps_eor_hp_pwm - dps_eor_true
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.errorbar(delays,res, yerr=0.68*np.abs(dps_eor_hp_err),marker="o",
        capsize=3,)
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"Data - true dps")
    ax.set_title("Residuals vs delays")
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir+'EoR_DPS_res_vs_delays.png',bbox_inches='tight',dpi=300)


    z_sc=sci_st.zscore(res)
    sig=np.std(dps_eor_hp_err)
    yerr_z=dps_eor_hp_err/sig

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.errorbar(delays,z_sc, yerr=np.abs(yerr_z),marker="o",markerfacecolor='blue',
        capsize=3,ecolor='blue')
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"Z score")
    ax.set_title("Z score vs delays")
    ax.set_ylim(-5,5)
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir+'EoR_DPS__Score_vs_delays.png',bbox_inches='tight',dpi=300)

    fig, ax = plt.subplots(figsize=(24, 5))
    ax.plot(delays,dps_eor_hp_err[0,:],marker="o",label='Lower limit',ls='dotted')
    ax.plot(delays,dps_eor_hp_err[1,:],marker="o",label='Upper limit',ls='dotted')
    ax.plot(delays,np.mean(dps_eor_hp_err, axis=0),marker="o",label='Mean',c='k')
    ax.set_title("Error bar means and upper-lower limits")
    ax.grid()
    ax.legend()
    plt.savefig(dir+'EoR_DPS_Error_bar_mins_limits.png',bbox_inches='tight',dpi=300)
