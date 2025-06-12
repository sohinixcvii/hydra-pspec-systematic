import matplotlib.pyplot as plt
import numpy as np
from .config_plots import *
import os
import scipy.stats as sci_st

def master_plotter(data_sets, col_labels=['Data A','Data B','Data C'],fig_title='Data comparison', plot_type='imshow', norm='linear',save_flag=True,cmap='seismic',dir=op_dir,imag_flag=True):
    '''
    A function to plot real, imaginary, and absolute realisations of any n data sets. 
    
    Parameters:
        data_sets:
            A set of n 2d data sets, passed as [set a, set b, set c....]. (n, ydim,xdim)
        col_labels:
            A set of n data labels for plotting
        fig_title:
            A title for the whole figure being saved. Will also be the filename
        plot_type:
            Type of plot you want, choose between imshow and matshow. Default is imshow
        norm:
            normalisation form for plotting (log/linear etc)
        save_flag:
            Whether you want to save the files
        cmap:
            colormap to use in these plots. Pass a valid matplotlib cmap or an imported cmasher cmap etc. 
        dir: 
            Output directory for the figures
        imag_flag:
            Boolean flag for turning .imag plots on/off (particularly useful for realified matrices)
        
    '''
    num_sets=len(data_sets)
    data_sets=np.array(data_sets)
    if len(col_labels)!=num_sets:
        print("Error! Incorrect number of labels")
        return 0
    
    if imag_flag==True:
        fig, ax = plt.subplots(3,num_sets, figsize=(num_sets*7,num_sets*7+1))
        ylabels = ['Real','Imaginary','Absolute']
        if num_sets!=1:
            for j in range(len(ax[:,0])):
                ax[j,0].set_ylabel(ylabels[j], fontsize=20)
        else:
            for j in range(len(ax)):
                ax[j].set_ylabel(ylabels[j])
    else:
        fig, ax = plt.subplots(1,num_sets, figsize=(num_sets*7+1,7))
        ylabels = ['Real']
        if num_sets==1:
            ax.set_ylabel(ylabels[0])
        else:
            ax[0].set_ylabel(ylabels[0])

    if num_sets!=1:
        plt.rcParams.update({'font.size': 20})
    
    if plot_type=='imshow':
        for i in range(num_sets):  
            if num_sets==1:
                if imag_flag==True:
                    im=ax[0].imshow(data_sets[i,:,:].real,origin='lower',norm=norm,cmap=cmap)
                    ax[0].set_title(col_labels[i])
                    plt.colorbar(im)

                    im=ax[1].imshow(data_sets[i,:,:].imag,origin='lower',norm=norm,cmap=cmap)
                    plt.colorbar(im)

                    im=ax[2].imshow(np.absolute(data_sets[i,:,:]),origin='lower',norm=norm,cmap=cmap)
                    plt.colorbar(im)
                else:
                    im = ax.imshow(data_sets[i,:,:].real,origin='lower',norm=norm,cmap=cmap)
                    ax.set_title(col_labels[i])
                    plt.colorbar(im)

            else:
                if imag_flag==True:
                    im=ax[0,i].imshow(data_sets[i,:,:].real,origin='lower',norm=norm,cmap=cmap)
                    ax[0,i].set_title(col_labels[i])
                    plt.colorbar(im)
                    
                    im=ax[1,i].imshow(data_sets[i,:,:].imag,origin='lower',norm=norm,cmap=cmap)
                    plt.colorbar(im)

                    im=ax[2,i].imshow(np.absolute(data_sets[i,:,:]),origin='lower',norm=norm,cmap=cmap)
                    plt.colorbar(im)
                else:
                    im=ax[i].imshow(data_sets[i,:,:].real,origin='lower',norm=norm,cmap=cmap)
                    ax[i].set_title(col_labels[i])
                    plt.colorbar(im)
    elif plot_type=='matshow':
        for i in range(num_sets):
            if num_sets==1:
                if imag_flag==True:
                    im=ax[0].matshow(data_sets[i,:,:].real,origin='lower',aspect='auto',cmap=cmap)
                    ax[0].set_title(col_labels[i])
                    plt.colorbar(im)
                    
                    im=ax[1].matshow(data_sets[i,:,:].imag,origin='lower',aspect='auto',cmap=cmap)
                    plt.colorbar(im)

                    im=ax[2].matshow(np.absolute(data_sets[i,:,:]),origin='lower',aspect='auto',cmap=cmap)
                    plt.colorbar(im)
                else:
                    im=ax.matshow(data_sets[i,:,:].real,origin='lower',aspect='auto',cmap=cmap)
                    ax.set_title(col_labels[i])
                    plt.colorbar(im)
            else:
                if imag_flag==True:
                    im=ax[0,i].matshow(data_sets[i,:,:].real,origin='lower',aspect='auto',cmap=cmap)
                    ax[0,i].set_title(col_labels[i])
                    plt.colorbar(im)
                    
                    im=ax[1,i].matshow(data_sets[i,:,:].imag,origin='lower',aspect='auto',cmap=cmap)
                    plt.colorbar(im)

                    im=ax[2,i].matshow(np.absolute(data_sets[i,:,:]),origin='lower',aspect='auto',cmap=cmap)
                    plt.colorbar(im)
                else:
                    im=ax[i].matshow(data_sets[i,:,:].real,origin='lower',aspect='auto',cmap=cmap)
                    ax[i].set_title(col_labels[i])
                    plt.colorbar(im)
    
    fig.suptitle(fig_title,ha='center',va='bottom')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    if save_flag==True:
        if os.path.isdir(dir) == False:
            os.makedirs(dir)
        plt.savefig(dir+fig_title+'.png',bbox_inches='tight',dpi=300)
    
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
