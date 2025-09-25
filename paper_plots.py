from astropy import constants, units
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import scipy 
from scipy import signal
from scipy.fft import fft, fftshift
from astropy.units import Quantity
from astropy import units
import matplotlib.ticker as ticker
import cmcrameri.cm as cmc
import sys

import hera_sim
from pyuvdata import UVData
from uvtools.dspec import gen_window
from uvtools.plot import waterfall
from uvtools.utils import FFT, fourier_freqs

from astropy import units as u
from functions import covariance_from_pspec,sys_modes,fourier_operator

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 20})

# Check power spectrum
def calc_ps(s):
    # NOTE: This uses inverse FFT instead of FFT to get the right normalisation
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs # CHECK: This takes an average

'''--------------------Setup-----------------------------------------------'''
result_dir='/nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/'
# run_version = 'low_dl_fr_0'
run_version = 'high_dl_fr_0'
# run_version = 'low_dl_low_fr'
# run_version = 'fixed_sky'

# Build systematics model
# nm_list = [(3,0),(4,0),(5,0),(6,0)] #low dl fr 0
nm_list = [(10,0), (11,0), (12,0), (13,0)] #high dl fr 0
# nm_list = [(3,3),(3,4),(3,5),(3,6)] #low dl low fr

Ntimes = 80 #60 #203
Nfreqs = 60
freqs = np.linspace(100., 120., 120) ##120) 
Nfgmodes = 12
freqs = freqs[:Nfreqs]
# Generate FG mode matrix
fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(Nfgmodes)
            ]).T

lsts = np.linspace(0., 1., Ntimes)
sys_modes_operator = sys_modes(freqs_Hz=freqs*1e6, 
                                    times_sec=lsts * 24./(2.*np.pi) * 3600., 
                                    modes=nm_list)

sys_amps_true = np.array([4., 4.1, 5., -2.]) #np.array([4., 4.01])
sys_prior = 4**2. * np.eye(sys_amps_true.size)

fourier_op = fourier_operator(freqs.size, unitary=True)

# Generate noise
noise_ps_val = 0.000004 #0.000004 # 0.0004
noise_ps_true = noise_ps_val * np.ones(freqs.size)
N_true = covariance_from_pspec(noise_ps_true, fourier_op)
Ninv = np.diag(1./np.diag(N_true)) # get diagonal, invert, pack back into diagonal
n = np.sqrt(N_true) @ (np.random.randn(freqs.size, Ntimes) 
                    + 1.j*np.random.randn(freqs.size, Ntimes)) / np.sqrt(2.)
# Note factor of sqrt(2) above
noise_ps_check = calc_ps(n.T)
'''-----------------------------------------------------------------------------------------------'''
'''------------------------DPS plots from test cases-----------------------------------------------'''
result_dir='/nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/'
run_version_arr = ['low_dl_fr_0','high_dl_fr_0','low_dl_low_fr']
conf_interval=95
Nburn = 10
bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)

dl_inds=[[3,4,5,6],[10,11,12,13],[3,3,3,3]]
fig, ax = plt.subplots(4,1,figsize=(25, 20))
i=0
colors=['r','b','k']
fig_labels=['I','II','III','Residuals']
for run_version in run_version_arr:
    eor_true=np.load(result_dir+'high_dl_fr_0/eor_true.npy')
    ps_sample = np.load(result_dir+run_version+'/dps-eor.npy')
    ln_post = np.load(result_dir+run_version+'/ln-post.npy')
    ps_true = calc_ps(eor_true[:Ntimes,:Nfreqs])
    ps_mean = np.mean(ps_sample, axis=0)
    df = (freqs[1] - freqs[0]) * u.MHz
    delays = np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=df.to("1/ns")))

    print(dl_inds[i])
    sys_delays= delays[np.unique(dl_inds[i])+int(Nfreqs/2)].value
    if Nburn > 0:
        ps_sample = ps_sample[Nburn:]
        ln_post = ln_post[Nburn:]
    # Posterior-weighted mean delay power spectrum
    dps_eor_hp_pwm = np.average(ps_sample, weights=ln_post, axis=0)
    
    # Confidence interval of delay power spectrum posteriors
    percentile = conf_interval/2 + 50
    dps_eor_hp_ubound = np.percentile(ps_sample, percentile, axis=0)
    dps_eor_hp_lbound = np.percentile(ps_sample, 100-percentile, axis=0)
    dps_eor_hp_err = np.vstack((
        dps_eor_hp_pwm - dps_eor_hp_lbound,
        dps_eor_hp_ubound - dps_eor_hp_pwm
    ))
    
    ax[i].errorbar(
                    delays,
                    dps_eor_hp_pwm,
                    yerr=np.abs(dps_eor_hp_err),
                    color=colors[i],
                    # ls="",
                    marker="o",
                    capsize=3,
                    label=f"Recovered ({conf_interval}% Confidence)"
                    )
    ax[i].plot(delays, ps_true, "k:", label="True")

    ax[i].legend(loc="best",fontsize=20)
    ax[i].set_ylabel(r"$P(\tau)$ [arb. units]")
    # ax.set_title("EoR Delay Power Spectrum Comparison")
    ax[i].set_yscale("log")
    # ax[i].grid()
    
    x0, x1 = np.min(sys_delays), np.max(sys_delays)   # the x-range to shade
    ax[i].axvspan(x0, x1, color=colors[i], alpha=0.1, ec=None, zorder=0.1)
    ax[3].axvspan(x0, x1, color=colors[i], alpha=0.1, ec=None, zorder=0.1)

    if i==2:
        for dl in sys_delays:
            ax[i].axvline(dl,ls='dotted',c=colors[i])
            ax[3].axvline(dl,ls='dotted',c=colors[i])
    # for dl in sys_delays:
    #     ax[i].axvline(dl,ls='dotted',c=colors[i])
    #     ax[-1].axvline(dl,ls='dotted',c=colors[i])
    ax[i].text(0.95,0.07,fig_labels[i],fontsize=15, bbox=bbox,
            transform=ax[i].transAxes, horizontalalignment='right')

    ax[-1].plot(delays,(dps_eor_hp_pwm-ps_true),color=colors[i],label=fig_labels[i])
    ax[-1].set_ylabel(r"$P(\tau)$ [arb. units]")
    ax[-1].set_xlabel(r'$\tau$ [ns]')
    ax[-1].legend()
    # ax[-1].grid()

    i=i+1
ax[-1].text(0.95,0.07,fig_labels[-1],fontsize=15, bbox=bbox,
            transform=ax[-1].transAxes, horizontalalignment='right')

fig.tight_layout()
plt.savefig(result_dir+'/delay_power_spectrum_2.pdf',bbox_inches='tight',dpi=300)
'''-----------------------------------------------------------------------------------------------'''


# '''------------------------DPS plots from test cases-----------------------------------------------'''
# result_dir='/nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/'
# run_version_arr = ['high_dl_fr_0','masked_data','filtered_data']
# conf_interval=95
# Nburn = 10
# bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)

# dl_inds=[[3,4,5,6],[10,11,12,13],[3,3,3,3]]
# fig, ax = plt.subplots(4,1,figsize=(25, 20))
# i=0
# colors=['r','b','k']
# fig_labels=['Systematics Model','Masked','Filtered','Residuals']
# for run_version in run_version_arr:
#     eor_true=np.load(result_dir+'high_dl_fr_0/eor_true.npy')
#     ps_sample = np.load(result_dir+run_version+'/dps-eor.npy')
#     ln_post = np.load(result_dir+run_version+'/ln-post.npy')
#     ps_true = calc_ps(eor_true[:Ntimes,:Nfreqs])
#     ps_mean = np.mean(ps_sample, axis=0)
#     df = (freqs[1] - freqs[0]) * u.MHz
#     delays = np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=df.to("1/ns")))

#     sys_delays= delays[np.unique(dl_inds[1])+int(Nfreqs/2)].value
#     if Nburn > 0:
#         ps_sample = ps_sample[Nburn:]
#         ln_post = ln_post[Nburn:]
#     # Posterior-weighted mean delay power spectrum
#     dps_eor_hp_pwm = np.average(ps_sample, weights=ln_post, axis=0)
    
#     # Confidence interval of delay power spectrum posteriors
#     percentile = conf_interval/2 + 50
#     dps_eor_hp_ubound = np.percentile(ps_sample, percentile, axis=0)
#     dps_eor_hp_lbound = np.percentile(ps_sample, 100-percentile, axis=0)
#     dps_eor_hp_err = np.vstack((
#         dps_eor_hp_pwm - dps_eor_hp_lbound,
#         dps_eor_hp_ubound - dps_eor_hp_pwm
#     ))
    
#     ax[i].errorbar(
#                     delays,
#                     dps_eor_hp_pwm,
#                     yerr=np.abs(dps_eor_hp_err),
#                     color=colors[i],
#                     # ls="",
#                     marker="o",
#                     capsize=3,
#                     label=f"Recovered ({conf_interval}% Confidence)"
#                     )
#     ax[i].plot(delays, ps_true, "k:", label="True")

#     ax[i].legend(loc="best",fontsize=20)
#     ax[i].set_ylabel(r"$P(\tau)$ [arb. units]")
#     # ax.set_title("EoR Delay Power Spectrum Comparison")
#     ax[i].set_yscale("log")
#     ax[i].grid()
    
#     for dl in sys_delays:
#         ax[i].axvline(dl,ls='dotted',c=colors[0])
#         ax[-1].axvline(dl,ls='dotted',c=colors[0])
#     ax[i].text(0.95,0.07,fig_labels[i],fontsize=15, bbox=bbox,
#             transform=ax[i].transAxes, horizontalalignment='right')

#     ax[-1].plot(delays,(dps_eor_hp_pwm-ps_true),color=colors[i],label=fig_labels[i])
#     ax[-1].set_ylabel(r"$P(\tau)$ [arb. units]")
#     ax[-1].set_xlabel(r'$\tau$ [ns]')
#     ax[-1].legend()
#     ax[-1].grid()

#     i=i+1
# ax[-1].text(0.95,0.07,fig_labels[-1],fontsize=15, bbox=bbox,
#             transform=ax[-1].transAxes, horizontalalignment='right')

# fig.tight_layout()
# plt.savefig(result_dir+'/masked_filtered_dps_10k.pdf',bbox_inches='tight',dpi=300)
# '''-----------------------------------------------------------------------------------------------'''
