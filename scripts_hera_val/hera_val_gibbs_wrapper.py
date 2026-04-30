"""
Gibbs sampler wrapper for hera_val simulated data.

Loads pre-extracted numpy arrays from res/hera_val_npy/ (produced by
extract_hera_val_data.py), builds the systematic mode operator from the
Gaussian-mask-selected nm_list, assembles all inputs, and sets up the
hp.pspec.gibbs_sample() call.

The sampler call itself is commented out — uncomment to run.
"""

import sys
import os
import numpy as np
from pathlib import Path

_script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_script_dir))

import hydra_pspec as hp

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR  = _script_dir / "res" / "hera_val_npy"
AUX_DIR   = _script_dir / "res" / "gauss_1_data"   # pre-computed noise/fg matrices
OUTPUT_DIR = _script_dir / "outputs" / "hera_val_gibbs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sampler settings ─────────────────────────────────────────────────────────
Niter    = 10000
SEED     = 42
Nfgmodes = 12      # number of Legendre-polynomial foreground modes to use

# ── 1. Load pre-extracted visibility data ────────────────────────────────────
print("Loading data ...")
vis_eor       = np.load(DATA_DIR / "vis_eor.npy")         # (Ntimes, Nfreqs)
vis_fg        = np.load(DATA_DIR / "vis_fg.npy")          # (Ntimes, Nfreqs)
vis_fg_eor    = np.load(DATA_DIR / "vis_fg_eor.npy")      # (Ntimes, Nfreqs)
vis_corrupted = np.load(DATA_DIR / "vis_corrupted.npy")   # (Ntimes, Nfreqs)
freqs_Hz      = np.load(DATA_DIR / "freqs_Hz.npy")        # (Nfreqs,)  Hz
lsts_rad      = np.load(DATA_DIR / "lsts_rad.npy")        # (Ntimes,)  radians
nm_list_arr   = np.load(DATA_DIR / "nm_list.npy")         # (N_modes, 2) int

Ntimes, Nfreqs = vis_corrupted.shape
freqs_MHz      = freqs_Hz / 1e6
times_sec      = lsts_rad * 24.0 / (2.0 * np.pi) * 3600.0   # LST rad → seconds

print(f"  Data shape  : Ntimes={Ntimes}, Nfreqs={Nfreqs}")
print(f"  Freqs range : {freqs_MHz[0]:.2f} – {freqs_MHz[-1]:.2f} MHz")
print(f"  LSTs range  : {lsts_rad[0]:.4f} – {lsts_rad[-1]:.4f} rad")

# ── 2. Systematic mode operator ──────────────────────────────────────────────
# nm_list entries are (nf, nt) = (delay wavenumber, fringe-rate wavenumber).
# These were selected by a bivariate Gaussian mask (center=[36,99], sigma=6,
# threshold=0.2) around the systematic peak in the DL-FR transform of the
# corrupted visibility, reproducing notebook cell 44 of test-3.2.0.ipynb.

nm_list   = [(int(nf), int(nt)) for nf, nt in nm_list_arr]
N_sys     = len(nm_list)
print(f"\nBuilding sys_modes operator for {N_sys} modes ...")

sys_modes = hp.sys_solver.sys_modes(
    freqs_Hz  = freqs_Hz,
    times_sec = times_sec,
    modes     = nm_list,
)
# sys_modes shape: (Nfreqs * Ntimes, N_sys)
print(f"  sys_modes shape: {sys_modes.shape}")

# ── 3. Foreground mode matrix ─────────────────────────────────────────────────
# fgmodes.npy contains all 120 Legendre-polynomial basis vectors; we keep
# only the first Nfgmodes (smoothest FG components, as in run-hydra-pspec.py).
fgmodes_full = np.load(AUX_DIR / "fgmodes.npy")      # (Nfreqs, Nfreqs)
fgmodes      = fgmodes_full[:, :Nfgmodes]             # (Nfreqs, Nfgmodes)
print(f"\nForeground modes: {fgmodes.shape}")

# ── 4. Noise model ────────────────────────────────────────────────────────────
noise_cov = np.load(AUX_DIR / "noise-cov.npy")       # (Nfreqs, Nfreqs)
Ninv      = np.linalg.inv(noise_cov)                  # (Nfreqs, Nfreqs)
print(f"Noise covariance: {noise_cov.shape}, Ninv computed.")

# ── 5. EoR signal power spectrum prior and initial guess ─────────────────────
# Broad flat prior: lower bound 1e-7, upper bound 1e-1 (per-mode variance units)
ps_prior = np.column_stack([
    1e-7 * np.ones(Nfreqs),   # shape (Nfreqs,) lower bound
    1e-1 * np.ones(Nfreqs),   # shape (Nfreqs,) upper bound
])  # shape (Nfreqs, 2) — will be transposed to (2, Nfreqs) in the call

# Initial PS from the diagonal of the pre-computed EoR covariance
eor_cov    = np.load(AUX_DIR / "eor-cov.npy")        # (Nfreqs, Nfreqs)
ps_initial = np.abs(np.diag(eor_cov))                 # (Nfreqs,)
print(f"EoR PS initial: min={ps_initial.min():.3e}, max={ps_initial.max():.3e}")

# ── 6. Systematic amplitude prior and initial guess ──────────────────────────
sys_prior   = 100.0**2 * np.eye(N_sys, dtype=float)   # (N_sys, N_sys)  broad prior
sys_initial = np.ones(N_sys, dtype=complex)            # start at 1

# ── 7. Flags (all channels unflagged) ────────────────────────────────────────
flags = np.ones(Nfreqs, dtype=int)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Gibbs sampler inputs ready:")
print(f"  vis (corrupted)       : {vis_corrupted.shape}")
print(f"  fgmodes               : {fgmodes.shape}")
print(f"  sys_modes             : {sys_modes.shape}")
print(f"  Ninv                  : {Ninv.shape}")
print(f"  ps_initial            : {ps_initial.shape}")
print(f"  ps_prior              : {ps_prior.shape}  (→ transposed in call)")
print(f"  sys_prior             : {sys_prior.shape}")
print(f"  sys_initial           : {sys_initial.shape}")
print(f"  flags                 : {flags.shape}")
print(f"  Niter                 : {Niter}")
print(f"  output dir            : {OUTPUT_DIR}")
print("="*60)

# ── Gibbs sample call (NOT run — uncomment to execute) ───────────────────────

# signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = \
#     hp.pspec.gibbs_sample(
#         vis               = vis_corrupted,       # (Ntimes, Nfreqs) complex
#         flags             = flags,               # (Nfreqs,) int, 1 = unflagged
#         signal_ps_initial = ps_initial,          # (Nfreqs,) initial EoR PS
#         fg_modes          = fgmodes,             # (Nfreqs, Nfgmodes)
#         Ninv              = Ninv,                # (Nfreqs, Nfreqs)
#         signal_ps_prior   = ps_prior.T,          # (2, Nfreqs): [lower, upper]
#         Niter             = Niter,
#         seed              = SEED,
#         freqs             = freqs_MHz,           # (Nfreqs,) in MHz
#         lsts              = lsts_rad,            # (Ntimes,) in radians
#         map_estimate      = False,
#         verbose           = True,
#         nproc             = 1,
#         write_Niter       = Niter,
#         out_dir           = str(OUTPUT_DIR),
#         sys_modes         = sys_modes,           # (Nfreqs*Ntimes, N_sys)
#         sys_prior         = sys_prior,           # (N_sys, N_sys)
#         sys_initial       = sys_initial,         # (N_sys,) complex
#         solver_tol        = 1e-13,
#         sample_systematics= True,
#         sample_eor_fg     = True,
#         sample_signal_ps  = True,
#         sky_model_initial = vis_fg_eor,          # (Ntimes, Nfreqs) initial sky
#     )
