"""
Extract and save visibility data from the hera_val test notebook as numpy arrays.

Reproduces the key data-extraction steps from tools/hera_val/test-3.2.0.ipynb
using local uvh5 files in res/test_data/.

Output arrays are saved to res/hera_val_npy/:
  vis_eor.npy       - EoR-only visibilities        (Ntimes, Nfreqs)
  vis_fg.npy        - Foreground-only visibilities  (Ntimes, Nfreqs)
  vis_fg_eor.npy    - FG + EoR visibilities         (Ntimes, Nfreqs)
  vis_corrupted.npy - FG + EoR + systematics        (Ntimes, Nfreqs)
  freqs_Hz.npy      - Frequency array               (Nfreqs,)  in Hz
  lsts_rad.npy      - LST array                     (Ntimes,)  in radians
  mask_indices.npy  - (time_idx, freq_idx) pairs of selected modes  (2, N_modes)
  nm_list.npy       - (nf, nt) wavenumber integer pairs for sys_modes (N_modes, 2)
"""

import sys
import numpy as np
from pathlib import Path

from pyuvdata import UVData
from uvtools.utils import FFT
from uvtools.dspec import gen_window

_script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_script_dir))

from hydra_pspec.utils import form_pseudo_stokes_vis

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = _script_dir / "res" / "test_data"
OUTPUT_DIR = _script_dir / "res" / "hera_val_npy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = (0, 1)   # baseline used throughout the notebook
POL      = "xx"     # pseudo-Stokes I is stored in XX after form_pseudo_stokes_vis


# ── Helper functions ────────────────────────────────────────────────────────
def load_vis(path, baseline=BASELINE, pol=POL):
    """
    Load a uvh5 file, conjugate baselines to canonical order, form pseudo-Stokes
    I visibilities (XX + YY), and return the data array for the requested baseline.

    Returns
    -------
    ndarray, shape (Ntimes, Nfreqs), complex
    """
    uvd = UVData()
    uvd.read(str(path))
    uvd.conjugate_bls()                 # ensure ant1 < ant2 ordering
    uvd = form_pseudo_stokes_vis(uvd)   # XX += YY, then select XX
    return uvd.get_data((baseline[0], baseline[1], pol), force_copy=True)


def generate_gaussian_mask(data_shape, center, sigma, threshold):
    """
    Boolean mask in a 2D array based on a bivariate Gaussian.

    Returns True where  G(i,j) >= threshold * G(center),  so that the
    selected region covers a compact area in delay–fringe-rate space.

    Parameters
    ----------
    data_shape : (int, int)
        Shape of the 2D array to mask.
    center : [float, float]
        Index-space centre [i0, j0] of the Gaussian.
    sigma : list-of-lists, shape (2, 2)
        Covariance matrix in index space.
    threshold : float
        Fraction of peak amplitude above which pixels are selected.

    Returns
    -------
    mask : ndarray of bool, shape data_shape
    """
    n, m  = data_shape
    cov   = np.array(sigma, dtype=float)
    c_inv = np.linalg.inv(cov)
    c_det = np.linalg.det(cov)
    norm  = 1.0 / (2.0 * np.pi * np.sqrt(c_det))

    # Vectorised evaluation over all (i, j) pairs
    ii, jj = np.mgrid[0:n, 0:m]                  # (n, m) each
    diffs  = np.stack([ii - center[0], jj - center[1]], axis=-1)   # (n, m, 2)
    exp_   = -0.5 * np.einsum("...i,ij,...j->...", diffs, c_inv, diffs)
    gauss  = norm * np.exp(exp_)

    peak = norm   # G at center = norm * exp(0) = norm
    return gauss >= threshold * peak


# ── 1. Load visibility data ─────────────────────────────────────────────────
print("Loading visibility data ...")
vis_eor       = load_vis(DATA_DIR / "vis-eor.uvh5")
vis_fg        = load_vis(DATA_DIR / "vis-ptsrc-gsm.uvh5")
vis_fg_eor    = load_vis(DATA_DIR / "vis-eor-ptsrc-gsm.uvh5")
vis_corrupted = load_vis(DATA_DIR / "vis_corrupted_gauss_1.uvh5")

print(f"  vis_eor:       {vis_eor.shape}")
print(f"  vis_fg:        {vis_fg.shape}")
print(f"  vis_fg_eor:    {vis_fg_eor.shape}")
print(f"  vis_corrupted: {vis_corrupted.shape}")

# ── 2. Frequency and LST metadata ───────────────────────────────────────────
print("\nLoading metadata ...")
_uvd  = UVData()
_uvd.read(str(DATA_DIR / "vis-eor-ptsrc-gsm.uvh5"))
freqs = _uvd.freq_array.flatten()          # (Nfreqs,) in Hz
lsts  = np.unique(_uvd.lst_array)          # (Ntimes,) in radians
print(f"  freqs: {freqs.shape},  "
      f"range [{freqs[0]/1e6:.2f} – {freqs[-1]/1e6:.2f}] MHz")
print(f"  lsts:  {lsts.shape},  "
      f"range [{lsts[0]:.4f} – {lsts[-1]:.4f}] rad")

Ntimes, Nfreqs = vis_corrupted.shape

# ── 3. Gaussian mask in delay–fringe-rate space ──────────────────────────────
# Notebook cells 36–44: apply 2D FFT with Blackman-Harris taper to the
# corrupted visibility, then fit a Gaussian mask around the systematic peak.
print("\nComputing delay–fringe-rate (DL–FR) transform of corrupted data ...")

freq_taper = gen_window("blackman-harris", Nfreqs)[np.newaxis, :]   # (1, Nfreqs)
time_taper = gen_window("blackman-harris", Ntimes)[:, np.newaxis]    # (Ntimes, 1)

# axis-0 FFT → fringe rate,  axis-1 FFT → delay
data_fr_dly = FFT(FFT(vis_corrupted * time_taper, axis=0) * freq_taper, axis=1)

# Gaussian mask parameters (from notebook cell 44)
MASK_CENTER    = [36, 99]
MASK_SIGMA     = [[6, 0], [0, 6]]
MASK_THRESHOLD = 0.2

mask    = generate_gaussian_mask(
    data_fr_dly.shape, MASK_CENTER, MASK_SIGMA, MASK_THRESHOLD
)
indices = np.where(mask)   # tuple of (time_fft_indices, freq_fft_indices)

print(f"  Gaussian mask selected {len(indices[0])} modes")
print(f"  indices[0] range (FR axis): {indices[0].min()} – {indices[0].max()}")
print(f"  indices[1] range (DL axis): {indices[1].min()} – {indices[1].max()}")

# ── 4. Convert FFT indices → wavenumber integers ────────────────────────────
# nfreq[k] = k for k in [0, Nfreqs/2), else k - Nfreqs   (standard fftfreq ordering)
# ntime[k] = k for k in [0, Ntimes/2), else k - Ntimes
nfreq = (np.fft.fftfreq(Nfreqs) * Nfreqs).astype(int)
ntime = (np.fft.fftfreq(Ntimes) * Ntimes).astype(int)

# mode = (nf, nt) = (delay wavenumber, fringe-rate wavenumber)
# indices[0] → fringe-rate (time FFT) axis → ntime
# indices[1] → delay (freq FFT) axis        → nfreq
nm_list = [
    (int(nfreq[fi]), int(ntime[ti]))
    for ti, fi in zip(indices[0], indices[1])
]
print(f"  nm_list has {len(nm_list)} pairs, "
      f"nf in [{min(p[0] for p in nm_list)}, {max(p[0] for p in nm_list)}], "
      f"nt in [{min(p[1] for p in nm_list)}, {max(p[1] for p in nm_list)}]")

# ── 5. Save all arrays ───────────────────────────────────────────────────────
print(f"\nSaving arrays to {OUTPUT_DIR} ...")

np.save(OUTPUT_DIR / "vis_eor.npy",        vis_eor)
np.save(OUTPUT_DIR / "vis_fg.npy",         vis_fg)
np.save(OUTPUT_DIR / "vis_fg_eor.npy",     vis_fg_eor)
np.save(OUTPUT_DIR / "vis_corrupted.npy",  vis_corrupted)
np.save(OUTPUT_DIR / "freqs_Hz.npy",       freqs)
np.save(OUTPUT_DIR / "lsts_rad.npy",       lsts)
np.save(OUTPUT_DIR / "mask_indices.npy",   np.array(indices))        # (2, N_modes)
np.save(OUTPUT_DIR / "nm_list.npy",        np.array(nm_list))        # (N_modes, 2)

print("  Done.\n")
print("Saved files:")
for p in sorted(OUTPUT_DIR.iterdir()):
    a = np.load(p)
    print(f"  {p.name:<30}  shape={str(a.shape):<20}  dtype={a.dtype}")
