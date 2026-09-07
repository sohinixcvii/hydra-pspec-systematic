
import numpy as np
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from scipy.signal.windows import blackmanharris
from astropy import units
from astropy.units import Quantity
import ast
import subprocess
import os
import shutil
from pathlib import Path
import h5py
import hashlib
import warnings
from contextlib import contextmanager
from datetime import datetime

# Names of the per-iteration sample datasets written by the Gibbs sampler, in
# the order gibbs_sample() appends them. Anything else stored in the HDF5 file
# (the RNG state group, chain metadata attributes) is bookkeeping and must be
# excluded from chain-length and truncation logic.
GIBBS_SAMPLE_DATASETS = (
    "signal_amps",
    "signal_ps",
    "fg_amps",
    "sys_amps",
    "chisq",
    "ln_post",
)

# Mapping from HDF5 dataset name to the .npy filename written by
# write_numpy_files() / export_npy_from_h5().
GIBBS_NPY_FILENAMES = {
    "signal_amps": "gcr-eor.npy",
    "signal_ps": "dps-eor.npy",
    "fg_amps": "fg-amps.npy",
    "sys_amps": "b-sys.npy",
    "chisq": "chisq.npy",
    "ln_post": "ln-post.npy",
}

# Name of the HDF5 group holding the persisted NumPy global RNG state.
RNG_STATE_GROUP = "rng_state"

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


def naive_pspec(data, subtract_mean=True, taper=True):
    """
	Compute the naive power spectrum of some data, by calculating the 
	product of the FFT'd data with its complex conjugate.

	Parameters:
		data (aray_like):
			Array of complex data to compute the power spectrum of.
		subtract_mean (bool):
			If True, subtract the mean of the data before calculating the 
			power spectrum.
		taper (bool):
			If True, apply a Blackman-Harris taper to the data before 
			computing the power spectrum.

	Returns:
		ps (array_like):
			Complex-valued power spectrum, with fftshift applied.
    """
    if len(data.shape) == 1:
        Nfreqs = data.size
    elif len(data.shape) == 2:
        Nfreqs = data.shape[1]

    if subtract_mean:
        d = data - np.mean(data, axis=1)[:,np.newaxis]
    
    if taper:
        d *= blackmanharris(Nfreqs)
        
    return np.fft.fftshift(abs(np.fft.fft(d))**2)

# Make a power spectrum
def calc_ps(s):
    # NOTE: This uses inverse FFT instead of FFT to get the right normalisation
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs # CHECK: This takes an average

def trim_flagged_channels(w, x):
    """
    Remove flagged channels from a 1D or 2D (square) array. This is 
    a necessary pre-processing step for LSSA.

    Parameters:
        w (array_like):
            1D array of mask values, where 1 means unmasked and 0 means 
            masked.
        
        x (array_like):
            1D or square 2D array to remove the masked channels from.

    Returns:
        xtilde (array_like):
            Input array with the flagged channels removed.
    """
    # Check inputs
    assert np.shape(x) == (w.size,) or np.shape(x) == (w.size, w.size), \
             "Input array must have shape (w.size) or (w.size, w.size)"

    # 1D case
    if len(x.shape) == 1:
        return x[w == 1.]
    else:
        return x[:,w == 1.][w == 1.,:]


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


def filter_freqs(freq_str, freqs_in):
    """
    Returns a subset of `freqs_in` based on the frequency info in `freq_str`.

    Parameters
    ----------
    freq_str : str
        Can be either a single frequency, a comma delimited list of frequencies
        (e.g. '100,110.4,150'), or a minimum and maximum frequency joined by
        '-' (e.g. '100-200.3').  Cannot contain spaces.  Frequencies are
        assumed to be in MHz.  If specifying individual frequencies and the
        specified frequency is not explicitly present in `freqs_in`, the
        closest frequency in `freqs_in` will be kept.
    freqs_in : array-like
        Frequencies in MHz in data to be filtered.

    Returns
    -------
    freqs_out : astropy.units.Quantity
        Masked frequency array containing only the frequencies from `freqs_in`
        that match `freq_str`.

    """
    if not isinstance(freqs_in, Quantity):
        freqs_in = Quantity(freqs_in, unit="MHz")
    else:
        freqs_in = freqs_in.to("MHz")
    freqs_in_range_str = (
        f"{freqs_in.min().value:.2f} - {freqs_in.max().value:.2f} MHz"
    )

    if '-' in freq_str:
        min_freq, max_freq = freq_str.split('-')
        min_freq = ast.literal_eval(min_freq) * units.MHz
        max_freq = ast.literal_eval(max_freq) * units.MHz
        freq_mask = np.logical_and(
            freqs_in >= min_freq, freqs_in <= max_freq
        )
        if np.sum(freq_mask) == 0:
            print(
                f"Frequency range {freq_str} MHz outside of the frequencies in"
                f" `freqs_in`, {freqs_in_range_str}."
            )
    else:
        if ',' in freq_str:
            freqs = [ast.literal_eval(freq) for freq in freq_str.split(',')]
        else:
            freqs = [ast.literal_eval(freq_str)]
        freqs = Quantity(freqs, unit='MHz')
        freqs_in_range = np.array(
            [freqs_in.min() <= freq <= freqs_in.max() for freq in freqs],
            dtype=bool
        )
        if not np.all(freqs_in_range):
            print(
                f"Frequency(ies) {freqs[~freqs_in_range]} are not within the "
                f"range of frequencies in `freqs_in`, {freqs_in_range_str}."
            )
        freqs_inds = [np.argmin(np.abs(freqs_in - freq)) for freq in freqs]
        freq_mask = np.zeros(freqs_in.size, dtype=bool)
        freq_mask[freqs_inds] = True

    freqs_out = freqs_in[freq_mask]

    return freqs_out


def get_git_version_info(directory=None):
    """
    Get git version info from repository in `directory`.

    Parameters
    ----------
    directory : str
        Path to GitHub repository.  If None, uses one directory up from
        __file__.

    Returns
    -------
    version_info : dict
        Dictionary containing git hash information.

    """
    cwd = os.getcwd()
    if directory is None:
        directory = Path(__file__).parent
    os.chdir(directory)

    version_info = {}
    version_info['git_origin'] = subprocess.check_output(
        ['git', 'config', '--get', 'remote.origin.url'],
        stderr=subprocess.STDOUT)
    version_info['git_hash'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        stderr=subprocess.STDOUT)
    version_info['git_description'] = subprocess.check_output(
        ['git', 'describe', '--dirty', '--tag', '--always'])
    version_info['git_branch'] = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        stderr=subprocess.STDOUT)
    for key in version_info.keys():
        version_info[key] = version_info[key].decode('utf8').strip('\n')
    
    os.chdir(cwd)

    return version_info


def add_mtime_to_filepath(fp, join_char="-"):
    """
    Appends the mtime to a filename or directory before the file suffix.

    Modifies the existing file on disk.

    Parameters
    ----------
    fp : str or Path
        Path to file or directory.
    join_char : str
        Character used to append mtime to filename or directory.  
        Defaults to '-'.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    mtime = datetime.fromtimestamp(os.path.getmtime(fp))
    mtime = mtime.isoformat()
    if fp.is_file():
        fp.rename(fp.with_stem(f"{fp.stem}{join_char}{mtime}"))
    elif fp.is_dir():
        shutil.move(
            fp,
            fp.with_name(f"{fp.name}{join_char}{mtime}")
        )



def write_numpy_files(
    fp,
    signal_amps,
    signal_ps,
    fg_amps,
    sys_amps,
    chisq,
    ln_post
):
    """
    Write sampling arrays to disk as numpy files.

    Parameters
    ----------
    fp : str or Path
        Output directory for files.
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
    b_sys (array_like):
        Samples of the systematics amplitudes, shape `(Niter,len(nm_list))
    chisq (array_like):
        Chi-squared value per iteration, shape `(Niter, Ntimes, Nfreqs)`.
    ln_post (array_like):
        Natural log of the posterior probability per iteration, shape
        `(Niter,)`.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    np.save(fp / f"gcr-eor.npy", signal_amps)
    np.save(fp / f"dps-eor.npy", signal_ps)
    np.save(fp / f"fg-amps.npy", fg_amps)
    np.save(fp / f"b-sys.npy", sys_amps)    
    np.save(fp / f"chisq.npy", chisq)
    np.save(fp / f"ln-post.npy", ln_post)


def _open_gibbs_sample_h5(fp, overwrite=False):
    """
    Resolve the output path and open ``fp/'gibbs_samples.h5'`` for appending.

    Shared by :func:`append_gibbs_sample_h5` and :class:`GibbsSampleH5Writer`
    so that both use identical path, truncation and open semantics.

    Parameters
    ----------
    fp : str or Path
        Output directory (creates fp/'gibbs_samples.h5').
    overwrite : bool
        If True and file exists, delete it (use at the start of a run).

    Returns
    -------
    h5py.File
        File opened in append mode. The caller is responsible for closing it.
    """
    fp = Path(fp)
    fp.mkdir(parents=True, exist_ok=True)
    h5_path = fp / "gibbs_samples.h5"

    if overwrite and h5_path.exists():
        os.remove(h5_path)

    return h5py.File(h5_path, "a")


def _append_arrays_to_h5(f, batch_axis=None, **arrays):
    """
    Append samples to datasets of an already-open HDF5 file, creating them on
    first use.

    This is the shared body of :func:`append_gibbs_sample_h5`; see that function
    for the semantics of `batch_axis` and `arrays`.

    Parameters
    ----------
    f : h5py.File
        File open in a writable mode.
    batch_axis : None or int
        If int, the axis in each array that indexes multiple samples to append.
    **arrays : name=array_like
        Per-quantity sample(s). Shapes must be consistent across calls.
    """
    for name, arr in arrays.items():
        arr = np.asarray(arr)

        # Arrange as (B, ...) where B=number of samples to append this call
        if batch_axis is None:
            batch = arr[np.newaxis, ...]   # (1, ...)
            per_sample_shape = arr.shape
            dtype = arr.dtype
        else:
            batch = np.moveaxis(arr, batch_axis, 0)  # (B, ...)
            per_sample_shape = batch.shape[1:]
            dtype = batch.dtype

        # Create dataset on first sight
        if name not in f:
            f.create_dataset(
                name,
                shape=(0,) + per_sample_shape,
                maxshape=(None,) + per_sample_shape,
                dtype=dtype,
                chunks=(max(1, min(32, batch.shape[0])),) + per_sample_shape,
                compression="gzip"
            )

        dset = f[name]

        # Validate shape consistency
        if dset.shape[1:] != per_sample_shape:
            raise ValueError(
                f"{name}: incoming per-sample shape {per_sample_shape} "
                f"does not match existing {dset.shape[1:]}."
            )

        # Append rows
        i0 = dset.shape[0]
        i1 = i0 + batch.shape[0]
        dset.resize(i1, axis=0)
        dset[i0:i1, ...] = batch


class GibbsSampleH5Writer:
    """
    Context manager that holds ``fp/'gibbs_samples.h5'`` open across many
    appends.

    Functionally identical to calling :func:`append_gibbs_sample_h5` once per
    sample -- same file path, dataset names, shapes, dtypes, chunking, gzip
    compression and append order -- but it avoids reopening and closing the
    file on every sample, which dominates the per-sample cost in long chains.

    The file is still flushed after every append (when `flush` is True), so a
    chain that is killed part-way through leaves a readable file containing
    every sample appended so far, exactly as before.

    Note that while the writer is open the file is held by HDF5's file lock, so
    other processes cannot open it for reading until the chain finishes.

    Parameters
    ----------
    fp : str or Path
        Output directory (creates fp/'gibbs_samples.h5').
    overwrite : bool
        If True and file exists, delete it on open (use at the start of a run).
    flush : bool
        Flush file to disk after each append.
    batch_axis : None or int
        Default `batch_axis` for :meth:`append`; see
        :func:`append_gibbs_sample_h5`.
    save_rng : bool
        If True, persist the NumPy global RNG state alongside every append
        (see :func:`save_rng_state`). Required for an exact warm start; adds
        one ~2.5 kB dataset write per sample.

    Examples
    --------
    >>> with GibbsSampleH5Writer(out_dir, overwrite=True) as writer:
    ...     for i in range(Niter):
    ...         writer.append(signal_ps=signal_ps[i], ln_post=ln_post[i])
    """

    def __init__(self, fp, overwrite=False, flush=True, batch_axis=None,
                 save_rng=False):
        self.flush = flush
        self.batch_axis = batch_axis
        self.save_rng = save_rng
        self._f = _open_gibbs_sample_h5(fp, overwrite=overwrite)

    @property
    def file(self):
        """The underlying open :class:`h5py.File` (None once closed)."""
        return self._f

    def append(self, batch_axis=None, **arrays):
        """
        Append one sample (or a batch) to the open file.

        Parameters
        ----------
        batch_axis : None or int
            Overrides the writer's default `batch_axis` when not None.
        **arrays : name=array_like
            Per-quantity sample(s). Shapes must be consistent across calls.
        """
        if batch_axis is None:
            batch_axis = self.batch_axis
        _append_arrays_to_h5(self._f, batch_axis=batch_axis, **arrays)
        if self.save_rng:
            # Recorded after the append so the state is the one the *next*
            # iteration will start from, tagged with the chain length it
            # belongs to (see load_rng_state).
            save_rng_state(self._f, chain_length=self.chain_length)
        if self.flush:
            self._f.flush()

    @property
    def chain_length(self):
        """Number of complete samples currently in the file (see
        :func:`h5_chain_length`)."""
        return h5_chain_length(self._f)

    def truncate_to(self, n):
        """
        Truncate every sample dataset to `n` rows (see
        :func:`truncate_h5_to`).
        """
        return truncate_h5_to(self._f, n)

    def repair(self, verbose=True):
        """
        Bring all sample datasets to a common length (see
        :func:`repair_h5_chain`).
        """
        return repair_h5_chain(self._f, verbose=verbose)

    def write_metadata(self, **metadata):
        """Record chain provenance (see :func:`write_chain_metadata`)."""
        write_chain_metadata(self._f, **metadata)
        if self.flush:
            self._f.flush()

    def export_npy(self, out_dir=None, chunk=256, verbose=False):
        """
        Export the chain so far to ``.npy`` files through the open handle.

        Uses the writer's own file object because HDF5's file lock prevents
        opening a second handle to the same file while the chain is running.
        See :func:`export_npy_from_h5`.
        """
        if self.flush:
            self._f.flush()
        return export_npy_from_h5(
            self._f, out_dir=out_dir, chunk=chunk, verbose=verbose
        )

    def close(self):
        """Close the underlying file. Idempotent."""
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def append_gibbs_sample_h5(fp, overwrite=False, flush=True,batch_axis=None, **arrays):
    """
    Append Gibbs samples to an HDF5 file, creating datasets on first use.

    - If batch_axis is None (default): each **array** must be ONE sample
      (e.g., (n,m)), and this appends 1 row.
    - If batch_axis is an int (e.g., 0): treats that axis as a batch of samples
      (e.g., (B,n,m)) and appends B rows at once.

    Complex dtypes are preserved.

    This opens and closes the file on every call. To append many samples in a
    loop, use :class:`GibbsSampleH5Writer`, which produces an identical file
    while keeping the file open.

    Parameters
    ----------
    fp : str or Path
        Output directory (creates fp/'gibbs_samples.h5').
    overwrite : bool
        If True and file exists, delete it (use at the start of a run).
    flush : bool
        Flush file to disk after writing.
    batch_axis : None or int
        If int, the axis in each array that indexes multiple samples to append.
    **arrays : name=array_like
        Per-quantity sample(s). Shapes must be consistent across calls.
    """
    with _open_gibbs_sample_h5(fp, overwrite=overwrite) as f:
        _append_arrays_to_h5(f, batch_axis=batch_axis, **arrays)

        if flush:
            f.flush()


# =============================================================================
# Chain resume (warm start) support
#
# The Gibbs chain state is just the pair (signal_ps, sys_amps) of the last
# iteration, so a chain can be continued from `gibbs_samples.h5` provided that
#
#   * the file is not truncated on open (see GibbsSampleH5Writer.overwrite),
#   * datasets desynchronised by a crash mid-append are repaired first,
#   * the NumPy global RNG state is restored rather than reseeded, and
#   * the resumed run is validated against the metadata of the original run.
#
# The helpers below provide those four pieces. See docs/warm-start.md.
# =============================================================================


@contextmanager
def _as_h5(fp, mode="r"):
    """
    Yield an open :class:`h5py.File`, accepting either a path or an open file.

    An already-open file is yielded unchanged and is *not* closed on exit, so
    the same helper works both from inside a running chain (where
    :class:`GibbsSampleH5Writer` holds the file open and the HDF5 lock
    prevents a second handle) and from a separate process.

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory containing ``gibbs_samples.h5``, a path to the file
        itself, or an already-open file.
    mode : str
        Mode used when `fp` is a path.

    Yields
    ------
    h5py.File
    """
    if isinstance(fp, h5py.File):
        yield fp
        return

    yield_path = gibbs_sample_h5_path(fp)
    f = h5py.File(yield_path, mode)
    try:
        yield f
    finally:
        f.close()


def gibbs_sample_h5_path(fp):
    """
    Resolve `fp` to the path of the Gibbs sample HDF5 file.

    Parameters
    ----------
    fp : str or Path
        Either an output directory (``fp/'gibbs_samples.h5'`` is returned) or
        the path of the ``.h5`` file itself (returned unchanged).

    Returns
    -------
    Path
        Path to the HDF5 sample file. Not guaranteed to exist.
    """
    fp = Path(fp)
    if fp.suffix in (".h5", ".hdf5"):
        return fp
    return fp / "gibbs_samples.h5"


def _sample_dataset_names(f):
    """
    Names of the per-iteration sample datasets present in an open HDF5 file.

    Returns them in :data:`GIBBS_SAMPLE_DATASETS` order, skipping any that a
    given file does not contain, and ignoring bookkeeping groups such as the
    RNG state.

    Parameters
    ----------
    f : h5py.File
        Open file.

    Returns
    -------
    list of str
    """
    names = [n for n in GIBBS_SAMPLE_DATASETS if isinstance(f.get(n), h5py.Dataset)]
    # Include any additional top-level datasets so that files written by
    # callers passing other quantities are still handled consistently.
    names += [
        n for n in f
        if n not in GIBBS_SAMPLE_DATASETS and isinstance(f.get(n), h5py.Dataset)
    ]
    return names


def h5_chain_length(fp):
    """
    Number of complete samples in a Gibbs sample HDF5 file.

    The length is the *minimum* over the sample datasets. A chain killed
    between two of the per-dataset resizes done by one append leaves the
    datasets with differing lengths; only the rows below this minimum are
    known to belong to the same iteration.

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory, path to the ``.h5`` file, or an open file.

    Returns
    -------
    int
        Number of complete samples. 0 if the file does not exist or holds no
        sample datasets.
    """
    if not isinstance(fp, h5py.File) and not gibbs_sample_h5_path(fp).exists():
        return 0

    with _as_h5(fp, mode="r") as f:
        names = _sample_dataset_names(f)
        if not names:
            return 0
        return int(min(f[n].shape[0] for n in names))


def truncate_h5_to(fp, n):
    """
    Truncate every sample dataset of a Gibbs sample HDF5 file to `n` rows.

    Used to repair a file whose datasets desynchronised because a crash landed
    between two of the per-dataset resizes of a single append. Datasets already
    at or below `n` rows are left alone.

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory, path to the ``.h5`` file, or an open file in a
        writable mode.
    n : int
        Number of rows to keep.

    Returns
    -------
    int
        `n`, for convenience.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if not isinstance(fp, h5py.File) and not gibbs_sample_h5_path(fp).exists():
        return n

    with _as_h5(fp, mode="a") as f:
        for name in _sample_dataset_names(f):
            if f[name].shape[0] > n:
                f[name].resize(n, axis=0)
        f.flush()
    return n


def repair_h5_chain(fp, verbose=True):
    """
    Bring all sample datasets of a Gibbs sample HDF5 file to a common length.

    Equivalent to ``truncate_h5_to(fp, h5_chain_length(fp))``. This must be
    called before reading the last sample of a chain that may have been killed
    mid-append: without it the datasets stay desynchronised and every
    subsequent index is silently offset for the rest of the chain.

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory, path to the ``.h5`` file, or an open file in a
        writable mode.
    verbose : bool
        Print a message when rows are actually discarded.

    Returns
    -------
    int
        The common chain length after repair.
    """
    if not isinstance(fp, h5py.File) and not gibbs_sample_h5_path(fp).exists():
        return 0

    with _as_h5(fp, mode="a") as f:
        names = _sample_dataset_names(f)
        if not names:
            return 0
        lengths = {n: f[n].shape[0] for n in names}
        n = int(min(lengths.values()))
        if verbose and max(lengths.values()) > n:
            ragged = {k: v for k, v in lengths.items() if v > n}
            print(
                f"Repairing partially written HDF5 chain: truncating {ragged} "
                f"to {n} samples."
            )
        truncate_h5_to(f, n)
    return n


def read_gibbs_sample_h5(fp, index=-1, names=None):
    """
    Read one sample from a Gibbs sample HDF5 file.

    Does *not* repair a desynchronised file; call :func:`repair_h5_chain`
    first if the chain may have been interrupted mid-append.

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory, path to the ``.h5`` file, or an open file.
    index : int
        Index of the sample to read. Negative indices count from the end, so
        the default -1 is the most recent sample.
    names : iterable of str or None
        Datasets to read. Default (None) reads every sample dataset present.

    Returns
    -------
    dict
        ``{dataset_name: array}`` for each requested dataset, plus
        ``'chain_length'``: the total number of complete samples in the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    IndexError
        If the file holds no samples, or `index` is out of range.
    """
    h5_path = None
    if not isinstance(fp, h5py.File):
        h5_path = gibbs_sample_h5_path(fp)
        if not h5_path.exists():
            raise FileNotFoundError(f"No Gibbs sample file at {h5_path}")

    with _as_h5(fp, mode="r") as f:
        available = _sample_dataset_names(f)
        n = int(min(f[name].shape[0] for name in available)) if available else 0
        if n == 0:
            raise IndexError(f"{h5_path or f.filename} contains no samples")

        i = index + n if index < 0 else index
        if not 0 <= i < n:
            raise IndexError(
                f"sample index {index} out of range for a chain of length {n}"
            )

        if names is None:
            names = available
        sample = {name: f[name][i] for name in names}

    sample["chain_length"] = n
    return sample


def save_rng_state(f, chain_length=None, state=None):
    """
    Persist the NumPy global RNG state into an open HDF5 file.

    Stores the 624-element MT19937 key array as a dataset and the stream
    position and cached-Gaussian fields as attributes, so that
    :func:`load_rng_state` can restore the stream exactly. A resumed chain that
    restores this state produces draws identical to those an uninterrupted run
    would have made; reseeding instead replays the draws already used, which is
    a real correlation artefact in the posterior samples.

    Parameters
    ----------
    f : h5py.File
        File open in a writable mode.
    chain_length : int or None
        Number of samples the file holds at the moment the state is recorded.
        Stored alongside it so that :func:`load_rng_state` can refuse a state
        that does not line up with the (possibly repaired) chain length.
    state : tuple or None
        RNG state tuple as returned by ``np.random.get_state()``. Default
        (None) captures the current global state.
    """
    if state is None:
        state = np.random.get_state()

    bit_generator, keys, pos, has_gauss, cached_gaussian = state

    grp = f.require_group(RNG_STATE_GROUP)
    if "keys" in grp:
        grp["keys"][...] = keys
    else:
        grp.create_dataset("keys", data=np.asarray(keys, dtype=np.uint32))

    grp.attrs["bit_generator"] = bit_generator
    grp.attrs["pos"] = int(pos)
    grp.attrs["has_gauss"] = int(has_gauss)
    grp.attrs["cached_gaussian"] = float(cached_gaussian)
    if chain_length is not None:
        grp.attrs["chain_length"] = int(chain_length)


def load_rng_state(fp, chain_length=None, apply=True):
    """
    Restore the NumPy global RNG state previously saved by
    :func:`save_rng_state`.

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory, path to the ``.h5`` file, or an open file.
    chain_length : int or None
        If given, the state is only accepted when it was recorded at exactly
        this chain length. A mismatch means the file was truncated by
        :func:`repair_h5_chain` after the state was written (or the state was
        written before the last append completed), so the state is one
        iteration out of step with the samples and cannot be used for an exact
        continuation.
    apply : bool
        If True (default), call ``np.random.set_state()`` with the loaded
        state.

    Returns
    -------
    tuple or None
        The restored state tuple, or None if no usable state was found. The
        caller is responsible for choosing a fallback (see the `resume`
        handling in :func:`hydra_pspec.pspec.gibbs_sample`).
    """
    if not isinstance(fp, h5py.File) and not gibbs_sample_h5_path(fp).exists():
        return None

    with _as_h5(fp, mode="r") as f:
        if RNG_STATE_GROUP not in f:
            return None
        grp = f[RNG_STATE_GROUP]

        if chain_length is not None and "chain_length" in grp.attrs:
            saved_n = int(grp.attrs["chain_length"])
            if saved_n != chain_length:
                warnings.warn(
                    f"Saved RNG state was recorded at chain length {saved_n} "
                    f"but the chain holds {chain_length} samples; the state is "
                    "out of step with the samples and will not be used.",
                    RuntimeWarning,
                )
                return None

        bit_generator = grp.attrs.get("bit_generator", "MT19937")
        if isinstance(bit_generator, bytes):
            bit_generator = bit_generator.decode()

        state = (
            str(bit_generator),
            np.asarray(grp["keys"][...], dtype=np.uint32),
            int(grp.attrs["pos"]),
            int(grp.attrs["has_gauss"]),
            float(grp.attrs["cached_gaussian"]),
        )

    if apply:
        np.random.set_state(state)
    return state


def array_hash(arr):
    """
    Stable SHA-256 hash of an array's dtype, shape and bytes.

    Used to detect a resume pointed at a chain that was run against different
    data, without storing the data itself in the sample file.

    Parameters
    ----------
    arr : array_like or None

    Returns
    -------
    str
        Hex digest, or ``'none'`` if `arr` is None.
    """
    if arr is None:
        return "none"
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def write_chain_metadata(f, **metadata):
    """
    Record chain provenance as attributes of an open HDF5 file.

    Values are stored under the ``meta_`` prefix so they cannot collide with
    dataset names. None is stored as the string ``'none'``; sequences are
    stored via ``repr`` so that heterogeneous entries such as ``nm_list``
    survive the round trip.

    Parameters
    ----------
    f : h5py.File
        File open in a writable mode.
    **metadata : name=value
        Scalars, strings or short sequences describing the run.
    """
    for key, value in metadata.items():
        if value is None:
            value = "none"
        elif isinstance(value, (list, tuple, dict)):
            value = repr(value)
        elif isinstance(value, np.generic):
            value = value.item()
        f.attrs[f"meta_{key}"] = value


def read_chain_metadata(fp):
    """
    Read the provenance attributes written by :func:`write_chain_metadata`.

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory, path to the ``.h5`` file, or an open file.

    Returns
    -------
    dict
        ``{name: value}`` with the ``meta_`` prefix stripped. Empty if the file
        does not exist or predates metadata support.
    """
    if not isinstance(fp, h5py.File) and not gibbs_sample_h5_path(fp).exists():
        return {}

    with _as_h5(fp, mode="r") as f:
        meta = {}
        for key, value in f.attrs.items():
            if not key.startswith("meta_"):
                continue
            if isinstance(value, bytes):
                value = value.decode()
            elif isinstance(value, np.generic):
                value = value.item()
            meta[key[len("meta_"):]] = value
    return meta


def check_chain_metadata(fp, strict=True, **metadata):
    """
    Verify that a resume is being run against the chain it belongs to.

    Compares the values in `metadata` with those recorded by
    :func:`write_chain_metadata`. Keys absent from the file (a chain written
    before metadata support) are reported but do not fail the check, since
    there is nothing to compare against.

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory, path to the ``.h5`` file, or an open file.
    strict : bool
        If True (default), raise on any mismatch. If False, warn instead.
    **metadata : name=value
        The same keyword values that :func:`write_chain_metadata` was given at
        the start of the original run.

    Returns
    -------
    list of str
        Human-readable descriptions of the mismatches found (empty if none).

    Raises
    ------
    ValueError
        If `strict` and any recorded value disagrees.
    """
    stored = read_chain_metadata(fp)
    if not stored:
        warnings.warn(
            "Gibbs sample file records no chain metadata; a resume cannot be "
            "validated against the original run's configuration.",
            RuntimeWarning,
        )
        return []

    mismatches = []
    for key, value in metadata.items():
        if key not in stored:
            continue
        if value is None:
            value = "none"
        elif isinstance(value, (list, tuple, dict)):
            value = repr(value)
        elif isinstance(value, np.generic):
            value = value.item()
        if stored[key] != value:
            mismatches.append(f"{key}: chain has {stored[key]!r}, run has {value!r}")

    if mismatches:
        msg = (
            "Resume configuration does not match the chain being resumed:\n  "
            + "\n  ".join(mismatches)
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)

    return mismatches


def export_npy_from_h5(fp, out_dir=None, chunk=256, verbose=False):
    """
    Write the ``.npy`` sampler outputs as a derived export of the full HDF5
    chain.

    The ``.npy`` files are fixed filenames written with ``np.save`` and have no
    append path, so writing them from the in-memory arrays of a resumed run
    would replace the whole chain with the latest segment only — silently,
    since every post-processing entry point reads the ``.npy`` files rather
    than the HDF5. Exporting from the HDF5 instead keeps them a faithful copy
    of the complete chain.

    Rows are streamed through ``np.lib.format.open_memmap`` in blocks of
    `chunk` samples, so the export never holds a full array in memory (the
    ``signal_amps`` array of a 250k-iteration 80x60 chain is 19.2 GB).

    Parameters
    ----------
    fp : str or Path or h5py.File
        Output directory, path to the ``.h5`` file, or an open file.
    out_dir : str or Path or None
        Directory to write the ``.npy`` files to. Default (None) is the
        directory containing the HDF5 file.
    chunk : int
        Number of samples copied per block.
    verbose : bool
        Print one line per exported file.

    Returns
    -------
    dict
        ``{dataset_name: Path}`` for each file written.

    Notes
    -----
    Filenames follow :data:`GIBBS_NPY_FILENAMES`, matching
    :func:`write_numpy_files`.
    """
    if out_dir is None:
        if isinstance(fp, h5py.File):
            out_dir = Path(fp.filename).parent
        else:
            out_dir = gibbs_sample_h5_path(fp).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    with _as_h5(fp, mode="r") as f:
        names = _sample_dataset_names(f)
        if not names:
            return written
        n = int(min(f[name].shape[0] for name in names))

        for name in names:
            fname = GIBBS_NPY_FILENAMES.get(name, f"{name}.npy")
            path = out_dir / fname
            dset = f[name]
            arr = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=dset.dtype,
                shape=(n,) + dset.shape[1:],
            )
            for i0 in range(0, n, chunk):
                i1 = min(i0 + chunk, n)
                arr[i0:i1] = dset[i0:i1]
            arr.flush()
            del arr
            written[name] = path
            if verbose:
                print(f"  exported {n} samples to {path}")

    return written
