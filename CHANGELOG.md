# Changelog

---

## 2026-09 — Warm start (chain resume)

A Gibbs chain can now be stopped and continued from its last recorded sample.
A segmented run is bit-identical to the uninterrupted run it replaces. No
change to the sampler mathematics. Full documentation in
[`docs/warm-start.md`](docs/warm-start.md); the survey it was built from is
[`docs/warm-start-plan.md`](docs/warm-start-plan.md).

### `hydra_pspec/utils.py`

**Added**
- `GIBBS_SAMPLE_DATASETS`, `GIBBS_NPY_FILENAMES`, `RNG_STATE_GROUP` module
  constants naming the six sample datasets, their `.npy` filenames, and the
  HDF5 group holding the RNG state.
- `gibbs_sample_h5_path(fp)`, and an internal `_as_h5` context manager so that
  every reader accepts an output directory, a path to the `.h5`, or an
  already-open `h5py.File` (HDF5's file lock forbids a second handle to a file
  a running chain holds open).
- `h5_chain_length(fp)`, `truncate_h5_to(fp, n)`, `repair_h5_chain(fp)` —
  the chain-length and crash-repair path. A run killed between two of the
  per-dataset resizes of one append leaves the datasets ragged; repairing to
  their common length before reading is what stops every subsequent index
  being silently off by one.
- `read_gibbs_sample_h5(fp, index=-1, names=None)` — the first read path in the
  module. Returns the requested sample plus `chain_length`.
- `save_rng_state(f, chain_length)` / `load_rng_state(fp, chain_length)` —
  persist and restore the MT19937 state, tagged with the chain length it
  belongs to so a state left out of step by a repair is refused rather than
  used.
- `array_hash(arr)`, `write_chain_metadata(f, **kw)`, `read_chain_metadata(fp)`,
  `check_chain_metadata(fp, **kw)` — run provenance stored as `meta_*` HDF5
  attributes, so a resume against different data, priors or systematics basis
  raises instead of quietly sampling the wrong thing.
- `export_npy_from_h5(fp)` — writes the six `.npy` files from the HDF5 chain,
  streaming rows through `np.lib.format.open_memmap` so no full array is held
  in memory (`signal_amps` is 19.2 GB for a 250k-iteration 80x60 run).
- `GibbsSampleH5Writer`: `save_rng` constructor flag, plus the `file` and
  `chain_length` properties and the `truncate_to()`, `repair()`,
  `write_metadata()` and `export_npy()` methods.

**Unchanged**
- `write_numpy_files`, `append_gibbs_sample_h5` and the existing writer
  behaviour. `write_numpy_files` is simply no longer what the sampler calls.

### `hydra_pspec/pspec.py`

**Added**
- `gibbs_sample(..., resume=False, export_npy=True, check_metadata=True)`.
  Under `resume`, `Niter` is the number of *additional* iterations, so a
  resume composes with a job scheduler without the caller knowing the current
  chain length.
- Chain metadata (seed, shapes, sampling toggles, solver, and SHA-256 hashes
  of `vis`, `flags`, `Ninv`, `fg_modes`, `sys_modes`, `sys_prior`,
  `signal_ps_prior`, `sky_model_initial`) written at the start of a run and
  checked before a resume is allowed to append.

**Fixed**
- The HDF5 writer was opened with a hardcoded `overwrite=True`, which deletes
  the file. Now `overwrite=not resume` — a resume would otherwise have
  destroyed the chain it was about to read.
- The resume path restores the saved RNG state instead of calling
  `np.random.seed()`, which would replay the draws the first segment already
  used and correlate the two segments. Falls back to a `seed + start_iter`
  reseed, with a `RuntimeWarning`, when no usable state is stored.
- Both `write_numpy_files` calls replaced by exports from the HDF5. `np.save`
  has fixed filenames and no append path, so a resumed run would otherwise
  have left the `.npy` files holding the last segment in place of the whole
  chain — silently, since all post-processing reads the `.npy` files.

**Changed (no mathematical effect)**
- The two initial-state assertions (`sys_initial` shape, `signal_ps_initial`
  within `signal_ps_prior`) now run against the arrays actually used: the
  caller's arguments on a fresh run, exactly as before; the values read back
  from the chain on a resume.
- The verbose iteration counter prints the global iteration number, so a
  resumed segment continues the previous numbering. Identical for a fresh
  chain.

### `sys_sampler_wrapper.py`

**Added**
- `argparse` CLI: `--resume`, `--Niter`, `--out-dir`, `--no-export-npy`.
- Under `--resume`, `data_true.npy`, `eor_true.npy`, `fg_true.npy` and
  `gain_true.npy` are loaded from the output directory rather than
  regenerated, and the `np.save` calls that would overwrite them are skipped.
  The regeneration is deterministic under `np.random.seed(11)` only while
  `Ntimes`, `Nfreqs` and `dummy_flag` are unchanged; loading removes that
  dependency rather than relying on it.
- The output directory is created if missing.

**Removed**
- Unused imports: `sys`, `pyuvdata.UVData`, `astropy.units`,
  `matplotlib.ticker`, `cmcrameri.cm`. The last of these is not installed in
  the `py10` environment and made the script unimportable there.

### `tests/test_warm_start.py`

**Added**
- The acceptance test from the plan: 10 iterations straight versus 5 plus a
  5-iteration resume, asserting all six outputs are bit-identical. Plus
  coverage for the crash repair, the RNG state round trip, the `.npy` export
  covering the full chain, and the rejection of a resume against mismatched
  data or an empty directory.

---

## 2026-04 — Script and module cleanup

### `sys_solver.py`

**Added**
- Module-level docstring listing all public symbols.
- Full NumPy-style docstrings for every function: `fourier_mode_2d`,
  `sys_modes`, `gcr_systematics`, `sq_mat_tr`, `sq_mat_tr2`, `inv_mat`,
  and `cholesky_inverse`.
- `gcr_systematics`: documented the `Raises` section (both `ValueError`
  paths for solver failure).

**Removed**
- Unused imports: `matplotlib.pylab`, `sklearn.metrics`,
  `scipy.linalg.fractional_matrix_power`, and
  `plotting_functions.master_plotter`.
- Debug `print(nf, nt)` statement inside `fourier_mode_2d` loop.
- `scipy.linalg as sl` alias (retained `scipy.sparse.linalg` which is
  actually used by the solvers).

**Fixed**
- `assert` error messages in `fourier_mode_2d` converted to f-strings.
- `ValueError` messages in `gcr_systematics` converted to f-strings.
- `sq_mat_tr2`: quadrant assignments simplified from four-element index
  notation to two-element slice notation (no logic change).
- `inv_mat`: removed intermediate `diag_el` variable; simplified to a
  single `diag_inv` computation.
- Consistent `===` section banners replacing ad-hoc inline comments.

---

### `sohini_test.py`

**Added**
- Module-level docstring clarifying this is the production 100 k-iteration
  run script and distinguishing it from `sys_sampler_wrapper.py`.
- `calc_ps` docstring with parameters and returns.
- `===` section banners throughout.

**Removed**
- Unused imports: `pylab`, `UVData`, `Quantity`, `units`, `ticker`,
  `cmcrameri`, `sys`.
- All commented-out `op_dir` and `nm_list` alternatives moved to inline
  comments on the active lines (case labels preserved).
- Commented-out uvh5 foreground loading block.
- Commented-out dummy foreground fitting block.
- Redundant `flags_i` variable — inlined directly into `gibbs_sample`.

**Fixed**
- f-strings replace all `.format()` calls.
- `lsts` pre-defined array now passed to `gibbs_sample` instead of
  recomputing `np.linspace(0., 1., Ntimes)` at the call site.
- Consistent spacing and argument alignment in `gibbs_sample` call.

---

## 2026-03-26 — Second version of HERA test notebook

**Added**

- Added new cable reflection notebook in tools/hera_val/
- New notebook loads npy data, creates sky visibilities by doing eor+fg and returns reflection systematics at +/- 1200ns
- Notebooks also plots DL-FR plot for paper

## 2026-03-26 — HERA validation notebook refactor

## 2026-03 — `sohini_test.py` cleanup (first pass)

### `sohini_test.py`

**Fixed**
- Removed spurious `exit()` call (original line 130) that halted the run before
  completing any Gibbs iterations.
- Made all data paths robust via `Path(__file__).parent` — no longer depends on
  the shell's current working directory.
- Added `os.makedirs(op_dir, exist_ok=True)` so the output directory is created
  automatically on first run.
- Passed the pre-defined `lsts` array to `gibbs_sample` instead of recomputing it.

**Removed**
- Unused imports: `pylab`, `UVData`, `Quantity`, `units`, `ticker`, `cmcrameri`, `sys`.

---

## 2026-03 — Package and scripts restructure

### `hydra_pspec/__init__.py`

**Added**
- `sys_solver` to explicit module exports so `from hydra_pspec import sys_solver`
  works without a separate import.

### `scripts/` (new directory)

**Moved**
- `hydra_pspec/calc-vis-cov-matrices.py` → `scripts/calc-vis-cov-matrices.py`.
  The hyphenated filename made it non-importable as a package member.

**Added**
- `scripts/extract_hera_val_data.py` — loads `res/test_data/*.uvh5`, forms
  pseudo-Stokes I, computes the 2D delay–fringe-rate transform, fits a Gaussian
  mask to identify systematic modes, and writes all pre-processed arrays to
  `res/hera_val_npy/`.
- `scripts/hera_val_gibbs_wrapper.py` — Gibbs sampler wrapper configured for
  the HERA validation data in `res/hera_val_npy/`, with all parameter choices
  documented inline.

---

## 2026-02-16 — February 2026 update

General development and run updates. See commit `629730e`.

---

## 2025-09-25 — Sanity save

See commit `985f1b0`.

---

## 2025 — Core development milestones

See `git log` for full details. Selected milestones:

| Commit | Description |
|---|---|
| `3c5598d`, `339b6a3`, `95c25e2` | Fixed critical reshape bug in `sys_modes()` in `sys_solver.py` |
| `238235b` | Final run version before paper |
| `5d98bb9` | Running with bigger priors and fitted foregrounds |
| `4379ea2` | Added masked-data and filtered-data run modes |
| `009d108` | Results after 1 k-iteration test runs |
| `ae9560b` | Saving changes after systematic gain model integration |
| `d3b5178` | Added plotting notebook |
| `28cc1bf` | Repository restructure |
| `b0b592a` | Added notebooks from Andromeda cluster |